#!/usr/bin/env python3
"""四通道石头/布二分类完整流程（Python >=3.9）。

把本文件及00、01、02、03、07二分类脚本放在同一文件夹。
安装：python -m pip install numpy pandas scipy scikit-learn joblib pylsl
采集另需PsychoPy；已有三分类原始数据可先用，不必立即重新采集。

首次：init --legacy-base 旧三分类base_model.joblib --today-input 新一天TXT
  只读取旧模型旁training_config.json中的历史数据路径，不使用旧SVM权重。
  也可用 --history-data 指定旧01预处理目录，或 --history-input 多个历史TXT。
以后：day --base-model 新二分类base_model.joblib --today-input 新一天TXT
实时：live --run 本次输出目录（终端会打印完整命令）

Marker11石头 -> class0；Marker13布 -> class1；Marker12剪刀不参与拟合。
Run1拟合，Run2选择权重/门控，Run1+2最终拟合，Run3+4独立测试。
1.3.0实时修订：仅测当前静息；自由做动作，由冻结SVM判断，再点击实际动作保存统计。
1.3.1：live可显式传入--activity-on-z与--activity-off-z；默认6/3，调参只影响活动门控。
1.3.2：可选--live-profile responsive，容忍短暂活动回落并限制投票历史年龄；不会超时强制输出。
1.4.0：live添加--trigger-mode manual，点击开始后做动作，固定采集4秒后判断，再标注实际动作。
离线回放继续使用冻结的原门控。要比较原规则，live添加--live-profile replay --activity-mode legacy。
手工动作标注只评估，不训练、不重置检测器。减为两类不能修复上游流积压。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import html

HERE = Path(__file__).resolve().parent
SCRIPTS = {
    'preprocess': '01_preprocess_rp_binary_4ch.py',
    'train': '02_train_base_rp_binary_4ch.py',
    'calibrate': '03_calibrate_rp_binary_4ch.py',
    'live': '07_live_rp_binary_label_4ch.py',
}
INPUT_SCHEMA = 'openbci_rp_binary_4ch_preprocess_v1'


def module(key):
    path = HERE/SCRIPTS[key]
    if not path.is_file():
        raise ValueError('请将配套二分类脚本放在同一文件夹：%s' % path)
    name = 'rp_binary_' + key
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        result = importlib.util.module_from_spec(spec)
        sys.modules[name] = result
        spec.loader.exec_module(result)
    return sys.modules[name]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def unique_child(parent, filename):
    matches = [p for p in Path(parent).iterdir() if p.is_dir() and (p/filename).is_file()]
    if len(matches) != 1:
        raise ValueError('无法唯一确定阶段输出：%s，匹配%d个。' % (parent,len(matches)))
    return matches[0]


def stage(key, args, parent, expected):
    script = HERE/SCRIPTS[key]
    if not script.is_file():
        raise ValueError('缺少配套代码：%s' % script)
    result = subprocess.run([sys.executable, str(script), *map(str,args)])
    if result.returncode:
        raise RuntimeError('%s失败，退出码%d；请查看上面的具体错误。' % (key,result.returncode))
    return unique_child(parent, expected)


def convert_history(source, destination, subject):
    """只转换历史训练特征；原目录/连续原始信号保持原样，不剪接任何时间轴。"""
    import numpy as np
    import pandas as pd
    source = Path(source).expanduser().resolve()
    if source.is_file(): source=source.parent
    config=read_json(source/'preprocess_config.json')
    mapping=read_json(source/'label_mapping.json')
    if config.get('schema') == INPUT_SCHEMA:
        # Reader validates schema, mapping, trial grouping and all QC eligibility.
        data=module('train').read_inputs(source)
        if data['subject_id'] != subject:
            raise ValueError('历史数据受试者%s与--subject %s不一致。' % (data['subject_id'],subject))
        return source
    if config.get('schema') != 'openbci_rps_4ch_preprocess_v1' or mapping.get('marker_to_class_id') != {'11':0,'12':1,'13':2}:
        raise ValueError('历史输入需为配套旧01三分类或新01二分类预处理输出；不猜测Marker。')
    with np.load(source/'rps_windows.npz',allow_pickle=False) as z:
        if str(z['schema'].item()) != config['schema']:
            raise ValueError('历史NPZ与配置schema不一致。')
        marker=z['marker_code'].astype(int)
        if not np.array_equal(marker,z['y'].astype(int)+11):
            raise ValueError('历史窗口Marker和标签不一致。')
        keep=np.isin(marker,[11,13])
        n=len(marker)
        arrays={k:(z[k][keep].copy() if z[k].ndim and len(z[k])==n and k not in ['feature_names','class_names'] else z[k].copy()) for k in z.files}
    trials=pd.read_csv(source/'trial_manifest.csv',encoding='utf-8-sig',keep_default_na=False)
    if set(trials.subject_id.astype(str)) != {subject}:
        raise ValueError('历史数据必须全部属于--subject %s。' % subject)
    if not np.array_equal(trials.marker_code.to_numpy(dtype=int), trials.class_id.to_numpy(dtype=int)+11):
        raise ValueError('历史Trial Marker和类别不一致。')
    excluded=trials[trials.marker_code.eq(12)].copy()
    excluded['exclusion_reason']='unsupported_scissors_marker12_not_rest'
    trials=trials[trials.marker_code.isin([11,13])].copy()
    trials['class_id']=trials.marker_code.map({11:0,13:1})
    trials['class_name']=trials.class_id.map({0:'石头',1:'布'})
    arrays['y']=np.where(arrays['marker_code']==11,0,1).astype(np.int64)
    arrays['class_names']=np.array(['石头','布'])
    arrays['schema']=np.array(INPUT_SCHEMA)
    if not len(trials) or set(trials.class_id) != {0,1}:
        raise ValueError('历史记录未覆盖石头和布。')
    # Keep existing held-out groups. If original data has no split, assign at group level.
    split=config.get('split',{})
    if split.get('mode') not in ['by-date','by-session','within-session']:
        dates=sorted(trials.collection_date.unique())
        sessions=sorted(trials.session_id.unique())
        if len(dates)>1:
            mode='by-date'; test=trials.collection_date.eq(dates[-1]); groups=trials.collection_date.astype(str)
        elif len(sessions)>1:
            mode='by-session'; test=trials.session_id.eq(sessions[-1]); groups=trials.session_id.astype(str)
        else:
            mode='within-session'; test=trials.run_id.eq(4)
            groups=trials.session_id.astype(str)+'::R'+trials.run_id.astype(str)
        trials['split_role']=np.where(test,'test','development')
        trials['inner_cv_group']=subject+'::'+groups
        split={'mode':mode,'test_run':4 if mode=='within-session' else None}
    lookup=trials.set_index('trial_id')
    if not set(arrays['trial_id']).issubset(lookup.index):
        raise ValueError('历史窗口找不到对应Trial。')
    for key in ['split_role','inner_cv_group']:
        arrays[key]=np.asarray(lookup.loc[arrays['trial_id'],key].astype(str).tolist())
    config['schema']=INPUT_SCHEMA
    config['split']={**split,'trial_counts':{str(k):int(v) for k,v in trials.split_role.value_counts().items()}}
    config['binary_conversion']={'source_directory':str(source),'excluded_marker_codes':[12],
        'excluded_trials':len(excluded),'continuous_signal_modified':False,'feature_only':True,
        'note':'历史特征转换只用于02/03训练；连续信号仍在原目录。需回放时请用新01处理原始TXT。'}
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(destination/'rps_windows.npz',**arrays)
    for name in ['trial_manifest.csv','split_manifest.csv']:
        trials.to_csv(destination/name,index=False,encoding='utf-8-sig')
    excluded.to_csv(destination/'excluded_actions.csv',index=False,encoding='utf-8-sig')
    write_json(destination/'preprocess_config.json',config)
    write_json(destination/'label_mapping.json',{'marker_to_class_id':{'11':0,'13':1},
        'class_id_to_marker':{'0':11,'1':13},'class_id_to_name':{'0':'石头','1':'布'},
        'gesture_names_supplied_by_user':True,'names_need_collection_config_verification':False,
        'excluded_marker_codes':[12],'marker_zero_is_rest':False})
    module('train').read_inputs(destination)
    print('[历史转换] 保留%d个石头/布Trial，排除%d个剪刀Trial。' % (len(trials),len(excluded)),flush=True)
    return destination


def replay(model_path, detector_path, data_path, output):
    """Process every original sample; only Run3/4 labels are used for post-hoc scoring."""
    import numpy as np
    import pandas as pd
    m=module('live')
    bundle,model_path=m.load_daily(model_path)
    before=m.file_sha256(model_path)
    params=read_json(detector_path)
    m.validate_detector(bundle,model_path,params)
    data=m.load_features(data_path)
    m.validate_protocol(data)
    m.check_contract(bundle['preprocess_config'],data['config'])
    m.require(data['hashes']==bundle['today_hashes'],'回放需使用该模型当天校准记录的原始预处理输出。')
    signal=m.load_signal(data,'auto')
    frames,features=m.collect_frames(signal,data['config'])
    frames=m.classify_frames(frames,features,bundle['pipeline'])
    activity=m.activity_scores(features,params['noise_reference_rms_uv'])
    events,trace=m.detect(frames,activity,params,keep_trace=True)
    trials=data['trials'][data['trials'].run_id.isin([3,4])]
    bounds=[m.run_intervals(data)[r] for r in [3,4]]
    scored_events=m.events_in_intervals(events,bounds)
    unsupported=m.excluded_intervals(data)
    unsupported_events=m.events_in_intervals(scored_events,unsupported)
    trial_scores,metrics=m.score_trials(events,trials,signal['fs'],len(signal['raw']))
    action_intervals=[(int(t.start_row),int(t.end_row)) for t in trials.itertuples(index=False)]
    outside=[e for e in scored_events if not any(a<=e['decision_row']<b for a,b in action_intervals+unsupported)]
    candidate_intervals=m.candidate_intervals(data,set(trials.trial_id))
    candidate_events=m.events_in_intervals(scored_events,candidate_intervals)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for column in trace:
        if column!='decision_row':frames[column]=trace[column]
    frames['decision_time_s']=frames.decision_row/signal['fs']
    m.write_csv(output/'window_trace.csv',frames)
    event_frame=pd.DataFrame(events,columns=['event_id','decision_row','onset_candidate_row','class_id','vote_agreement','activity_score','release_row','release_reason'])
    event_frame['marker_code_for_display']=[{0:11,1:13}[int(v)] for v in event_frame.class_id]
    event_frame['class_name']=[bundle['class_names'][int(v)] for v in event_frame.class_id]
    event_frame['in_independent_test_run']=[any(a<=v<b for a,b in bounds) for v in event_frame.decision_row]
    event_frame['in_unsupported_scissors_interval']=[any(a<=v<b for a,b in unsupported) for v in event_frame.decision_row]
    m.write_csv(output/'events.csv',event_frame)
    m.write_csv(output/'trial_event_scores.csv',trial_scores)
    report={'schema':'openbci_rp_binary_holdout_continuous_v1','evaluation_runs':[3,4],
        'class_names':bundle['class_names'],'marker_mapping':{'0':11,'1':13},
        'event_metrics':metrics,'unsupported_scissors_triggers':len(unsupported_events),
        'outside_target_and_unsupported_action_triggers':len(outside),
        'candidate_baseline_triggers_not_verified_rest':len(candidate_events),
        'verified_rest_false_triggers_per_minute':None,
        'detector_receives_truth_markers':False,'all_original_rows_processed':len(signal['raw']),
        'activity_mode':'legacy','model_sha256':before,
        'note':'剪刀为范围外动作；保留连续状态，不当静息、不按Marker重置。参数和SVM只由Run1/2选择；同一测试集反复修改后重跑只能用于调试。'}
    m.require(m.file_sha256(model_path)==before,'回放期间模型意外改变。')
    m.write_json(output/'evaluation.json',report)
    esc=lambda x:html.escape(str(x))
    rows=''.join('<tr><th>%s</th>%s</tr>'%(esc(bundle['class_names'][i]),''.join('<td>%d</td>'%v for v in row))
                 for i,row in enumerate(metrics['confusion_stone_paper_no_trigger']))
    page='''<!doctype html><meta charset="utf-8"><title>石头/布连续回放</title>
<style>body{font-family:system-ui,Microsoft YaHei;max-width:950px;margin:35px auto;line-height:1.8}td,th{border:1px solid #ccc;padding:10px}table{border-collapse:collapse}</style>
<h1>石头/布 · Run3+4连续回放</h1>'''
    page+='<p>首次正确输出：<b>%d/%d</b>；漏检%d，首次错分类%d，重复输出%d。</p>'%(metrics['correct_first_triggers'],metrics['n_trials_retained'],metrics['missed_trials'],metrics['wrong_first_class_trials'],metrics['duplicate_trigger_count'])
    page+='<table><tr><th>真实 / 首次输出</th><th>石头</th><th>布</th><th>无新输出</th></tr>'+rows+'</table>'
    page+='<p>剪刀区间内输出%d次，单独审计。其余目标动作标记外输出%d次，为未标注输出；真实静息误触发率未测。</p>'%(len(unsupported_events),len(outside))
    page+='<p>'+esc(report['note'])+'</p><p>该回放使用离线活动门控。实时选择robust模式后，这份回放成绩不能作为robust模式的验证结果。</p>'
    (output/'OPEN_REPORT.html').write_text(page,encoding='utf-8')
    print('[连续独立测试] 首次正确%d/%d；报告：%s'%(metrics['correct_first_triggers'],metrics['n_trials_retained'],output/'OPEN_REPORT.html'),flush=True)
    return report


def run_day(args):
    import numpy as np
    for key in SCRIPTS: module(key)
    raw=Path(args.today_input).expanduser().resolve()
    if not raw.is_file(): raise ValueError('找不到当天TXT：%s'%raw)
    output=Path(args.output).expanduser().resolve()/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    if args.command=='init':
        if args.history_input:
            parent=output/'HISTORY_PREPROCESS'
            history=stage('preprocess',['--input',*args.history_input,'--subject',args.subject,'--output',parent],parent,'rps_windows.npz')
        else:
            source=args.history_data
            if not source:
                sidecar=Path(args.legacy_base).expanduser().resolve().parent/'training_config.json'
                source=read_json(sidecar).get('source_directory')
                if not source:raise ValueError('旧training_config.json没有source_directory，请用--history-data指定旧预处理目录。')
            history=convert_history(source,output/'HISTORY_BINARY',args.subject)
        # Stop before fitting if the proposed "today" is already in the history.
        with np.load(history/'rps_windows.npz',allow_pickle=False) as z:
            last_history=max(z['collection_date'].astype(str))
        model_parent=output/'BASE_MODEL'
    else:
        base=Path(args.base_model).expanduser().resolve()
        if not base.is_file():raise ValueError('找不到二分类基础模型：%s'%base)
    parent=output/'TODAY_PREPROCESS'
    today=stage('preprocess',['--input',raw,'--subject',args.subject,'--split-mode','none','--output',parent],parent,'rps_windows.npz')
    # Validate 48 target trials before spending time training a base model.
    today_data=module('live').load_features(today)
    module('live').validate_protocol(today_data)
    if args.command=='init':
        today_date=str(today_data['trials'].collection_date.iloc[0])
        if today_date<=last_history:
            raise ValueError('当天%s必须晚于历史最后日期%s；请从历史训练池去掉当天数据，保留独立测试。'%(today_date,last_history))
        model_dir=stage('train',['--data',history,'--output',model_parent],model_parent,'base_model.joblib')
        base=model_dir/'base_model.joblib'
    parent=output/'DAILY_MODEL'
    model_dir=stage('calibrate',['--base-model',base,'--today',today,'--output',parent],parent,'daily_model.joblib')
    model=model_dir/'daily_model.joblib'
    parent=output/'LIVE_SETUP'
    gate_dir=stage('live',['--model',model,'--source-data',today,'--prepare-only','--output',parent],parent,'replay_detector_config.json')
    detector=gate_dir/'replay_detector_config.json'
    state={'schema':'openbci_rp_binary_pipeline_v1','base_model':str(base),'daily_model':str(model),
        'detector':str(detector),'today_data':str(today),'subject':args.subject,'run_directory':str(output)}
    # Save valid live paths even if a subsequent report generation fails.
    write_json(output/'pipeline_state.json',state)
    replay(model,detector,today,output/'HOLDOUT_REPLAY')
    print('\n二分类基础模型（以后day复用）：\n%s'%base)
    print('\n当天模型报告：\n%s'%(model_dir/'OPEN_REPORT.html'))
    print('\n启动实时识别：\npython "%s" live --run "%s"'%(Path(__file__).resolve(),output))
    return output


def diagnose_live(log_dir):
    """Read existing runtime logs; never changes model, thresholds or recording files."""
    from collections import Counter
    import numpy as np
    import pandas as pd
    root=Path(log_dir).expanduser().resolve()
    if (root/'live_configuration.json').is_file():
        config_path=root/'live_configuration.json'
    else:
        matches=list(root.rglob('live_configuration.json'))
        if not matches:raise ValueError('该目录下没有实时配置日志：%s'%root)
        config_path=max(matches,key=lambda p:p.stat().st_mtime_ns)
        root=config_path.parent
    config=read_json(config_path)
    summary=read_json(root/'session_summary.json') if (root/'session_summary.json').is_file() else {}
    params=config.get('runtime_detector_settings',config.get('frozen_detector',{}))
    gate_file=root/('activity_calibration.json' if (root/'activity_calibration.json').is_file() else 'rest_calibration.json')
    gate=read_json(gate_file) if gate_file.is_file() else {}
    counts=Counter();qc=Counter();reasons=Counter();candidates=Counter();transport=Counter();total=0;stale=0
    scores=[]
    path=root/'window_trace.csv'
    if path.is_file():
        columns=pd.read_csv(path,nrows=0,encoding='utf-8-sig').columns
        use=[k for k in ['detector_state','qc_status','qc_reasons','predicted_class_id','activity_score','output_suppressed_for_data_age'] if k in columns]
        for chunk in pd.read_csv(path,usecols=use,encoding='utf-8-sig',chunksize=50000,keep_default_na=False):
            total+=len(chunk)
            if 'detector_state' in chunk:counts.update(chunk.detector_state.astype(str))
            if 'qc_status' in chunk:qc.update(chunk.qc_status.astype(str))
            if 'qc_reasons' in chunk:
                for value in chunk.qc_reasons.astype(str):reasons.update(v for v in value.split(';') if v)
            if 'predicted_class_id' in chunk:candidates.update(chunk.predicted_class_id.astype(str))
            if 'activity_score' in chunk:
                values=pd.to_numeric(chunk.activity_score,errors='coerce').to_numpy(dtype=float)
                scores.extend(values[np.isfinite(values)].tolist())
            if 'output_suppressed_for_data_age' in chunk:stale+=int(chunk.output_suppressed_for_data_age.astype(str).str.lower().eq('true').sum())
    issues=root/'transport_issues.csv'
    if issues.is_file():
        for chunk in pd.read_csv(issues,usecols=['reason'],encoding='utf-8-sig',chunksize=50000,keep_default_na=False):
            transport.update(chunk.reason.astype(str))
    raw_info={}
    raw_path=root/'raw_samples.csv'
    if raw_path.is_file():
        channel_columns=['ch%d_uv'%k for k in range(1,5)]
        raw_count=0;nonfinite={k:0 for k in channel_columns};tail=pd.DataFrame(columns=channel_columns)
        for chunk in pd.read_csv(raw_path,usecols=channel_columns,encoding='utf-8-sig',chunksize=50000):
            chunk=chunk.apply(pd.to_numeric,errors='coerce');raw_count+=len(chunk)
            for k in channel_columns:nonfinite[k]+=int((~np.isfinite(chunk[k].to_numpy(dtype=float))).sum())
            chunk=chunk.replace([np.inf,-np.inf],np.nan)
            tail=chunk.tail(1250) if tail.empty else pd.concat([tail,chunk]).tail(1250)
        raw_info={'received_rows':raw_count,'nonfinite_by_channel':nonfinite,
            'last_1250_rows_peak_to_peak_uv':{k:float(tail[k].max()-tail[k].min()) if tail[k].notna().any() else None for k in channel_columns}}
    findings=[]
    status=gate.get('status',{})
    if status.get('state')=='failed':findings.append('参考测量失败，分类输出被门控暂停：'+status.get('error','见参考日志'))
    if status.get('state')=='need_actions':findings.append('该日志来自旧动作参考模式；新版rest只需静息，不提前选择动作。')
    if qc.get('REVIEW',0):findings.append('存在输入质量拦截；查看qc_reasons中的具体通道和原因。')
    if stale or summary.get('counters',{}).get('discarded_stale_samples',0):findings.append('存在过时或超前时间戳数据，相关数据未用于输出。')
    if counts.get('waiting_for_release',0):findings.append('部分窗口等待放松确认；这与分类投票不稳定是不同状态。')
    if counts.get('unstable',0):findings.append('部分窗口活动已达到门控，但SVM类别投票未稳定。')
    if transport:findings.append('接收/连续性日志存在记录；小幅批次时间戳抖动仅作警告，其他重置原因需结合计数检查。')
    if not total and raw_info.get('received_rows',0):findings.append('已经收到原始数据，但没有生成分类窗口；优先检查时间戳拦截和反复滤波预热。')
    if not summary:findings.append('未找到结束汇总；建议停止并保存后再次诊断。')
    if not findings:findings.append('日志未显示上述常见拦截；仍需动作真值判断是否存在错分类或漏检。')
    if config.get('trigger_mode')=='manual':findings.append('本次为按钮触发固定时段分类，活动启动/释放门限未参与判断。')
    result={'log_directory':str(root),'script_version':config.get('script_version'),
        'trigger_mode':config.get('trigger_mode','auto'),
        'model_calibration_date':config.get('calibration_date'),'activity_mode':config.get('activity_gate_options',{}).get('mode'),
        'stream':config.get('stream'),'channels_one_based':config.get('channels_one_based'),'input_unit':config.get('input_unit'),
        'thresholds':{k:params.get(k) for k in ['on_threshold','off_threshold','settle_seconds','vote_windows','minimum_votes','release_windows']},
        'windows':total,'state_counts':dict(counts),'qc_counts':dict(qc),'qc_reasons':dict(reasons),
        'transport_issue_counts':dict(transport),'raw_input':raw_info,
        'candidate_class_counts_not_accuracy':dict(candidates),'stale_windows':stale,
        'activity_quantiles':pd.Series(scores,dtype=float).quantile([.1,.5,.9,.99]).dropna().to_dict(),
        'activity_reference_state':status,'output_events':summary.get('event_count'),
        'posthoc_evaluation':summary.get('posthoc_evaluation'),'findings':findings}
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=p.add_subparsers(dest='command',required=True)
    for command in ['init','day']:
        q=sub.add_parser(command,help='首次重训二分类基础模型并校准' if command=='init' else '复用二分类基础模型进行新一天校准')
        q.add_argument('--today-input',required=True,help='今天采集的原始TXT完整路径')
        q.add_argument('--subject',default='USER01')
        q.add_argument('--output',default='RP_BINARY_NEW_DAY')
        if command=='init':
            g=q.add_mutually_exclusive_group(required=True)
            g.add_argument('--legacy-base',help='旧三分类模型路径；仅用旁边training_config定位历史数据，不使用其SVM')
            g.add_argument('--history-data',help='旧三分类或新二分类01预处理输出子目录')
            g.add_argument('--history-input',nargs='+',help='历史原始TXT（不可包括今天待测这份记录）')
        else:q.add_argument('--base-model',required=True,help='新02生成的二分类base_model.joblib')
    q=sub.add_parser('live',help='启动实时识别及两个实际动作标注按钮')
    q.add_argument('--run',required=True,help='包含pipeline_state.json的本次输出目录')
    q.add_argument('--stream-name',default='rps_raw')
    q.add_argument('--activity-mode',choices=['rest','legacy','robust'],default='rest')
    q.add_argument('--live-profile',choices=['stable','responsive','replay'],default='stable')
    q.add_argument('--trigger-mode',choices=['auto','manual'],default='auto',help='manual：点击后做动作，固定采集后分类，绕过活动门控')
    q.add_argument('--manual-duration-seconds',type=float,default=4.,help='按钮点击后采集3~4秒，默认4秒')
    q.add_argument('--rest-seconds',type=float,default=8.)
    q.add_argument('--activity-on-z',type=float,default=6.,help='rest/robust启动门限；6为默认工程设置，并非个体最优值')
    q.add_argument('--activity-off-z',type=float,default=3.,help='rest/robust释放门限，须小于启动门限')
    q.add_argument('--input-unit',choices=['uv','mv','v'],default='uv')
    q.add_argument('--channels',nargs=4,type=int,default=[1,2,3,4])
    q.add_argument('--source-id')
    q.add_argument('--output',default='RP_BINARY_LIVE_LABEL')
    q.add_argument('--duration',type=float,default=0)
    q.add_argument('--no-ui',action='store_true')
    q=sub.add_parser('replay',help='用冻结模型/门控回放当天完整记录，评分Run3/4')
    q.add_argument('--run',required=True)
    q=sub.add_parser('diagnose',help='只读取实时日志，列出门控、QC、接收和投票的具体阻塞情况')
    q.add_argument('--log-dir',required=True,help='某次实时输出子目录，或包含多次实时记录的父目录')
    args=p.parse_args()
    try:
        if args.command=='diagnose':diagnose_live(args.log_dir);return 0
        if args.command in ['init','day']:run_day(args)
        else:
            root=Path(args.run).expanduser().resolve()
            state=read_json(root/'pipeline_state.json')
            if state.get('schema')!='openbci_rp_binary_pipeline_v1':raise ValueError('运行状态文件不是二分类格式。')
            if args.command=='replay':
                replay(state['daily_model'],state['detector'],state['today_data'],root/('REPLAY_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
            else:
                cmd=[sys.executable,str(HERE/SCRIPTS['live']),'--model',state['daily_model'],'--detector',state['detector'],
                     '--stream-name',args.stream_name,'--activity-mode',args.activity_mode,'--live-profile',args.live_profile,
                     '--trigger-mode',args.trigger_mode,'--manual-duration-seconds',str(args.manual_duration_seconds),
                     '--rest-seconds',str(args.rest_seconds),
                     '--activity-on-z',str(args.activity_on_z),'--activity-off-z',str(args.activity_off_z),
                     '--input-unit',args.input_unit,'--channels',*map(str,args.channels),
                     '--output',args.output,'--duration',str(args.duration)]
                if args.source_id:cmd.extend(['--source-id',args.source_id])
                if args.no_ui:cmd.append('--no-ui')
                return subprocess.run(cmd).returncode
        return 0
    except KeyboardInterrupt:return 130
    except Exception as exc:
        print('[失败] %s: %s'%(type(exc).__name__,exc),file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
