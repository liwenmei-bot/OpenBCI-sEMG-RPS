# -*- coding: utf-8 -*-
"""OpenBCI 4-channel sEMG Rock vs Paper pipeline, v2.1.

Commands: check, collect, preprocess, train, replay, live, auto.
Training protocol: grouped CV on Runs 1-2; refit on Runs 1-3; untouched Run 4 test.
"""
from __future__ import annotations
import argparse, csv, json, sys, time, random
from datetime import datetime
from pathlib import Path
import numpy as np

DEFAULT_CONFIG = Path(__file__).with_name('config_rp.json')
LABELS = {0: 'Rock', 1: 'Paper'}


def stamp(): return datetime.now().strftime('%Y%m%d_%H%M%S_%f')

def load_config(path=None):
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, 'r', encoding='utf-8') as f: cfg = json.load(f)
    required = ['stream_name','stream_type','sampling_rate_hz','runs','trials_per_run','rock_trials_per_run','paper_trials_per_run','rest_seconds','prepare_seconds','action_seconds','recovery_seconds','filter_band_hz','notch_hz','window_seconds','step_seconds','feature_sets']
    missing = [k for k in required if k not in cfg]
    if missing: raise ValueError(f'配置文件缺少字段：{missing}')
    if cfg['runs'] != 4 or cfg['trials_per_run'] != 20 or cfg['rock_trials_per_run'] != 10 or cfg['paper_trials_per_run'] != 10:
        raise ValueError('本版本固定采用4 Run × 20 Trial，每Run石头10次、布10次。')
    if cfg.get("class_mapping") != {"0":"Rock","1":"Paper"} or cfg.get("channels_used") != [1,2,3,4] or cfg["sampling_rate_hz"]!=250: raise ValueError("类别、通道或采样率不匹配RP协议")
    if any(cfg[k]!=v for k,v in {"action_seconds":4.0,"window_seconds":.2,"step_seconds":.1,"feature_action_start_s":.5,"feature_action_end_s":3.5,"feature_sets":[20,25]}.items()): raise ValueError("本版本固定4秒动作和29个窗口，请勿改变协议")
    return cfg


def resolve_stream(name, stype, timeout=10, expected_fs=None):
    from pylsl import resolve_byprop
    print(f'查找LSL：name={name!r}, type={stype!r}')
    streams = [v for v in resolve_byprop('name', name, timeout=timeout) if v.type()==stype]
    if len(streams)!=1: raise RuntimeError(f'需要唯一匹配的LSL流，找到{len(streams)}个。')
    info=streams[0]
    print(f'找到：{info.name()} | {info.type()} | {info.channel_count()}通道 | 标称采样率={info.nominal_srate()} Hz')
    if expected_fs is not None and abs(info.nominal_srate()-expected_fs)>1: raise RuntimeError('标称采样率不符，停止采集')
    return info

class EMGStream:
    def __init__(self,name,stype,fs,timeout=10,channels=4):
        from pylsl import StreamInlet
        info=resolve_stream(name,stype,timeout,fs)
        if info.channel_count()<channels: raise RuntimeError('通道不足')
        self.inlet=StreamInlet(info,max_buflen=10,recover=False,processing_flags=0)
        self.inlet.open_stream(timeout=timeout)
        self.info,self.fs,self.nch=info,float(fs),channels
        self.last_stats={}
    def pull(self,duration,extra=.5):
        # Flush stale buffered data before each cue segment. Keep original sample order.
        self.inlet.flush()
        start=time.monotonic(); target=int(round(duration*self.fs)); xs=[]; ts=[]; count=0
        last=start; max_pause=0.
        while count<target and time.monotonic()-start<duration+extra:
            chunk,t=self.inlet.pull_chunk(timeout=.05,max_samples=min(256,target-count))
            if chunk:
                now=time.monotonic(); max_pause=max(max_pause,now-last); last=now
                xs.extend(np.asarray(chunk,float)[:,:self.nch]); ts.extend(t); count+=len(chunk)
        elapsed=time.monotonic()-start
        self.last_stats={'elapsed_s':elapsed,'max_arrival_pause_s':max(max_pause,time.monotonic()-last)}
        return np.asarray(xs,float).reshape(-1,self.nch),np.asarray(ts,float)


def quality_report(x,fs,timestamps=None,duration=4,stats=None,cfg=None):
    x=np.asarray(x,float); fail=[]; warn=[]; details={}
    limits=(cfg or {}).get('quality_limits',{})
    if x.ndim!=2 or x.shape[1]!=4: return 'FAIL',['数据形状不是 N×4'],details
    if len(x)!=round(fs*duration): fail.append(f'样本不足或多余：{len(x)}/{round(fs*duration)}')
    if not np.isfinite(x).all(): fail.append('存在NaN/Inf')
    if timestamps is not None:
        t=np.asarray(timestamps,float)
        if len(t)!=len(x) or not np.isfinite(t).all(): fail.append('时间戳缺失或非有限')
        elif len(t)>1:
            dt=np.diff(t); positive=dt[dt>0]; n=int(np.sum(dt<=0)); gap=float(max(0,np.max(dt)))
            rate=float((len(t)-1)/(t[-1]-t[0])) if t[-1]>t[0] else None
            details.update(nonincreasing_count=n,nonincreasing_fraction=n/len(dt),max_gap_ms=gap*1000,gap_over_12ms=int(np.sum(dt>.012)),gap_over_50ms=int(np.sum(dt>.05)),timestamp_rate_hz=rate,positive_interval_percentiles_ms=np.percentile(positive*1000,[50,95,99]).tolist() if len(positive) else [])
            if n: warn.append(f'非递增时间戳 {n}次')
            if gap>.012: warn.append(f'时间戳最大间隔 {gap*1000:.1f}ms（不能单独证明丢包）')
            if gap>limits.get('max_timestamp_gap_s',.1) or n/len(dt)>limits.get('max_nonincreasing_fraction',.10): fail.append('时间戳严重异常，片段可靠性不足')
            if rate is None or abs(rate-fs)>fs*limits.get('max_timestamp_rate_error',.10): fail.append('时间戳跨度严重异常')
            elif abs(rate-fs)>fs*.03: warn.append('时间戳估计采样率偏差超过3%')
    if stats:
        details.update(stats)
        if stats['max_arrival_pause_s']>limits.get('max_arrival_pause_s',.25): fail.append('接收暂停过长，拒绝本片段')
        if abs(stats['elapsed_s']-duration)>limits.get('max_duration_error_s',.5): fail.append('采集实际时长偏差过大')
    for c in range(4):
        v=x[:,c]
        if len(v)==0 or np.ptp(v)<1e-7: fail.append(f'CH{c+1}无数据或平直')
        elif np.mean(np.abs(v)>1e5)>.01: fail.append(f'CH{c+1}幅值异常（非硬件饱和的确证）')
    return ('FAIL' if fail else 'WARNING' if warn else 'PASS'),fail+warn,details


def filter_signal(x, cfg):
    from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
    x=np.asarray(x,float)
    if len(x)<max(80,int(cfg['sampling_rate_hz']*.5)): raise ValueError('数据太短，无法可靠滤波')
    fs=float(cfg['sampling_rate_hz']); low,high=map(float,cfg['filter_band_hz']); nyq=fs/2
    if not (0<low<high<nyq): raise ValueError(f'滤波范围{low}-{high}Hz必须低于奈奎斯特频率{nyq}Hz')
    sos=butter(int(cfg.get('filter_order',4)),[low,high],btype='bandpass',fs=fs,output='sos')
    y=sosfiltfilt(sos,x,axis=0)
    notch=float(cfg.get('notch_hz',0))
    if notch>0 and notch<nyq:
        b,a=iirnotch(notch,Q=float(cfg.get('notch_q',30)),fs=fs); y=filtfilt(b,a,y,axis=0)
    return y


def window_features(x, cfg, feature_set='25'):
    x=np.asarray(x,float)
    if len(x)<10 or x.shape[1]!=4: raise ValueError('特征窗口长度或通道数错误')
    feats=[]; rms=[]
    for c in range(4):
        v=x[:,c]; av=np.abs(v); r=float(np.sqrt(np.mean(v*v))+1e-12); d=np.diff(v)
        eps=float(cfg.get('zc_ssc_noise_fraction',.01))*(np.std(v)+1e-12)
        zc=int(np.sum(((v[:-1]*v[1:])<0)&(np.abs(d)>=eps)))
        ssc=int(np.sum(((d[:-1]*d[1:])<0)&(np.abs(np.diff(d))>=eps))) if len(d)>1 else 0
        feats.extend([r,float(np.mean(av)),float(np.sum(np.abs(d))),zc,ssc]); rms.append(r)
    if str(feature_set)=='20': return np.asarray(feats,float)
    r=np.asarray(rms); rel=r/(np.sum(r)+1e-12)
    ratio=np.log((r[0]+r[1]+1e-12)/(r[2]+r[3]+1e-12))
    return np.r_[feats,rel,ratio]


def extract_trial_windows(raw,cfg,feature_set='25'):
    x=filter_signal(raw,cfg); fs=float(cfg['sampling_rate_hz'])
    start=int(float(cfg['feature_action_start_s'])*fs); end=min(len(x),int(float(cfg['feature_action_end_s'])*fs)); x=x[start:end]
    w=int(float(cfg['window_seconds'])*fs); step=int(float(cfg['step_seconds'])*fs)
    if w<=0 or step<=0: raise ValueError('窗口和步长必须大于0')
    out=[window_features(x[i:i+w],cfg,feature_set) for i in range(0,len(x)-w+1,step)]
    return np.asarray(out,float)


def create_session_dir(root):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    session=root/f'RP_{stamp()}'; session.mkdir()
    return session


def resolve_data_dir(path):
    p=Path(path)
    if (p/'LATEST_SESSION.txt').exists():
        name=(p/'LATEST_SESSION.txt').read_text(encoding='utf-8').strip(); candidate=p/name
        if candidate.is_dir(): return candidate
    return p


def load_dataset(data_dir):
    p=resolve_data_dir(data_dir); files=sorted(p.glob('rp_trials_*_run*.npz'))
    if not files: raise RuntimeError(f'{p}中没有本次Session的Run数据文件。')
    raws=[]; labels=[]; runs=[]; ids=[]; qcs=[]
    for f in files:
        with np.load(f,allow_pickle=False) as z:
            if str(z['protocol'].item())!='RP_v2.1': raise RuntimeError('非RP v2.1数据')
            if float(z['fs'])!=250 or len(z['timestamps'])!=len(z['raw']): raise RuntimeError('NPZ采样率或时间戳不符')
            qc_values = z['qc'] if 'qc' in z else np.asarray(['UNKNOWN'] * len(z['raw']))
            for raw,y,r,tid,qc in zip(z['raw'],z['labels'],z['runs'],z['trial_ids'],qc_values):
                raws.append(np.asarray(raw,float)); labels.append(int(y)); runs.append(int(r)); ids.append(str(tid)); qcs.append(str(qc))
    return raws,np.asarray(labels),np.asarray(runs),np.asarray(ids),np.asarray(qcs),p


def cmd_check(a,cfg):
    st=EMGStream(a.stream_name or cfg['stream_name'],a.stream_type or cfg['stream_type'],cfg['sampling_rate_hz'],a.timeout)
    x,t=st.pull(10); status,msg,stats=quality_report(x,st.fs,t,10,st.last_stats,cfg)
    print(json.dumps(stats,ensure_ascii=False,indent=2)); print('质量检查：',status,'；'.join(msg))
    for c in range(4):
        if len(x): print(f'CH{c+1}: mean={np.mean(x[:,c]):.3f}, SD={np.std(x[:,c]):.3f}, P-P={np.ptp(x[:,c]):.3f}')
    if status=='FAIL': raise RuntimeError('检查失败，请先排查')


def cmd_collect(a,cfg):
    st=EMGStream(a.stream_name or cfg['stream_name'],a.stream_type or cfg['stream_type'],cfg['sampling_rate_hz'],a.timeout)
    root=Path(a.out); session=create_session_dir(root)
    # Block default training after an incomplete new acquisition.
    (root/'LATEST_SESSION.txt').unlink(missing_ok=True)
    cfg=dict(cfg,stream_name=st.info.name(),stream_type=st.info.type())
    (session/'session_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest=session/'collection_manifest.csv'; attempts_file=session/'attempts.jsonl'
    print('Session:',session,'; 4 Run，每Run石头10次、布10次。')
    for run in range(1,5):
        labels=[0]*10+[1]*10; random.Random(a.seed+run).shuffle(labels)
        raws=[]; stamps=[]; kept=[]; tids=[]; qcs=[]
        for j,label in enumerate(labels,1):
            tid=f'R{run}_T{j:02d}'; attempt=0
            while True:
                attempt+=1
                print(f'{tid} 放松3秒',flush=True); st.pull(cfg['rest_seconds'])
                print(f'准备：{LABELS[label]}',flush=True); st.pull(cfg['prepare_seconds'])
                print(f'GO：{"石头" if label==0 else "布"}，保持4秒',flush=True)
                x,t=st.pull(cfg['action_seconds']); q,msg,stats=quality_report(x,st.fs,t,cfg['action_seconds'],st.last_stats,cfg)
                print('放松',flush=True); st.pull(cfg['recovery_seconds'])
                row={'trial_id':tid,'run':run,'label':label,'attempt':attempt,'quality':q,'warnings':'；'.join(msg),'samples':len(x)}
                with attempts_file.open('a',encoding='utf-8') as f: f.write(json.dumps(dict(row,stats=stats),ensure_ascii=False)+'\n')
                print(q,'；'.join(msg))
                if q=='FAIL':
                    if input('FAIL：Enter重做 / q退出：').strip().lower()=='q': raise RuntimeError('采集中止；完整Run仍保存')
                    continue
                break
            raws.append(x); stamps.append(t); kept.append(label); tids.append(tid); qcs.append(q)
            with manifest.open('a',newline='',encoding='utf-8-sig' if not manifest.exists() else 'utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(row));
                if f.tell()==0: w.writeheader()
                w.writerow(row)
            # Save every accepted trial, including partial Run, for recovery/audit.
            np.savez_compressed(session/f'rp_trials_{run:02d}_run{run}.npz',raw=np.stack(raws),timestamps=np.stack(stamps),labels=np.asarray(kept),runs=np.full(len(kept),run),trial_ids=np.asarray(tids),qc=np.asarray(qcs),fs=st.fs,protocol='RP_v2.1')
        if run<4: input('休息后按Enter进入下一Run')
    (root/'LATEST_SESSION.txt').write_text(session.name,encoding='utf-8')
    print('完成80 Trial：',session)


def session_config(session,cfg):
    p=session/'session_config.json'
    if not p.exists(): raise RuntimeError('缺少Session配置')
    saved=json.loads(p.read_text(encoding='utf-8'))
    # Capture settings are authoritative, including actual stream identity.
    for k in ('sampling_rate_hz','channels_used','filter_band_hz','filter_order','notch_hz','notch_q','window_seconds','step_seconds','feature_action_start_s','feature_action_end_s','feature_sets','zc_ssc_noise_fraction','action_seconds','quality_limits','class_mapping'):
        if saved.get(k)!=cfg.get(k): raise RuntimeError(f'配置变更 {k}：请使用本Session的配置')
    return saved


def cmd_preprocess(a,cfg):
    import pandas as pd
    raws,ys,runs,ids,qcs,session=load_dataset(a.data); _validate_design(ys,runs,ids,qcs); cfg=session_config(session,cfg)
    rows=[]; qcrows=[]
    for raw,y,r,tid,qc0 in zip(raws,ys,runs,ids,qcs):
        q,msg,_=quality_report(raw,cfg['sampling_rate_hz'],duration=cfg['action_seconds'],cfg=cfg)
        qc='FAIL' if q=='FAIL' else str(qc0)
        qcrows.append({'trial_id':tid,'run':int(r),'label':LABELS[int(y)],'samples':len(raw),'qc':qc,'warnings':'；'.join(msg)})
        if qc=='FAIL': continue
        for fsname in cfg['feature_sets']:
            feats=extract_trial_windows(raw,cfg,str(fsname))
            for wi,row in enumerate(feats):
                d={'trial_id':tid,'run':int(r),'label':int(y),'feature_set':str(fsname),'window':wi,'qc':qc}
                d.update({f'f{k+1}':float(v) for k,v in enumerate(row)}); rows.append(d)
    out=Path(a.out)/session.name; out.mkdir(parents=True,exist_ok=True)
    for dest in (out,session):
        pd.DataFrame(rows).to_csv(dest/'window_features.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(qcrows).to_csv(dest/'trial_qc.csv',index=False,encoding='utf-8-sig')
    import hashlib
    (session/'feature_manifest.json').write_text(json.dumps({'config':cfg,'raw_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(session.glob('rp_trials_*_run*.npz'))},'feature_hash':hashlib.sha256((session/'window_features.csv').read_bytes()).hexdigest()},ensure_ascii=False),encoding='utf-8')
    print(f'Session={session}; Trial={len(qcrows)}; PASS={sum(x["qc"]=="PASS" for x in qcrows)}; WARNING={sum(x["qc"]=="WARNING" for x in qcrows)}; FAIL={sum(x["qc"]=="FAIL" for x in qcrows)}; 窗口={len(rows)}')


def _validate_design(labels,runs,ids,qcs):
    if len(ids)!=80: raise RuntimeError(f'本Session应有80 Trial，实际{len(ids)}。请检查是否中断或数据文件不完整。')
    if len(set(ids))!=80: raise RuntimeError('Trial ID重复，停止训练以避免数据混淆。')
    for r in range(1,5):
        ix=np.where(runs==r)[0]
        if len(ix)!=20: raise RuntimeError(f'Run{r}应有20 Trial，实际{len(ix)}。')
        if sum(labels[ix]==0)!=10 or sum(labels[ix]==1)!=10: raise RuntimeError(f'Run{r}类别数量不平衡或标签错误。')
    bad=[ids[i] for i,q in enumerate(qcs) if q not in ('PASS','WARNING')]
    if bad: raise RuntimeError(f'发现{len(bad)}个FAIL/未知 Trial：{bad[:8]}。请检查trial_qc.csv并重新采集，不建议直接训练。')


def cmd_train(a,cfg):
    import pandas as pd, joblib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,classification_report
    import pandas as pd
    raws,labels,runs,ids,qcs,session=load_dataset(a.data); _validate_design(labels,runs,ids,qcs)
    feature_path=session/'window_features.csv'
    if not feature_path.exists(): raise RuntimeError('找不到本Session的window_features.csv，请先运行03_preprocess_rp_4ch.py。')
    cfg=session_config(session,cfg)
    import hashlib
    meta=json.loads((session/'feature_manifest.json').read_text(encoding='utf-8'))
    if meta['config']!=cfg or meta['feature_hash']!=hashlib.sha256(feature_path.read_bytes()).hexdigest() or meta['raw_hashes']!={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(session.glob('rp_trials_*_run*.npz'))}: raise RuntimeError('原始数据/特征/配置已变化，请重新预处理')
    feature_df=pd.read_csv(feature_path)
    print('采集WARNING Trial数：',int(np.sum(qcs=='WARNING')),'；保留用于主分析，不等于已证实无丢包')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    candidates={
      'LinearSVM':Pipeline([('scale',StandardScaler()),('clf',SVC(kernel='linear',C=1,class_weight='balanced'))]),
      'RBFSVM':Pipeline([('scale',StandardScaler()),('clf',SVC(kernel='rbf',C=1,gamma='scale',class_weight='balanced'))]),
      'LDA':Pipeline([('scale',StandardScaler()),('clf',LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'))]),
      'RandomForest':RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight='balanced',random_state=42,n_jobs=-1)
    }
    results=[]; cached={}
    # Train from the canonical CSV generated by preprocessing, not a second feature calculation.
    for fsname in map(str,cfg['feature_sets']):
        trials=[]
        part=feature_df[feature_df['feature_set'].astype(str)==fsname]
        for tid in ids:
            one=part[part['trial_id'].astype(str)==tid].sort_values('window')
            n_features=20 if fsname=='20' else 25
            cols=[f'f{k}' for k in range(1,n_features+1)]
            fw=one[cols].to_numpy(dtype=float)
            i=np.where(ids==tid)[0][0]
            if not np.isfinite(fw).all() or not (one['label']==labels[i]).all() or not (one['run']==runs[i]).all() or list(one['window'])!=list(range(len(one))): raise RuntimeError(f'{tid} 特征元数据异常')
            if len(fw)!=29: raise RuntimeError(f'{tid} {fsname}维特征窗口数应为29，实际{len(fw)}；请检查预处理或原始数据。')
            trials.append(fw)
        cached[fsname]=trials
        train_idx=np.where(np.isin(runs,[1,2]))[0]
        y_train_trials=labels[train_idx]; groups=ids[train_idx]
        cv=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=42)
        for name,base_model in candidates.items():
            fold_scores=[]
            for fold,(tr_rel,va_rel) in enumerate(cv.split(np.zeros(len(train_idx)),y_train_trials,groups),1):
                tr_ids=train_idx[tr_rel]; va_ids=train_idx[va_rel]
                Xtr=np.vstack([trials[i] for i in tr_ids]); ytr=np.concatenate([np.full(len(trials[i]),labels[i]) for i in tr_ids])
                model=__import__('sklearn.base',fromlist=['clone']).clone(base_model); model.fit(Xtr,ytr)
                yt=[]; yp=[]
                for i in va_ids:
                    pred=model.predict(trials[i]); yt.append(labels[i]); yp.append(int(np.mean(pred)>=.5))
                fold_scores.append(float(balanced_accuracy_score(yt,yp)))
            score=float(np.mean(fold_scores)); results.append({'features':fsname,'model':name,'cv_mean_ba':score,'cv_fold_ba':fold_scores})
            print(f'{fsname}维 | {name:12s} | grouped 4-fold Trial BA={score*100:.1f}% | folds={[round(x*100,1) for x in fold_scores]}')
    best=max(results,key=lambda x:x['cv_mean_ba']); print(f'\n选择：{best["features"]}维 + {best["model"]}，CV平均Trial BA={best["cv_mean_ba"]*100:.1f}%')
    model=__import__('sklearn.base',fromlist=['clone']).clone(candidates[best['model']])
    chosen=cached[best['features']]
    idx_train=np.where(np.isin(runs,[1,2,3]))[0]; idx_test=np.where(runs==4)[0]
    X=np.vstack([chosen[i] for i in idx_train]); y=np.concatenate([np.full(len(chosen[i]),labels[i]) for i in idx_train]); model.fit(X,y)
    yt=[]; yp=[]; details=[]
    for i in idx_test:
        pred=model.predict(chosen[i]); frac=float(np.mean(pred==1)); vote=int(frac>=.5)
        yt.append(int(labels[i])); yp.append(vote)
        details.append({'trial_id':ids[i],'run':4,'true':LABELS[int(labels[i])],'pred':LABELS[vote],'correct':vote==int(labels[i]),'paper_window_fraction':frac,'n_windows':len(pred),'qc':str(qcs[i])})
    acc=float(accuracy_score(yt,yp)); ba=float(balanced_accuracy_score(yt,yp)); cm=confusion_matrix(yt,yp,labels=[0,1])
    print(f'\n===== Run4独立Trial验证 =====\nN={len(yt)} | Accuracy={acc*100:.1f}% | Balanced Accuracy={ba*100:.1f}%\n混淆矩阵（真实行/预测列，Rock/Paper）：\n{cm}')
    print(classification_report(yt,yp,target_names=['Rock','Paper'],zero_division=0))
    bundle={'protocol':'RP_v2.1','model':model,'feature_set':best['features'],'fs':float(cfg['sampling_rate_hz']),'channels':4,'window_s':float(cfg['window_seconds']),'step_s':float(cfg['step_seconds']),'action_start_s':float(cfg['feature_action_start_s']),'action_end_s':float(cfg['feature_action_end_s']),'config':cfg,'labels':LABELS,'stream_name':cfg['stream_name'],'stream_type':cfg['stream_type'],'selected_model':best['model'],'cv_mean_ba':best['cv_mean_ba'],'run4_accuracy':acc,'run4_balanced_accuracy':ba,'training_session':str(session.resolve()),'created_at':stamp()}
    modelpath=out/f'rp_binary_model_{stamp()}.joblib'; joblib.dump(bundle,modelpath)
    pd.DataFrame(details).to_csv(out/f'run4_predictions_{stamp()}.csv',index=False,encoding='utf-8-sig')
    summary={'session':str(session.resolve()),'n_trials':80,'run4_trials':20,'selected_model':best['model'],'feature_set':best['features'],'grouped_cv_mean_ba':best['cv_mean_ba'],'candidate_results':results,'run4_accuracy':acc,'run4_balanced_accuracy':ba,'confusion_matrix':cm.tolist(),'run4_trial_predictions':details,'run4_used_for_model_selection':False,'warning_trials':int(np.sum(qcs=='WARNING')),'run4_warning_trials':int(np.sum(qcs[idx_test]=='WARNING')),'run4_pass_only_accuracy':float(np.mean(np.asarray(yt)[qcs[idx_test]=='PASS']==np.asarray(yp)[qcs[idx_test]=='PASS'])) if np.any(qcs[idx_test]=='PASS') else None,'run4_pass_only_n':int(np.sum(qcs[idx_test]=='PASS'))}
    with open(out/f'training_summary_{stamp()}.json','w',encoding='utf-8') as f: json.dump(summary,f,ensure_ascii=False,indent=2)
    print('模型保存：',modelpath)


def predict_segment(raw,bundle):
    cfg=bundle['config']; fw=extract_trial_windows(raw,cfg,bundle['feature_set'])
    if len(fw)==0:return None,0.,0
    pred=bundle['model'].predict(fw); frac=float(np.mean(pred==1)); vote=int(frac>=.5)
    return vote,max(frac,1-frac),len(fw)


def validate_bundle(b):
    if b.get('protocol')!='RP_v2.1' or b.get('labels')!=LABELS: raise RuntimeError('模型不属于石头/布 v2.1，禁止混用旧模型')


def cmd_live(a,cfg):
    import joblib
    b=joblib.load(a.model); validate_bundle(b); cfg=b['config']
    st=EMGStream(a.stream_name or b['stream_name'],a.stream_type or b['stream_type'],b['fs'],a.timeout)
    out=Path('outputs/live')/stamp(); out.mkdir(parents=True); log=out/'predictions.csv'; ntrial=0
    print('Enter开始，q退出；准备1秒后做石头或布4秒。')
    while True:
        if input('Enter开始 / q退出：').strip().lower()=='q': break
        print('准备',flush=True); st.pull(cfg['prepare_seconds']); print('GO：石头或布',flush=True)
        raw,t=st.pull(cfg['action_seconds']); q,msg,stats=quality_report(raw,b['fs'],t,cfg['action_seconds'],st.last_stats,cfg)
        vote,conf,n=(None,0.,0) if q=='FAIL' else predict_segment(raw,b)
        pred=LABELS[vote] if vote is not None else 'REJECT'
        print(f'结果：{pred} | 投票一致率={conf:.1%}（不是概率） | QC={q} | {"；".join(msg)}')
        actual=input('真实动作 r=石头 / p=布 / Enter跳过：').strip().lower(); truth={'r':'Rock','p':'Paper'}.get(actual,'')
        ntrial+=1; np.savez_compressed(out/f'trial_{ntrial:03d}.npz',raw=raw,timestamps=t,protocol='RP_v2.1')
        row=dict(trial=ntrial,true=truth,pred=pred,correct=int(truth==pred) if truth else '',vote_agreement=conf,windows=n,qc=q,warnings='；'.join(msg),timing=json.dumps(stats))
        with log.open('a',newline='',encoding='utf-8-sig' if not log.exists() else 'utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(row));
            if f.tell()==0:w.writeheader()
            w.writerow(row)
        print('日志：',log)


def cmd_auto(a,cfg):
    raise RuntimeError('自动门控版本暂不建议用于正式实验。本版先完成手动触发识别；旧版auto存在动作起点与4秒片段未严格对齐的问题，需单独完成在线状态机和因果滤波验证后再启用。')


def build_parser():
    p=argparse.ArgumentParser(); p.add_argument('--config',default=str(DEFAULT_CONFIG)); sub=p.add_subparsers(dest='cmd',required=True)
    def stream_args(q): q.add_argument('--stream-name',default=None); q.add_argument('--stream-type',default=None); q.add_argument('--timeout',type=float,default=10)
    q=sub.add_parser('check'); q.add_argument('--config',default=argparse.SUPPRESS); stream_args(q); q.set_defaults(func=cmd_check)
    q=sub.add_parser('collect'); q.add_argument('--config',default=argparse.SUPPRESS); stream_args(q); q.add_argument('--out',default='data'); q.add_argument('--seed',type=int,default=20261125); q.add_argument('--max-retries',type=int,default=2); q.add_argument('--keep-warning',action='store_true'); q.set_defaults(func=cmd_collect)
    q=sub.add_parser('preprocess'); q.add_argument('--config',default=argparse.SUPPRESS); q.add_argument('--data',default='data'); q.add_argument('--out',default='outputs/preprocess'); q.set_defaults(func=cmd_preprocess)
    q=sub.add_parser('train'); q.add_argument('--config',default=argparse.SUPPRESS); q.add_argument('--data',default='data'); q.add_argument('--out',default='models'); q.set_defaults(func=cmd_train)
    q=sub.add_parser('live'); q.add_argument('--config',default=argparse.SUPPRESS); q.add_argument('--model',required=True); q.add_argument('--stream-name'); q.add_argument('--stream-type'); q.add_argument('--timeout',type=float,default=10); q.set_defaults(func=cmd_live)
    q=sub.add_parser('auto'); q.add_argument('--config',default=argparse.SUPPRESS); q.set_defaults(func=cmd_auto)
    return p


def main():
    a=build_parser().parse_args(); cfg=load_config(a.config)
    try: a.func(a,cfg)
    except KeyboardInterrupt: print('\n用户中断'); sys.exit(130)
    except Exception as e: print(f'\n错误：{e}',file=sys.stderr); sys.exit(2)

if __name__=='__main__':main()
