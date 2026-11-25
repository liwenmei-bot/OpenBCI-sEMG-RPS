# -*- coding: utf-8 -*-
"""OpenBCI 4-channel sEMG Rock vs Scissors system.

Commands:
  check   : inspect an LSL stream and four channels
  collect : collect 4 runs x 20 trials (10 Rock, 10 Scissors per run)
  train   : train and evaluate models using run-wise splits
  live    : manual-trigger 4-second gesture recognition
  auto    : continuous activity-gated recognition (experimental)
  daily   : print the recommended command sequence

Dependencies: numpy, scipy, pandas, scikit-learn, joblib, pylsl
Install: pip install numpy scipy pandas scikit-learn joblib pylsl
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time, random, threading, queue, warnings
from datetime import datetime
from pathlib import Path

import numpy as np

DEFAULT_STREAM_NAME = "rps_raw"  # change with --stream-name if your LSL stream differs
DEFAULT_STREAM_TYPE = "EMG"
DEFAULT_FS = 250.0
LABELS = {0: "Rock", 1: "Scissors"}


def stamp(): return datetime.now().strftime('%Y%m%d_%H%M%S')

def ensure_deps():
    try:
        import scipy, pandas, sklearn, joblib
        from pylsl import resolve_byprop, StreamInlet
    except Exception as e:
        raise RuntimeError("缺少依赖。请在 bci 环境运行：pip install numpy scipy pandas scikit-learn joblib scipy pylsl\n原始错误：" + str(e))
    return resolve_byprop, StreamInlet


def resolve_stream(name, stype, timeout=10):
    from pylsl import resolve_byprop
    print(f"正在查找 LSL：name={name!r}, type={stype!r}，最多等待 {timeout} 秒...")
    streams = resolve_byprop('name', name, timeout=timeout)
    if not streams and stype:
        streams = resolve_byprop('type', stype, timeout=3)
        if streams:
            print(f"未找到指定名称，找到 type={stype!r} 的流，将使用第一个：{streams[0].name()} ")
    if not streams:
        raise RuntimeError("没有找到 LSL 数据流。请先打开 OpenBCI GUI、连接板子并启动 LSL 输出；检查 Stream Name/Type。")
    # Prefer exact name if available
    exact = [s for s in streams if s.name() == name]
    info = exact[0] if exact else streams[0]
    print(f"已找到：name={info.name()}, type={info.type()}, channels={info.channel_count()}, srate={info.nominal_srate()}")
    return info


class EMGStream:
    def __init__(self, name=DEFAULT_STREAM_NAME, stype=DEFAULT_STREAM_TYPE, fs=DEFAULT_FS, channels=4, timeout=10):
        from pylsl import StreamInlet
        info = resolve_stream(name, stype, timeout)
        if info.channel_count() < channels:
            raise RuntimeError(f"LSL 只有 {info.channel_count()} 个通道，但程序要求至少 {channels} 个。")
        self.inlet = StreamInlet(info, max_buflen=60, max_chunklen=0, recover=True)
        self.fs = fs
        self.nch = channels
        self.info = info
        self.inlet.flush()

    def pull(self, duration, max_wait_extra=5):
        """Collect approximately duration seconds, retaining first four channels."""
        target = max(1, int(round(duration * self.fs)))
        xs, ts = [], []
        start = time.time()
        deadline = start + duration + max_wait_extra
        while sum(len(x) for x in xs) < target and time.time() < deadline:
            chunk, t = self.inlet.pull_chunk(timeout=0.25, max_samples=256)
            if chunk:
                a = np.asarray(chunk, dtype=np.float64)
                if a.ndim == 2 and a.shape[1] >= self.nch:
                    xs.append(a[:, :self.nch])
                    ts.extend(t)
        if not xs:
            return np.empty((0, self.nch)), np.empty(0)
        x = np.vstack(xs)
        t = np.asarray(ts, dtype=float)
        if len(x) > target:
            x = x[-target:]
            t = t[-target:]
        return x, t

    def drain(self, seconds=0.15):
        self.pull(seconds, max_wait_extra=1)


def quality_report(x, fs=DEFAULT_FS):
    x = np.asarray(x, float)
    report = []
    if x.ndim != 2 or x.shape[1] != 4:
        return ["数据形状错误"]
    if len(x) < int(fs * 0.5): report.append("数据长度不足")
    if not np.all(np.isfinite(x)): report.append("存在 NaN/Inf")
    for c in range(4):
        v = x[:, c]
        if len(v) == 0: continue
        sd = float(np.nanstd(v)); ptp = float(np.nanmax(v)-np.nanmin(v))
        if sd < 1e-8 or ptp < 1e-7: report.append(f"CH{c+1}疑似平直/无信号")
        if np.mean(np.abs(v) > 1e5) > 0.01: report.append(f"CH{c+1}疑似饱和/数值异常")
        if sd > 1e5: report.append(f"CH{c+1}幅值异常大")
    return report


def filter_signal(x, fs=DEFAULT_FS, low=20, high=100, notch=50):
    from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
    x = np.asarray(x, dtype=np.float64)
    if len(x) < max(40, int(fs*0.25)): return x.copy()
    nyq = fs/2
    hi = min(high, nyq*0.8)
    if low >= hi: raise ValueError("滤波参数错误：低截止频率必须小于高截止频率。")
    sos = butter(4, [low, hi], btype='bandpass', fs=fs, output='sos')
    # zero-phase filtering for offline trials; live mode uses short buffered segments
    y = sosfiltfilt(sos, x, axis=0)
    if notch and notch < nyq-1:
        b, a = iirnotch(notch, Q=30, fs=fs)
        y = filtfilt(b, a, y, axis=0)
    return y


def window_features(x, fs=DEFAULT_FS, feature_set='25'):
    """x: samples x 4; 5 time features/channel plus relative RMS features."""
    x = np.asarray(x, float)
    if len(x) < 10: raise ValueError("窗口太短")
    feats=[]; rmss=[]
    for c in range(x.shape[1]):
        v=x[:,c]; av=np.abs(v)
        rms=float(np.sqrt(np.mean(v*v))+1e-12)
        mav=float(np.mean(av))
        wl=float(np.sum(np.abs(np.diff(v))))
        # Noise-tolerant ZC/SSC thresholds
        eps=0.01*(np.std(v)+1e-12)
        zc=int(np.sum(((v[:-1]*v[1:])<0) & (np.abs(np.diff(v))>=eps)))
        d1=np.diff(v)
        ssc=int(np.sum(((d1[:-1]*d1[1:])<0) & (np.abs(np.diff(d1))>=eps))) if len(d1)>1 else 0
        feats.extend([rms,mav,wl,zc,ssc]); rmss.append(rms)
    if feature_set == '20': return np.asarray(feats, dtype=float)
    r=np.asarray(rmss)
    rel=r/(np.sum(r)+1e-12)
    ratio=np.log((r[0]+r[1]+1e-12)/(r[2]+r[3]+1e-12))
    return np.concatenate([np.asarray(feats), rel, [ratio]])


def extract_trial_windows(raw, fs=DEFAULT_FS, win_s=.2, step_s=.1, feature_set='25', action_start_s=.5, action_end_s=3.5):
    x=filter_signal(raw, fs)
    start=int(action_start_s*fs); end=min(len(x), int(action_end_s*fs))
    x=x[start:end]
    w=int(win_s*fs); step=int(step_s*fs)
    out=[]
    for i in range(0, len(x)-w+1, step): out.append(window_features(x[i:i+w], fs, feature_set))
    return np.asarray(out)


def load_dataset(data_dir):
    p=Path(data_dir)
    files=sorted(p.glob('rs_trials_*.npz'))
    if not files: raise RuntimeError(f"{p} 下没有 rs_trials_*.npz，请先运行 collect。")
    raws=[]; labels=[]; runs=[]; ids=[]
    for f in files:
        z=np.load(f, allow_pickle=True)
        raws.extend(list(z['raw']))
        labels.extend(z['labels'].astype(int).tolist())
        runs.extend(z['runs'].astype(int).tolist())
        ids.extend(z['trial_ids'].astype(str).tolist())
    return raws, np.asarray(labels), np.asarray(runs), np.asarray(ids)


def cmd_check(a):
    s=EMGStream(a.stream_name,a.stream_type,a.fs,4,a.timeout)
    print("开始读取4秒数据...")
    x,t=s.pull(4)
    print(f"接收样本数={len(x)}，理论值约={4*a.fs:.0f}，实际估计采样率={((len(t)-1)/(t[-1]-t[0])) if len(t)>1 and t[-1]>t[0] else float('nan'):.2f} Hz")
    for i in range(4):
        if len(x): print(f"CH{i+1}: mean={np.mean(x[:,i]):.5g}, std={np.std(x[:,i]):.5g}, p-p={np.ptp(x[:,i]):.5g}")
    q=quality_report(x,a.fs)
    print("质量检查：" + ("PASS" if not q else "WARNING: "+"；".join(q)))


def cmd_collect(a):
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    s=EMGStream(a.stream_name,a.stream_type,a.fs,4,a.timeout)
    print("\n采集说明：保持坐姿和前臂姿势一致；不要用最大力量；动作期间不要移动电极/手臂。")
    print("每个Run 20个Trial（石头10、剪刀10），共4个Run、80个Trial。")
    print("每Trial：放松3秒 -> 准备1秒 -> 动作4秒 -> 恢复2.5秒。")
    input("准备好后按 Enter 开始 Run1...")
    all_manifest=[]
    for run in range(1,5):
        labels=[0]*10+[1]*10; random.Random(a.seed+run).shuffle(labels)
        raws=[]; kept=[]; tids=[]
        print(f"\n========== RUN {run}/4 ==========")
        for j,label in enumerate(labels,1):
            gesture=LABELS[label]
            print(f"Trial {j:02d}/20：即将进行 {gesture}。先放松3秒。")
            s.pull(3.0); print("准备..."); s.pull(1.0)
            print(f"现在做：{'石头（握拳）' if label==0 else '剪刀（食指中指伸展）'}，保持4秒！")
            x,t=s.pull(4.0)
            print("恢复放松2.5秒..."); s.pull(2.5)
            q=quality_report(x,a.fs)
            if len(x) < int(a.fs*3.5): q.append(f"样本不足({len(x)})")
            if q:
                print("  [WARNING] " + "；".join(q))
                if not a.keep_warning:
                    ans=input("  该Trial不通过，是否立即重做？输入 y 重做，其他键保留：").strip().lower()
                    if ans=='y':
                        print("  重做：放松3秒..."); s.pull(3); s.pull(1); print("  动作4秒..."); x,t=s.pull(4); s.pull(2.5)
            raws.append(x); kept.append(label); tid=f"R{run}_T{j:02d}"; tids.append(tid)
            all_manifest.append({'trial_id':tid,'run':run,'label':gesture,'samples':len(x),'quality':'WARNING' if q else 'PASS','warnings':'；'.join(q)})
            print(f"  已保存内存：{len(x)} samples；{('WARNING' if q else 'PASS')}")
        fn=out/f"rs_trials_{stamp()}_run{run}.npz"
        np.savez_compressed(fn,raw=np.asarray(raws,dtype=object),labels=np.asarray(kept),runs=np.full(len(kept),run),trial_ids=np.asarray(tids),fs=a.fs)
        print(f"Run{run} 已保存：{fn}")
        if run<4: input("休息至少1分钟，准备好后按 Enter 继续下一Run...")
    mf=out/f"collection_manifest_{stamp()}.csv"
    with open(mf,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(all_manifest[0].keys())); w.writeheader(); w.writerows(all_manifest)
    print(f"\n采集结束。Manifest：{mf}\n数据目录：{out}")


def cmd_train(a):
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
    import joblib
    raws,labels,runs,ids=load_dataset(a.data)
    if set(np.unique(labels)) != {0,1}: raise RuntimeError("数据必须同时包含 Rock(0) 和 Scissors(1)。")
    rows=[]
    for fsname in ['20','25']:
        trial_feats=[]; trial_y=[]; trial_run=[]; trial_id=[]
        for raw,y,r,tid in zip(raws,labels,runs,ids):
            try:
                fw=extract_trial_windows(raw,a.fs,feature_set=fsname)
                if len(fw)<5: continue
                trial_feats.append(fw); trial_y.append(y); trial_run.append(r); trial_id.append(tid)
            except Exception as e: print(f"跳过 {tid}: {e}")
        # fit windows on Run1+2; choose candidate by trial-level vote on Run3
        candidates={
          'LinearSVM':Pipeline([('scale',StandardScaler()),('clf',SVC(kernel='linear',C=1,class_weight='balanced'))]),
          'RBFSVM':Pipeline([('scale',StandardScaler()),('clf',SVC(kernel='rbf',C=1,gamma='scale',class_weight='balanced'))]),
          'LDA':Pipeline([('scale',StandardScaler()),('clf',LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'))]),
          'RandomForest':RandomForestClassifier(n_estimators=300,max_depth=None,min_samples_leaf=2,class_weight='balanced',random_state=42,n_jobs=-1)
        }
        for name,model in candidates.items():
            tr_idx=[i for i,r in enumerate(trial_run) if r in (1,2)]
            va_idx=[i for i,r in enumerate(trial_run) if r==3]
            te_idx=[i for i,r in enumerate(trial_run) if r==4]
            if not tr_idx or not va_idx or not te_idx:
                print(f"{fsname}维/{name}: 缺少Run1-4完整数据，跳过严格验证。可先采集4个Run。")
                continue
            Xtr=np.vstack([trial_feats[i] for i in tr_idx]); ytr=np.concatenate([np.full(len(trial_feats[i]),trial_y[i]) for i in tr_idx])
            model.fit(Xtr,ytr)
            def trial_predict(indices):
                yy=[]; pp=[]
                for i in indices:
                    pred=model.predict(trial_feats[i])
                    vote=int(np.mean(pred)>=.5)
                    yy.append(trial_y[i]); pp.append(vote)
                return np.asarray(yy),np.asarray(pp)
            yv,pv=trial_predict(va_idx)
            val_ba=balanced_accuracy_score(yv,pv)
            rows.append({'features':fsname,'model':name,'run3_balanced_accuracy':val_ba,'run3_accuracy':accuracy_score(yv,pv),'pipeline':model,'trial_feats':trial_feats,'trial_y':trial_y,'trial_run':trial_run,'trial_id':trial_id})
            print(f"特征{fsname}维 | {name:12s} | Run3 Trial BA={val_ba*100:.1f}% | Acc={accuracy_score(yv,pv)*100:.1f}%")
    if not rows: raise RuntimeError("没有可训练模型。检查是否有4个Run且每类都有数据。")
    best=max(rows,key=lambda z:z['run3_balanced_accuracy'])
    print(f"\nRun3选型：{best['features']}维 + {best['model']}，BA={best['run3_balanced_accuracy']*100:.1f}%")
    # Refit chosen estimator on all Runs 1-3, then one untouched Run4 evaluation
    idx_train=[i for i,r in enumerate(best['trial_run']) if r in (1,2,3)]
    idx_test=[i for i,r in enumerate(best['trial_run']) if r==4]
    model=best['pipeline']
    X=np.vstack([best['trial_feats'][i] for i in idx_train]); y=np.concatenate([np.full(len(best['trial_feats'][i]),best['trial_y'][i]) for i in idx_train])
    model.fit(X,y)
    yt=[]; yp=[]; detail=[]
    for i in idx_test:
        pred=model.predict(best['trial_feats'][i]); vote=int(np.mean(pred)>=.5)
        yt.append(best['trial_y'][i]); yp.append(vote)
        detail.append({'trial_id':best['trial_id'][i],'true':LABELS[best['trial_y'][i]],'pred':LABELS[vote],'correct':bool(vote==best['trial_y'][i]),'window_scissors_fraction':float(np.mean(pred))})
    ba=balanced_accuracy_score(yt,yp); acc=accuracy_score(yt,yp); cm=confusion_matrix(yt,yp,labels=[0,1])
    print(f"\n========== Run4 独立验证（Trial级）==========\nAccuracy={acc*100:.1f}% | Balanced Accuracy={ba*100:.1f}%")
    print("混淆矩阵（行=真实 Rock/Scissors，列=预测 Rock/Scissors）：")
    print(cm)
    print(classification_report(yt,yp,target_names=['Rock','Scissors'],zero_division=0))
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    bundle={'model':model,'feature_set':best['features'],'fs':a.fs,'channels':4,'window_s':.2,'step_s':.1,'action_start_s':.5,'action_end_s':3.5,'labels':LABELS,'stream_name':a.stream_name,'stream_type':a.stream_type,'selected_model':best['model'],'selection_run3_ba':float(best['run3_balanced_accuracy']),'run4_accuracy':float(acc),'run4_balanced_accuracy':float(ba),'created_at':stamp()}
    modelpath=out/f"rs_binary_model_{stamp()}.joblib"; joblib.dump(bundle,modelpath)
    pd.DataFrame(detail).to_csv(out/f"run4_predictions_{stamp()}.csv",index=False,encoding='utf-8-sig')
    summary={'selected_model':best['model'],'feature_set':best['features'],'run3_balanced_accuracy':float(best['run3_balanced_accuracy']),'run4_accuracy':float(acc),'run4_balanced_accuracy':float(ba),'confusion_matrix':cm.tolist(),'n_run4_trials':len(yt),'note':'Run4 is independent only if collection order and trial IDs were not reused in training.'}
    with open(out/f"training_summary_{stamp()}.json",'w',encoding='utf-8') as f: json.dump(summary,f,ensure_ascii=False,indent=2)
    print(f"模型已保存：{modelpath}")


def predict_segment(raw,bundle):
    fw=extract_trial_windows(raw,bundle['fs'],bundle['window_s'],bundle['step_s'],bundle['feature_set'],bundle['action_start_s'],bundle['action_end_s'])
    if len(fw)==0: return None,0.0,0
    preds=bundle['model'].predict(fw)
    frac=float(np.mean(preds==1)); vote=int(frac>=.5)
    # confidence = majority proportion, not calibrated probability
    conf=max(frac,1-frac)
    return vote,conf,len(fw)


def cmd_live(a):
    import joblib
    bundle=joblib.load(a.model)
    s=EMGStream(a.stream_name or bundle.get('stream_name',DEFAULT_STREAM_NAME),a.stream_type or bundle.get('stream_type',DEFAULT_STREAM_TYPE),a.fs,4,a.timeout)
    print("\n手动触发实时识别。按 Enter 开始；程序提示后做动作并保持4秒。输入 q 退出。")
    while True:
        cmd=input("\n按 Enter 开始识别 / q退出：").strip().lower()
        if cmd=='q': break
        print("准备1秒..."); s.pull(1)
        print("现在做石头或剪刀，保持4秒！")
        raw,t=s.pull(4)
        q=quality_report(raw,a.fs)
        if len(raw)<int(a.fs*3.5): q.append(f"样本不足({len(raw)})")
        if q:
            print("信号质量警告："+"；".join(q)+"。本次不输出可信分类。")
            continue
        vote,conf,n=predict_segment(raw,bundle)
        if vote is None: print("窗口不足，无法判断。"); continue
        print(f"识别结果：{'✊ 石头 (Rock)' if vote==0 else '✌️ 剪刀 (Scissors)'} | 投票一致率={conf*100:.1f}% | 窗口数={n}")
        actual=input("真实动作？输入 r=石头 / s=剪刀 / Enter跳过：").strip().lower()
        if actual in ('r','s'):
            correct=(actual==('r' if vote==0 else 's'))
            print("本次："+("正确" if correct else "错误"))


def cmd_auto(a):
    """Experimental auto mode. Uses RMS envelope threshold calibrated from 10s rest."""
    import joblib
    bundle=joblib.load(a.model)
    s=EMGStream(a.stream_name or bundle.get('stream_name',DEFAULT_STREAM_NAME),a.stream_type or bundle.get('stream_type',DEFAULT_STREAM_TYPE),a.fs,4,a.timeout)
    print("请完全放松，进行10秒静息校准...")
    base,_=s.pull(10)
    if len(base)<a.fs*8: raise RuntimeError("静息校准数据不足")
    b=filter_signal(base,a.fs)
    env=np.sqrt(np.mean(b*b,axis=1))
    rest=float(np.median(env)); mad=float(np.median(np.abs(env-rest)))+1e-9
    on=max(rest+6*mad,rest*1.8,1e-6); off=max(rest+3*mad,rest*1.3,1e-6)
    print(f"静息RMS={rest:.5g}，ON={on:.5g}，OFF={off:.5g}。按 Ctrl+C 退出。")
    active=False; buf=np.empty((0,4)); quiet_since=None; last_pred=0; classified=False
    while True:
        x,t=s.pull(.1,max_wait_extra=1)
        if len(x)==0: continue
        buf=np.vstack([buf,x])[-int(a.fs*4.5):]
        if len(buf)<int(a.fs*.2): continue
        recent=filter_signal(buf[-int(a.fs*.2):],a.fs)
        level=float(np.sqrt(np.mean(recent*recent)))
        now=time.time()
        if not active and level>on:
            active=True; quiet_since=None; classified=False; print("检测到动作，正在分析...")
        if active:
            if level<off:
                if quiet_since is None: quiet_since=now
            else: quiet_since=None
            if len(buf)>=int(a.fs*3.5) and not classified and now-last_pred>0.5:
                # classify last 4s once enough signal is present; require majority windows
                segment=buf[-int(a.fs*4):]
                if len(segment)>=int(a.fs*3.5):
                    q=quality_report(segment,a.fs)
                    if not q:
                        vote,conf,n=predict_segment(segment,bundle)
                        if vote is not None and conf>=.7:
                            print(f"结果：{'石头' if vote==0 else '剪刀'} | 窗口投票一致率 {conf*100:.0f}%")
                            last_pred=now; classified=True
            if quiet_since and now-quiet_since>=.6:
                active=False; quiet_since=None; classified=False; buf=np.empty((0,4)); print("动作结束，等待下一次动作。")


def build_parser():
    p=argparse.ArgumentParser(description='OpenBCI 4CH sEMG Rock vs Scissors')
    sub=p.add_subparsers(dest='cmd',required=True)
    def common(q,stream=True):
        q.add_argument('--fs',type=float,default=DEFAULT_FS)
        if stream:
            q.add_argument('--stream-name',default=DEFAULT_STREAM_NAME)
            q.add_argument('--stream-type',default=DEFAULT_STREAM_TYPE)
            q.add_argument('--timeout',type=float,default=10)
    q=sub.add_parser('check',help='检查LSL流和四通道'); common(q); q.set_defaults(func=cmd_check)
    q=sub.add_parser('collect',help='采集4 Run共80 Trial'); common(q); q.add_argument('--out',default='data'); q.add_argument('--seed',type=int,default=20261125); q.add_argument('--keep-warning',action='store_true',help='质控警告时不询问重做，直接保留'); q.set_defaults(func=cmd_collect)
    q=sub.add_parser('train',help='训练模型并Run4独立验证'); common(q); q.add_argument('--data',default='data'); q.add_argument('--out',default='models'); q.set_defaults(func=cmd_train)
    q=sub.add_parser('live',help='手动触发实时识别'); q.add_argument('--model',required=True); q.add_argument('--stream-name',default=None); q.add_argument('--stream-type',default=None); q.add_argument('--fs',type=float,default=DEFAULT_FS); q.add_argument('--timeout',type=float,default=10); q.set_defaults(func=cmd_live)
    q=sub.add_parser('auto',help='实验性自动触发实时识别'); q.add_argument('--model',required=True); q.add_argument('--stream-name',default=None); q.add_argument('--stream-type',default=None); q.add_argument('--fs',type=float,default=DEFAULT_FS); q.add_argument('--timeout',type=float,default=10); q.set_defaults(func=cmd_auto)
    q=sub.add_parser('daily',help='显示推荐运行顺序'); q.set_defaults(func=lambda a: print('1) python rs_binary_4ch.py check\n2) python rs_binary_4ch.py collect --out data\n3) python rs_binary_4ch.py train --data data --out models\n4) python rs_binary_4ch.py live --model models\\rs_binary_model_xxx.joblib\n5) 模式稳定后再尝试 auto'))
    return p


def main():
    parser=build_parser(); args=parser.parse_args()
    try:
        if args.cmd!='daily': ensure_deps()
        args.func(args)
    except KeyboardInterrupt:
        print("\n用户中断。")
    except Exception as e:
        print(f"\n错误：{e}",file=sys.stderr)
        sys.exit(2)

if __name__=='__main__': main()
