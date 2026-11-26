# -*- coding: utf-8 -*-
"""对当前Session执行逐Trial离线回放。"""
import argparse, csv
from pathlib import Path
import numpy as np, joblib
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
from rp_core import load_dataset, extract_trial_windows, LABELS, quality_report, load_config, validate_bundle

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='config_rp.json'); p.add_argument('--data',default='data'); p.add_argument('--model',required=True); p.add_argument('--out',default='outputs/replay'); p.add_argument('--all-runs',action='store_true'); a=p.parse_args()
    cfg=load_config(a.config); b=joblib.load(a.model); raws,ys,runs,ids,qcs,session=load_dataset(a.data); rows=[]; validate_bundle(b)
    if not a.all_runs and Path(b['training_session']).resolve()!=session.resolve(): raise RuntimeError('默认Run4回放要求原训练Session')
    if a.all_runs: print('注意：包含训练数据，不是独立验证结果')
    for raw,y,r,tid,qc0 in zip(raws,ys,runs,ids,qcs):
        if not a.all_runs and r!=4: continue
        status,q,_=quality_report(raw,b['fs'],duration=b['config']['action_seconds'],cfg=b['config'])
        if status=='FAIL' or qc0 not in ('PASS','WARNING'): rows.append({'trial_id':tid,'run':int(r),'true':LABELS[int(y)],'pred':'REJECT','correct':False,'vote_fraction':'','warning':'；'.join(q) or '采集时FAIL'}); continue
        fw=extract_trial_windows(raw,b['config'],b['feature_set']); pp=b['model'].predict(fw); frac=float(np.mean(pp==1)); pred=int(frac>=.5)
        rows.append({'trial_id':tid,'run':int(r),'true':LABELS[int(y)],'pred':LABELS[pred],'correct':bool(pred==int(y)),'vote_fraction':frac,'warning':str(qc0)})
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    with open(out/'replay_predictions.csv','w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    valid=[z for z in rows if z['pred'] in LABELS.values()]
    if valid:
        yt=[0 if z['true']=='Rock' else 1 for z in valid]; yp=[0 if z['pred']=='Rock' else 1 for z in valid]
        print(f'Session={session}; 有效Trial={len(valid)}/{len(rows)}; Accuracy={accuracy_score(yt,yp)*100:.1f}%; BA={balanced_accuracy_score(yt,yp)*100:.1f}%')
        print('混淆矩阵（真实行/预测列，Rock/Paper）：\n',confusion_matrix(yt,yp,labels=[0,1])); print(classification_report(yt,yp,target_names=['Rock','Paper'],zero_division=0))
    print('逐Trial结果：',(out/'replay_predictions.csv').resolve())
if __name__=='__main__': main()
