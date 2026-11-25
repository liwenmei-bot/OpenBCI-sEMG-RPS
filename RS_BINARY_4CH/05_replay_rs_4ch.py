# -*- coding: utf-8 -*-
"""离线回放：用已训练模型逐 Trial 预测，导出逐 Trial 结果。"""
import argparse, csv
from pathlib import Path
import numpy as np, joblib
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
from rs_core import load_dataset, extract_trial_windows, LABELS, quality_report, DEFAULT_FS

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data'); p.add_argument('--model',required=True); p.add_argument('--out',default='outputs/replay'); p.add_argument('--fs',type=float,default=DEFAULT_FS)
    a=p.parse_args(); b=joblib.load(a.model); raws,ys,runs,ids=load_dataset(a.data); rows=[]
    for raw,y,r,tid in zip(raws,ys,runs,ids):
        q=quality_report(raw,a.fs)
        if q: rows.append({'trial_id':tid,'run':int(r),'true':LABELS[int(y)],'pred':'REJECT','correct':False,'vote_fraction':None,'warning':'；'.join(q)}); continue
        fw=extract_trial_windows(raw,b['fs'],b['window_s'],b['step_s'],b['feature_set'],b['action_start_s'],b['action_end_s'])
        if not len(fw): pred=None; frac=None
        else:
            pp=b['model'].predict(fw); frac=float(np.mean(pp==1)); pred=int(frac>=.5)
        rows.append({'trial_id':tid,'run':int(r),'true':LABELS[int(y)],'pred':LABELS[pred] if pred is not None else 'REJECT','correct':bool(pred==int(y)) if pred is not None else False,'vote_fraction':frac,'warning':''})
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    with open(out/'replay_predictions.csv','w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    valid=[z for z in rows if z['pred'] in LABELS.values()]
    if valid:
        yt=[0 if z['true']=='Rock' else 1 for z in valid]; yp=[0 if z['pred']=='Rock' else 1 for z in valid]
        print(f"有效Trial={len(valid)}；Accuracy={accuracy_score(yt,yp)*100:.1f}%；BA={balanced_accuracy_score(yt,yp)*100:.1f}%")
        print('混淆矩阵（真实行/预测列，Rock、Scissors）：\n',confusion_matrix(yt,yp,labels=[0,1]))
        print(classification_report(yt,yp,target_names=['Rock','Scissors'],zero_division=0))
    print('逐Trial结果：',(out/'replay_predictions.csv').resolve())
if __name__=='__main__': main()
