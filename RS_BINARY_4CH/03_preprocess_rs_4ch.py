# -*- coding: utf-8 -*-
"""离线预处理：滤波、Trial 质量检查、提取 20/25 维窗口特征并导出 CSV。"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from rs_core import load_dataset, extract_trial_windows, quality_report, DEFAULT_FS

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data'); p.add_argument('--out',default='outputs/preprocess'); p.add_argument('--fs',type=float,default=DEFAULT_FS)
    a=p.parse_args(); raws,ys,runs,ids=load_dataset(a.data); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    rows=[]; qc=[]
    for raw,y,r,tid in zip(raws,ys,runs,ids):
        warnings=quality_report(raw,a.fs)
        qc.append({'trial_id':tid,'run':int(r),'label':'Rock' if y==0 else 'Scissors','n_samples':len(raw),'qc':'WARNING' if warnings else 'PASS','warnings':'；'.join(warnings)})
        for fsname in ('20','25'):
            try: feats=extract_trial_windows(raw,a.fs,feature_set=fsname)
            except Exception as e: print(f'{tid} {fsname}维跳过：{e}'); continue
            for wi,row in enumerate(feats):
                d={'trial_id':tid,'run':int(r),'label':int(y),'feature_set':fsname,'window':wi}
                d.update({f'f{k+1}':float(v) for k,v in enumerate(row)}); rows.append(d)
    pd.DataFrame(rows).to_csv(out/'window_features.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(qc).to_csv(out/'trial_qc.csv',index=False,encoding='utf-8-sig')
    print(f'完成：{len(qc)} Trial，{len(rows)} 个窗口特征行；输出 {out.resolve()}')
if __name__=='__main__': main()
