# -*- coding: utf-8 -*-
"""当天流程入口：检查、采集80 Trial、预处理、训练、Run4验证、回放。"""
import argparse, subprocess, sys
from pathlib import Path

def run(cmd,cwd):
    print('\n执行：',' '.join(map(str,cmd))); rc=subprocess.call(list(map(str,cmd)),cwd=cwd)
    if rc: raise SystemExit(rc)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='config_rp.json'); p.add_argument('--stream-name',default=None); p.add_argument('--stream-type',default=None); a=p.parse_args()
    root=Path(__file__).resolve().parent; py=sys.executable; cfg=['--config',str((root/a.config).resolve())]
    stream=[]
    if a.stream_name: stream += ['--stream-name',a.stream_name]
    if a.stream_type: stream += ['--stream-type',a.stream_type]
    print('流程：检查 → 采集4×20 Trial → 预处理 → 分组CV选型 → Run1-3训练 → Run4独立验证 → 离线回放')
    run([py,root/'rp_core.py',*cfg,'check',*stream],root)
    if input('检查通过后按Enter采集；输入q退出：').strip().lower()=='q': return
    run([py,root/'rp_core.py',*cfg,'collect','--out',root/'data',*stream],root)
    run([py,root/'rp_core.py',*cfg,'preprocess','--data',root/'data','--out',root/'outputs'/'preprocess'],root)
    run([py,root/'rp_core.py',*cfg,'train','--data',root/'data','--out',root/'models'],root)
    models=sorted((root/'models').glob('rp_binary_model_*.joblib'),key=lambda x:x.stat().st_mtime,reverse=True)
    model=models[0] if models else None
    if model:
        run([py,root/'05_replay_rp_4ch.py','--config',root/a.config,'--data',root/'data','--model',model,'--out',root/'outputs'/'replay'],root)
        print('\n手动实时识别命令：'); print(f'python 06_live_rp_manual_4ch.py --model "{model}"')
    print('\n当天流程结束。自动门控暂不建议使用。')
if __name__=='__main__': main()
