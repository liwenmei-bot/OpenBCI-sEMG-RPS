# -*- coding: utf-8 -*-
"""当天工作流入口：显示并逐步执行检查、采集、训练、回放、实时识别命令。"""
import argparse, subprocess, sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--stream-name',default='rps_raw'); p.add_argument('--stream-type',default='EMG'); p.add_argument('--model',default=''); a=p.parse_args()
    root=Path(__file__).resolve().parent
    commands=[
      [sys.executable,str(root/'01_check_lsl_4ch.py'),'--stream-name',a.stream_name,'--stream-type',a.stream_type],
      [sys.executable,str(root/'02_collect_rs_4ch.py'),'--stream-name',a.stream_name,'--stream-type',a.stream_type,'--out',str(root/'data')],
      [sys.executable,str(root/'03_preprocess_rs_4ch.py'),'--data',str(root/'data'),'--out',str(root/'outputs'/'preprocess')],
      [sys.executable,str(root/'04_train_rs_4ch.py'),'--stream-name',a.stream_name,'--stream-type',a.stream_type,'--data',str(root/'data'),'--out',str(root/'models')],
    ]
    print('工作流：检查 -> 采集80 Trial -> 训练并Run4独立验证。每一步结束后请确认再继续。')
    for cmd in commands:
        print('\n执行：',' '.join(cmd)); rc=subprocess.call(cmd,cwd=root)
        if rc: raise SystemExit(rc)
        if input('该步骤完成。按 Enter 继续，输入 q 退出：').strip().lower()=='q': return
    models=sorted((root/'models').glob('rs_binary_model_*.joblib'),key=lambda x:x.stat().st_mtime,reverse=True)
    model=a.model or (str(models[0]) if models else '')
    if model:
        print('模型：',model)
        replay=[sys.executable,str(root/'05_replay_rs_4ch.py'),'--data',str(root/'data'),'--model',model,'--out',str(root/'outputs'/'replay')]
        print('执行离线回放：',' '.join(replay)); rc=subprocess.call(replay,cwd=root)
        if rc: raise SystemExit(rc)
        print('手动实时命令：',sys.executable,'06_live_rs_manual_4ch.py --model',repr(model))
    else: print('未找到模型文件，请检查训练输出。')
if __name__=='__main__': main()
