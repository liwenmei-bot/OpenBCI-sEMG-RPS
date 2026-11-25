# -*- coding: utf-8 -*-
"""检查 OpenBCI LSL 流、通道数、采样率和四通道信号质量。"""
import argparse
from rs_core import EMGStream, quality_report, DEFAULT_STREAM_NAME, DEFAULT_STREAM_TYPE, DEFAULT_FS

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stream-name',default=DEFAULT_STREAM_NAME); p.add_argument('--stream-type',default=DEFAULT_STREAM_TYPE)
    p.add_argument('--fs',type=float,default=DEFAULT_FS); p.add_argument('--timeout',type=float,default=10)
    a=p.parse_args(); s=EMGStream(a.stream_name,a.stream_type,a.fs,4,a.timeout)
    x,t=s.pull(10)
    print(f'收到 {len(x)} 点；理论约 {10*a.fs:.0f} 点')
    if len(t)>1 and t[-1]>t[0]: print(f'时间戳估计采样率：{(len(t)-1)/(t[-1]-t[0]):.2f} Hz')
    for c in range(4): print(f'CH{c+1}: mean={x[:,c].mean():.6g}, SD={x[:,c].std():.6g}, P-P={x[:,c].ptp():.6g}')
    q=quality_report(x,a.fs); print('质控：'+('PASS' if not q else 'WARNING: '+'；'.join(q)))
if __name__=='__main__': main()
