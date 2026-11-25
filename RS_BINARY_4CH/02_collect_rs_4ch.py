# -*- coding: utf-8 -*-
"""采集当天石头/剪刀数据：4 Run，每 Run 每类 10 Trial。"""
import argparse
from rs_core import cmd_collect, DEFAULT_STREAM_NAME, DEFAULT_STREAM_TYPE, DEFAULT_FS

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='data'); p.add_argument('--stream-name',default=DEFAULT_STREAM_NAME); p.add_argument('--stream-type',default=DEFAULT_STREAM_TYPE)
    p.add_argument('--fs',type=float,default=DEFAULT_FS); p.add_argument('--timeout',type=float,default=10); p.add_argument('--seed',type=int,default=20261125); p.add_argument('--keep-warning',action='store_true')
    cmd_collect(p.parse_args())
if __name__=='__main__': main()
