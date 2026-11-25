# -*- coding: utf-8 -*-
"""实验性自动门控实时识别；建议手动模式稳定后再使用。"""
import argparse
from rs_core import cmd_auto, DEFAULT_FS

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--stream-name',default=None); p.add_argument('--stream-type',default=None); p.add_argument('--fs',type=float,default=DEFAULT_FS); p.add_argument('--timeout',type=float,default=10)
    cmd_auto(p.parse_args())
if __name__=='__main__': main()
