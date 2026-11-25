# -*- coding: utf-8 -*-
"""手动触发实时识别：按 Enter 后做动作，保持约 4 秒。"""
import argparse
from rs_core import cmd_live, DEFAULT_FS

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--stream-name',default=None); p.add_argument('--stream-type',default=None); p.add_argument('--fs',type=float,default=DEFAULT_FS); p.add_argument('--timeout',type=float,default=10)
    cmd_live(p.parse_args())
if __name__=='__main__': main()
