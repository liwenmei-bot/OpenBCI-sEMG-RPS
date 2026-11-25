# -*- coding: utf-8 -*-
"""训练并比较模型；Run1-2训练、Run3选型、Run1-3重训、Run4独立验证。"""
import argparse
from rs_core import cmd_train, DEFAULT_STREAM_NAME, DEFAULT_STREAM_TYPE, DEFAULT_FS

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data'); p.add_argument('--out',default='models'); p.add_argument('--stream-name',default=DEFAULT_STREAM_NAME); p.add_argument('--stream-type',default=DEFAULT_STREAM_TYPE); p.add_argument('--fs',type=float,default=DEFAULT_FS)
    cmd_train(p.parse_args())
if __name__=='__main__': main()
