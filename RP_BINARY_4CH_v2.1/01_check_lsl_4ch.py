# -*- coding: utf-8 -*-
"""检查 OpenBCI LSL 流和四通道信号。"""
import sys
from rp_core import main
if __name__ == '__main__': sys.argv = [sys.argv[0], 'check', *sys.argv[1:]]; main()
