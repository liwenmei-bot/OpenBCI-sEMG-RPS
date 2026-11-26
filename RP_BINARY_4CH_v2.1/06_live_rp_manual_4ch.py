# -*- coding: utf-8 -*-
"""手动触发批量实时识别：按Enter，准备1秒，动作4秒。"""
import sys
from rp_core import main
if __name__ == '__main__': sys.argv = [sys.argv[0], 'live', *sys.argv[1:]]; main()
