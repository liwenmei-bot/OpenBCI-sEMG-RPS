# -*- coding: utf-8 -*-
"""分组交叉验证选型、Run1-3拟合、Run4独立Trial级验证。"""
import sys
from rp_core import main
if __name__ == '__main__': sys.argv = [sys.argv[0], 'train', *sys.argv[1:]]; main()
