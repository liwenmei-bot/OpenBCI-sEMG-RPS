# -*- coding: utf-8 -*-
"""预处理并导出20/25维窗口特征和Trial质控表。"""
import sys
from rp_core import main
if __name__ == '__main__': sys.argv = [sys.argv[0], 'preprocess', *sys.argv[1:]]; main()
