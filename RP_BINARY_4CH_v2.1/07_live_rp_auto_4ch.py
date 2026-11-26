# -*- coding: utf-8 -*-
"""自动门控暂时禁用：需先完成在线因果滤波和动作起点状态机验证。"""
import sys
from rp_core import main
if __name__ == '__main__': sys.argv = [sys.argv[0], 'auto', *sys.argv[1:]]; main()
