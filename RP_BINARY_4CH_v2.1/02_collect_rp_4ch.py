# -*- coding: utf-8 -*-
"""采集4 Run × 20 Trial（每Run石头10、布10）。"""
import sys
from rp_core import main
if __name__ == '__main__': sys.argv = [sys.argv[0], 'collect', *sys.argv[1:]]; main()
