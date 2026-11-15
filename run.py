#!/usr/bin/env python3
"""OpenBCI 4CH sEMG project entry. Default workflow: rock/paper binary.

python run.py collect --mode formal --subject USER01 --session DAY01
python run.py init --history-input HISTORY.txt --today-input TODAY.txt
python run.py day --base-model BASE_MODEL.joblib --today-input TODAY.txt
python run.py live --run RUN_DIRECTORY
python run.py diagnose --logs outputs
python run.py check --help-checks
python run.py rps --help
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RPS = {
    'collect': 'collection/openbci_rps_collection.py',
    'preprocess': '01_preprocess_openbci_rps_4ch.py',
    'train': '02_train_svm_openbci_rps_4ch.py',
    'calibrate': '03_calibrate_svm_openbci_rps_4ch.py',
    'evaluate': '04_evaluate_svm_cross_session_4ch.py',
    'replay': '05_replay_svm_openbci_rps_4ch.py',
    'stream': '06_live_svm_openbci_rps_4ch.py',
    'live': '07_live_svm_rps_label_4ch.py',
    'today-only': '08_today_only_svm_rps_4ch.py',
    'daily': 'run_daily_calibration.py',
}


def command(argv):
    if not argv or argv[0] in ['-h', '--help']:
        print(__doc__)
        print('Commands: collect, init, day, live, replay, streams, diagnose, compare, check, rps')
        return None
    name, rest = argv[0], list(argv[1:])
    if name == 'collect': target, prefix = 'binary/00_collect_rp_binary_4ch.py', []
    elif name in ['init', 'day', 'live', 'replay']:
        target, prefix = 'binary/run_rp_binary.py', [name]
    elif name == 'streams': target, prefix = 'binary/07_live_rp_binary_label_4ch.py', ['--list-streams']
    elif name == 'diagnose': target, prefix = 'tools/diagnose_rp_signal.py', []
    elif name == 'compare': target, prefix = 'tools/compare_rp_live_versions.py', []
    elif name == 'check': target, prefix = 'tools/check_repository.py', []
    elif name == 'rps':
        if not rest or rest[0] in ['-h', '--help']:
            print('Three-class historical workflow: python run.py rps STAGE [arguments]')
            print('Stages: ' + ', '.join(RPS))
            return None
        stage = rest.pop(0)
        if stage not in RPS: raise ValueError('Unknown RPS stage: ' + stage)
        target, prefix = 'rps/'+RPS[stage], []
    else: raise ValueError('Unknown command: %s; run python run.py --help' % name)
    path = ROOT/target
    if not path.is_file(): raise FileNotFoundError(path)
    return [sys.executable, str(path), *prefix, *rest]


def main(argv=None):
    try:
        args = sys.argv[1:] if argv is None else argv
        parts = command(args)
        return 0 if parts is None else subprocess.run(parts).returncode
    except KeyboardInterrupt: return 130
    except (OSError, ValueError) as exc:
        print('[Failed] '+str(exc), file=sys.stderr); return 1


if __name__ == '__main__': raise SystemExit(main())
