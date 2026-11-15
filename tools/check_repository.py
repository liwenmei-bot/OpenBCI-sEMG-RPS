#!/usr/bin/env python3
"""Validate source/configuration; optional CLI checks never connect to hardware."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--help-checks', action='store_true', help='Run --help for active scripts and collection configuration self-tests')
    parser.add_argument('--verify-snapshot', action='store_true', help='Verify the distributed initial file manifest; source edits intentionally change these hashes')
    args = parser.parse_args(argv)
    try:
        files = [path for path in ROOT.rglob('*.py') if not any(part in
            {'__pycache__', '.git', '.venv', 'venv', 'outputs', 'data', 'models'} for part in path.relative_to(ROOT).parts)]
        for path in files: ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        config = json.loads((ROOT/'configs/rp_collection.json').read_text(encoding='utf-8'))
        assert config['markers'] == {'rock': 11, 'paper': 13, 'end': 20, 'reject': 90}, 'RP marker mapping changed'
        assert config['protocol']['formal'] == {'runs': 4, 'trials_per_run': 12}, 'RP protocol changed'
        for name in ['run.py', 'README.md', 'requirements.txt', 'binary/run_rp_binary.py',
                'binary/07_live_rp_binary_label_4ch.py', 'binary/07_live_rp_binary_extended_4ch.py',
                'rps/08_today_only_svm_rps_4ch.py', 'tools/diagnose_rp_signal.py']:
            assert (ROOT/name).is_file(), 'Missing file: ' + name
        if args.verify_snapshot:
            manifest = json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
            for item in manifest['files']:
                path = ROOT/item['path']
                assert path.is_file(), 'Missing snapshot file: '+item['path']
                assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], 'Changed snapshot file: '+item['path']
            print('Snapshot hashes verified:', len(manifest['files']))
        if args.help_checks:
            targets = sorted((ROOT/'binary').glob('*.py')) + sorted((ROOT/'rps').glob('*.py'))
            targets += [ROOT/'rps/collection/openbci_rps_collection.py', ROOT/'tools/diagnose_rp_signal.py',
                ROOT/'tools/compare_rp_live_versions.py']
            commands = [[sys.executable, str(path), '--help'] for path in targets]
            commands += [[sys.executable, str(ROOT/'binary/00_collect_rp_binary_4ch.py'), '--self-test'],
                [sys.executable, str(ROOT/'rps/collection/openbci_rps_collection.py'), '--self-test'],
                [sys.executable, str(ROOT/'run.py'), 'live', '--help']]
            for command in commands:
                result = subprocess.run(command, capture_output=True, text=True, timeout=45, cwd=ROOT)
                if result.returncode:
                    raise RuntimeError('CLI failed: %s\n%s\n%s' % (command, result.stdout[-1500:], result.stderr[-1500:]))
            print('CLI/configuration checks passed:', len(commands))
        print('Python source syntax passed:', len(files))
        return 0
    except (OSError, ValueError, AssertionError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print('[CHECK FAILED] '+str(exc), file=sys.stderr); return 1


if __name__ == '__main__': raise SystemExit(main())
