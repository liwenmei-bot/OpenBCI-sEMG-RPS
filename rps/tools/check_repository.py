#!/usr/bin/env python3
"""Check source integrity and CLI entry points without loading data or connecting hardware."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-cli", action="store_true", help="Only syntax/config/hash checks; does not check installed dependencies")
    args = parser.parse_args()
    snapshot = json.loads((ROOT / "SOURCE_SNAPSHOT.json").read_text(encoding="utf-8"))
    for row in snapshot["source_files"]:
        path = ROOT / row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise SystemExit("Source differs from the recorded snapshot: " + row["path"])
    paths = list(ROOT.glob("*.py")) + list((ROOT / "collection").glob("*.py")) + list((ROOT / "tests").glob("*.py"))
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path), feature_version=(3, 9))
    config = json.loads((ROOT / "collection/config_rps_emg.json").read_text(encoding="utf-8"))
    if config["protocol"]["formal"] != {"runs": 4, "trials_per_run": 18}:
        raise SystemExit("The example formal protocol must be 4 Runs x 18 Trials.")
    print("Syntax, source snapshot and 4-Run example config: OK", flush=True)
    if not args.skip_cli:
        for path in sorted(ROOT.glob("[0-9][0-9]_*.py")) + [ROOT / "run_daily_calibration.py"]:
            result = subprocess.run([sys.executable, str(path), "--help"], cwd=ROOT, capture_output=True, text=True)
            if result.returncode:
                raise SystemExit("CLI failed: %s\n%s" % (path.name, result.stderr))
            print("CLI OK: " + path.name, flush=True)
        subprocess.run([sys.executable, str(ROOT / "collection/openbci_rps_collection.py"), "--self-test"], cwd=ROOT, check=True)
    print("No participant data loaded and no UDP/LSL connection opened.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
