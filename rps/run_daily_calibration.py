#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""串联既有01/03/06脚本；不更改算法，完成后打印07命令，不自动连接设备。"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def display_command(parts):
    return subprocess.list2cmdline(parts) if os.name == "nt" else shlex.join(parts)


def stage_directory(parent, expected_file):
    found = [p for p in parent.iterdir() if p.is_dir() and (p / expected_file).is_file()]
    if len(found) != 1:
        raise RuntimeError("无法唯一确定输出目录：%s（找到%d个包含%s的目录）" % (parent, len(found), expected_file))
    return found[0]


def run_stage(script, arguments):
    command = [sys.executable, str(ROOT / script), *map(str, arguments)]
    print("\n[执行] " + display_command(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="当天完整4 Run、72 Trial原始TXT")
    parser.add_argument("--base-model", required=True, type=Path, help="已有基础模型base_model.joblib")
    parser.add_argument("--subject", default="USER01")
    parser.add_argument("--output", default="outputs/RPS_NEW_DAY", type=Path)
    parser.add_argument("--stream-name", default="rps_raw")
    parser.add_argument("--activity-mode", choices=["legacy", "robust"], default="legacy",
                        help="仅控制最后打印的实时命令；robust需启动后保持静息8秒")
    args = parser.parse_args(argv)
    try:
        raw, base = args.input.expanduser().resolve(), args.base_model.expanduser().resolve()
        scripts = ["01_preprocess_openbci_rps_4ch.py", "03_calibrate_svm_openbci_rps_4ch.py",
                   "06_live_svm_openbci_rps_4ch.py", "07_live_svm_rps_label_4ch.py"]
        for path in [raw, base, *(ROOT / name for name in scripts)]:
            if not path.is_file():
                raise FileNotFoundError("找不到文件：%s" % path)
        output = args.output.expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output.mkdir(parents=True, exist_ok=False)
        (output / "pipeline_request.json").write_text(json.dumps({
            "input": str(raw), "base_model": str(base), "subject": args.subject,
            "activity_mode": args.activity_mode, "output": str(output)}, ensure_ascii=False, indent=2), encoding="utf-8")
        run_stage(scripts[0], ["--input", raw, "--subject", args.subject, "--split-mode", "none",
                              "--output", output / "CAL_PREPROCESS"])
        data = stage_directory(output / "CAL_PREPROCESS", "rps_windows.npz")
        run_stage(scripts[1], ["--base-model", base, "--today", data, "--output", output / "CAL_MODEL"])
        model_dir = stage_directory(output / "CAL_MODEL", "daily_model.joblib")
        model = model_dir / "daily_model.joblib"
        run_stage(scripts[2], ["--model", model, "--source-data", data, "--prepare-only", "--output", output / "LIVE_SETUP"])
        gate = stage_directory(output / "LIVE_SETUP", "replay_detector_config.json") / "replay_detector_config.json"
        live = [sys.executable, str(ROOT / scripts[3]), "--model", str(model), "--detector", str(gate),
                "--stream-name", args.stream_name, "--activity-mode", args.activity_mode,
                "--output", str(ROOT / "outputs" / ("LIVE_" + args.activity_mode.upper()))]
        if args.activity_mode == "robust":
            live += ["--rest-seconds", "8", "--min-active-channels", "2"]
        command = display_command(live)
        (output / "live_command.txt").write_text(command + "\n", encoding="utf-8")
        print("\n当天校准报告：\n%s\n\n后续实时识别命令：\n%s" % (model_dir / "OPEN_REPORT.html", command))
        print("\n先确认OpenBCI GUI的TimeSeriesRaw LSL正在发送，再执行上述命令。")
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print("[失败] %s；流程已停止，请保留已生成的输出目录。" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

