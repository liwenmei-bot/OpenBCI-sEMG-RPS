#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四通道石头/布二分类预处理；支持旧两类TXT与新二类TXT。

石头 Marker11 -> class0；布 Marker13 -> class1。静息由活动门控处理。
Python >= 3.9；依赖 numpy pandas scipy scikit-learn joblib，实时另需 pylsl。
二分类独立版本 1.0.0；不能加载旧二分类模型。
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import platform
import re
import sys
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
    import scipy
    from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos
except ImportError as exc:
    raise SystemExit("缺少依赖，请运行：python -m pip install numpy pandas scipy\n" + str(exc))


VERSION = "1.0.0"
SCHEMA = "openbci_rp_binary_4ch_preprocess_v1"
CHANNELS = ["EXG Channel %d" % i for i in range(4)]
FEATURES = ["MAV", "RMS", "WL", "ZC", "SSC"]
FEATURE_NAMES = ["ch%d_%s" % (i + 1, f) for i in range(4) for f in FEATURES]
FS_DEFAULT = 250.0
ACTION_CODES = (11, 13)
RAW_ACTION_CODES = (11, 12, 13)
RUN_STARTS = {101: 1, 102: 2, 103: 3, 104: 4}
RUN_ENDS = {111: 1, 112: 2, 113: 3, 114: 4}


def log(message: str) -> None:
    print(message, flush=True)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def write_csv(path: Path, records: List[Dict[str, Any]], columns=None) -> None:
    frame = pd.DataFrame(records)
    if columns is not None:
        for name in columns:
            if name not in frame:
                frame[name] = pd.Series(dtype="object")
        frame = frame[columns + [c for c in frame.columns if c not in columns]]
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10g")


def safe_id(text: str) -> str:
    value = re.sub(r"[^\w.\-]+", "_", text.strip(), flags=re.UNICODE).strip("._")
    if not value:
        raise ValueError("编号不能为空，也不能只含特殊符号。")
    return value


@dataclass(frozen=True)
class Source:
    path: Path
    member: Optional[str] = None

    @property
    def name(self) -> str:
        return Path(self.member).name if self.member else self.path.name

    @property
    def display(self) -> str:
        return str(self.path) + (" :: " + self.member if self.member else "")

    @contextmanager
    def open(self):
        if self.member is None:
            with self.path.open("rb") as stream:
                yield stream
        else:
            with zipfile.ZipFile(self.path) as archive:
                with archive.open(self.member) as stream:
                    yield stream

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        with self.open() as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()


def looks_like_openbci(head: bytes) -> bool:
    lines = head.decode("utf-8-sig", errors="replace").splitlines()
    header = next((line.strip() for line in lines if line.strip() and not line.lstrip().startswith("%")), "")
    return ("Sample Index" in header and "EXG Channel 0" in header
            and "Marker Channel" in header and "," in header)


def discover_sources(inputs: List[str]) -> List[Source]:
    sources = []
    visited = set()
    for entry in inputs:
        path = Path(entry).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError("输入不存在：%s" % path)
        paths = sorted(path.rglob("*")) if path.is_dir() else [path]
        for candidate in paths:
            if not candidate.is_file() or any(p.startswith(".") for p in candidate.parts):
                continue
            if candidate.suffix.lower() == ".zip":
                with zipfile.ZipFile(candidate) as archive:
                    for info in archive.infolist():
                        if (info.is_dir() or not info.filename.lower().endswith(".txt")
                                or "__MACOSX" in info.filename or Path(info.filename).name.startswith(".")):
                            continue
                        with archive.open(info) as stream:
                            if not looks_like_openbci(stream.read(8192)):
                                continue
                        key = (str(candidate), info.filename)
                        if key not in visited:
                            sources.append(Source(candidate, info.filename))
                            visited.add(key)
            elif candidate.suffix.lower() in (".txt", ".csv"):
                # 文件夹扫描跳过脚本以前输出的说明/汇总；显式文件则保留报错机会。
                if path.is_dir():
                    with candidate.open("rb") as stream:
                        head = stream.read(8192)
                    if not looks_like_openbci(head):
                        continue
                key = (str(candidate), None)
                if key not in visited:
                    sources.append(Source(candidate))
                    visited.add(key)
    if not sources:
        raise ValueError("没有找到可读取的 OpenBCI TXT/CSV；ZIP 中应包含原始 TXT。")
    return sources


def read_source(source: Source, args) -> Dict[str, Any]:
    with source.open() as stream:
        head = stream.read(65536).decode("utf-8-sig", errors="replace")
    sample_match = re.search(r"%\s*Sample\s+Rate\s*=\s*([0-9.]+)\s*Hz", head, re.I)
    header_fs = float(sample_match.group(1)) if sample_match else None
    fs = args.fs if args.fs is not None else (header_fs or FS_DEFAULT)
    if header_fs is not None and args.fs is not None and not math.isclose(header_fs, fs):
        raise ValueError("--fs 与文件头不一致：%s，文件头=%s，指定=%s" % (source.name, header_fs, fs))
    wanted = set(CHANNELS + ["Sample Index", "Timestamp", "Marker Channel", "Timestamp (Formatted)"])
    with source.open() as stream:
        frame = pd.read_csv(stream, comment="%", skipinitialspace=True,
                            encoding="utf-8-sig", low_memory=False,
                            usecols=lambda name: name.strip() in wanted)
    frame.columns = [c.strip() for c in frame.columns]
    required = CHANNELS + ["Sample Index", "Timestamp", "Marker Channel"]
    missing = [c for c in required if c not in frame]
    if missing:
        raise ValueError("%s 缺少列：%s。必须按列名读取，不能把 Sample Index 当成通道。" %
                         (source.name, ", ".join(missing)))
    if len(frame) == 0:
        raise ValueError("原始文件没有数据行：%s" % source.name)
    raw = frame[CHANNELS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    scales = {"uv": 1.0, "mv": 1000.0, "v": 1000000.0}
    raw *= scales[args.input_unit]
    index = pd.to_numeric(frame["Sample Index"], errors="coerce").to_numpy(dtype=np.float64)
    timestamp = pd.to_numeric(frame["Timestamp"], errors="coerce").to_numpy(dtype=np.float64)
    markers = pd.to_numeric(frame["Marker Channel"], errors="coerce").to_numpy(dtype=np.float64)
    clocks = frame["Timestamp (Formatted)"].astype(str).to_numpy() if "Timestamp (Formatted)" in frame else None
    match = re.search(r"(20\d{2}-\d{2}-\d{2})[_ T](\d{2})[-:](\d{2})[-:](\d{2})", source.name)
    first_date = None
    if clocks is not None and re.match(r"^20\d{2}-\d{2}-\d{2}", clocks[0]):
        first_date = clocks[0][:10]
    elif match:
        first_date = match.group(1)
    elif np.isfinite(timestamp[0]):
        tz = timezone(timedelta(hours=args.utc_offset_hours))
        first_date = datetime.fromtimestamp(float(timestamp[0]), tz).strftime("%Y-%m-%d")
    if first_date is None:
        raise ValueError("无法确定采集日期：%s；请补齐有效时间戳或原文件日期。" % source.name)
    board = re.search(r"%\s*Board\s*=\s*([^\r\n]+)", head)
    return {"raw": raw, "index": index, "timestamp": timestamp, "markers": markers,
            "clocks": clocks, "fs": float(fs), "header_fs": header_fs,
            "recording_date": first_date, "board": board.group(1).strip() if board else None}


def sample_count(seconds: float, fs: float, name: str) -> int:
    value = seconds * fs
    if not math.isclose(value, round(value), abs_tol=1e-7):
        raise ValueError("%s 换算后不是整数采样点：%s × %s；请调整参数。" % (name, seconds, fs))
    return int(round(value))


def make_config(args, fs: float) -> Dict[str, Any]:
    if not (0 < args.low_hz < args.high_hz < fs / 2):
        raise ValueError("带通须满足 0 < low < high < fs/2；当前 fs=%s。" % fs)
    if args.order < 1 or args.notch_q <= 0:
        raise ValueError("滤波阶数和陷波 Q 必须为正数。")
    if args.notch_hz < 0 or args.notch_hz >= fs / 2:
        raise ValueError("陷波频率应为 0（关闭）或小于 fs/2。")
    core_start = sample_count(args.core_start, fs, "核心段起点")
    core_stop = sample_count(args.core_stop, fs, "核心段终点")
    win = sample_count(args.window_ms / 1000, fs, "窗口长度")
    hop = sample_count(args.hop_ms / 1000, fs, "窗口步长")
    base_start = sample_count(args.baseline_start, fs, "基线起点")
    base_stop = sample_count(args.baseline_stop, fs, "基线终点")
    if not (0 <= core_start < core_stop and win >= 3 and hop > 0 and core_stop-core_start >= win):
        raise ValueError("核心段、窗长或步长设置不合法。")
    if not (base_start < base_stop <= 0):
        raise ValueError("基线偏移必须满足 baseline_start < baseline_stop <= 0。")
    if args.warmup_seconds < 0 or args.zc_threshold_uv < 0 or args.ssc_threshold_uv < 0:
        raise ValueError("预热长度与特征噪声门槛不能为负数。")
    if args.counter_modulus < 2 or args.host_pause_seconds <= 0:
        raise ValueError("计数器模数和主机长暂停阈值无效。")
    sections = []
    if args.notch_hz > 0:
        b, a = iirnotch(args.notch_hz, args.notch_q, fs=fs)
        sections.append(tf2sos(b, a))
    sections.append(butter(args.order, [args.low_hz, args.high_hz], btype="bandpass", fs=fs, output="sos"))
    sos = np.vstack(sections)
    return {
        "schema": SCHEMA, "script_version": VERSION, "fs": fs, "n_channels": 4,
        "channel_columns": CHANNELS, "physical_channels": ["CH1", "CH2", "CH3", "CH4"],
        "ignored_exg_columns": ["EXG Channel %d" % i for i in range(4, 8)],
        "input_unit": args.input_unit, "processing_unit": "uV", "filter_mode": "causal_sos",
        "low_hz": args.low_hz, "high_hz": args.high_hz, "butter_prototype_order": args.order,
        "notch_hz": args.notch_hz, "notch_q": args.notch_q, "sos": sos.tolist(),
        "filter_state_initialization": "sosfilt_zi_times_first_current_sample_per_channel",
        "warmup_samples": int(math.ceil(args.warmup_seconds*fs)),
        "core_start_samples": core_start, "core_stop_samples": core_stop,
        "core_length_samples": core_stop-core_start,
        "baseline_start_samples": base_start, "baseline_stop_samples": base_stop,
        "window_samples": win, "hop_samples": hop,
        "feature_names": FEATURE_NAMES, "feature_order": "channel_major_MAV_RMS_WL_ZC_SSC",
        "wl_normalized": False, "zc_threshold_uv": args.zc_threshold_uv,
        "ssc_threshold_uv": args.ssc_threshold_uv,
        "zc_definition": "different_sign_and_abs_adjacent_difference_ge_threshold",
        "ssc_definition": "local_extremum_and_max_adjacent_difference_ge_threshold",
        "feature_threshold_policy": "fixed_cli_values_no_fit_on_development_or_test",
        "standardization": "not_fitted_here_fit_only_in_each_training_fold_later",
        "qc": {"flat_p2p_uv": args.flat_p2p_uv, "warn_filtered_p2p_uv": args.warn_p2p_uv,
               "saturation_abs_uv": args.saturation_uv, "corr_warning": args.corr_warning,
               "saturation_enabled": args.saturation_uv is not None,
               "policy": "all_trials_and_windows_retained_recommendation_mask_only"},
        "counter_modulus": args.counter_modulus, "host_pause_seconds": args.host_pause_seconds,
        "marker_zero_meaning": "no_event_pulse_not_rest_label",
        "action_codes": list(ACTION_CODES), "end_code": 20,
        "run_start_codes": RUN_STARTS, "run_end_codes": RUN_ENDS,
    }


class CausalEMGFilter:
    """实时可复用的四路有状态单向滤波器。新 Session/缺口后显式 reset。"""

    def __init__(self, sos, n_channels: int = 4):
        self.sos = np.asarray(sos, dtype=np.float64)
        self.n_channels = int(n_channels)
        self.state = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(config["sos"], config["n_channels"])

    def reset(self) -> None:
        self.state = None

    def process(self, samples):
        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.n_channels:
            raise ValueError("滤波输入必须为 (采样点数, %d)。" % self.n_channels)
        if not np.isfinite(values).all():
            raise ValueError("滤波输入含非有限值；应切断该段，在下一有效段 reset 并预热。")
        if not len(values):
            return values.copy()
        if self.state is None:
            self.state = sosfilt_zi(self.sos)[:, :, None] * values[0][None, None, :]
        result, self.state = sosfilt(self.sos, values, axis=0, zi=self.state)
        return result


def continuity(index, timestamps, config) -> Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray]:
    n = len(index)
    breaks = np.zeros(n, dtype=bool)
    breaks[0] = True
    increments = np.ones(n, dtype=np.float64)
    issues = []
    modulus = config["counter_modulus"]
    for i in range(1, n):
        a, b = index[i-1], index[i]
        valid = (np.isfinite(a) and np.isfinite(b) and a == round(a) and b == round(b)
                 and 0 <= a < modulus and 0 <= b < modulus)
        dt = timestamps[i]-timestamps[i-1]
        reason = None
        inferred = None
        step = None
        if not valid:
            reason = "invalid_sample_index"
        else:
            step = int((b-a) % modulus)
            if step == 0:
                reason = "duplicate_or_full_wrap_ambiguous"
            elif 1 < step <= modulus//2:
                reason = "counter_jump_possible_missing_samples"
                inferred = step-1
                increments[i] = step
            elif step > modulus//2:
                reason = "counter_reset_or_out_of_order_ambiguous"
        # 批量接收导致的毫秒级抖动不视为丢包；特别长的暂停仅保守分段。
        if reason is None and np.isfinite(dt) and dt > config["host_pause_seconds"]:
            reason = "long_host_pause_uncertain_sampling_continuity"
        if reason is not None:
            breaks[i] = True
            issues.append({"after_row": i, "before_row": i-1, "reason": reason,
                           "counter_before": a, "counter_after": b, "modulo_step": step,
                           "inferred_missing_samples": inferred, "host_dt_seconds": dt})
    reconstructed = (np.cumsum(increments)-increments[0])/config["fs"]
    return breaks, issues, reconstructed


def causal_filter_continuous(raw, breaks, config):
    finite = np.isfinite(raw).all(axis=1)
    changes = np.flatnonzero(finite[1:] != finite[:-1])+1
    boundaries = np.unique(np.r_[0, len(raw), changes, np.flatnonzero(breaks)])
    filtered = np.full(raw.shape, np.nan, dtype=np.float64)
    ready = np.zeros(len(raw), dtype=bool)
    processor = CausalEMGFilter.from_config(config)
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        lo, hi = int(lo), int(hi)
        if lo == hi or not finite[lo]:
            continue
        processor.reset()
        filtered[lo:hi] = processor.process(raw[lo:hi])
        start = min(hi, lo+config["warmup_samples"])
        ready[start:hi] = True
    return filtered, ready


def marker_events(markers) -> Tuple[List[Dict[str, Any]], List[str]]:
    events = []
    warnings = []
    i = 0
    while i < len(markers):
        value = markers[i]
        if not np.isfinite(value):
            warnings.append("nonfinite_marker_at_row_%d" % i)
            i += 1
            continue
        if value == 0:
            i += 1
            continue
        stop = i+1
        while stop < len(markers) and markers[stop] == value:
            stop += 1
        if not math.isclose(float(value), round(float(value)), abs_tol=1e-6):
            warnings.append("noninteger_marker_at_row_%d" % i)
            events.append({"row": i, "last_row": stop-1, "pulse_samples": stop-i,
                           "code": None, "raw_code": float(value)})
        else:
            events.append({"row": i, "last_row": stop-1, "pulse_samples": stop-i,
                           "code": int(round(float(value))), "raw_code": float(value)})
        i = stop
    return events, warnings


def parse_protocol(events, n_rows: int):
    trials, runs, warnings = [], [], []
    active_run = None
    active_trial = None
    run_instance = 0
    trial_counts = Counter()

    def close_trial(end_row, reason=None):
        nonlocal active_trial
        if active_trial is None:
            return
        if reason is not None:
            active_trial["protocol_reasons"].append(reason)
            active_trial["end_row"] = -1
        else:
            active_trial["end_row"] = int(end_row)
        active_trial["available_stop_row"] = int(end_row)
        trials.append(active_trial)
        active_trial = None

    def close_run(end_row, reason=None):
        nonlocal active_run
        if active_run is not None:
            active_run["end_row"] = int(end_row) if reason is None else -1
            active_run["available_stop_row"] = int(end_row)
            if reason:
                active_run["protocol_reasons"].append(reason)
            runs.append(active_run)
            active_run = None

    for event in events:
        row, code = event["row"], event["code"]
        if code in RUN_STARTS:
            close_trial(row, "missing_trial_end_before_run_start")
            close_run(row, "missing_run_end_before_new_run")
            run_instance += 1
            active_run = {"run_id": RUN_STARTS[code], "run_instance": run_instance,
                          "start_row": row, "start_marker": code, "end_row": -1,
                          "protocol_reasons": []}
        elif code in RAW_ACTION_CODES:
            close_trial(row, "missing_trial_end_before_new_action")
            instance = active_run["run_instance"] if active_run else 0
            trial_counts[instance] += 1
            active_trial = {"trial_in_session": len(trials)+1,
                            "run_id": active_run["run_id"] if active_run else 0,
                            "run_instance": instance, "trial_in_run": trial_counts[instance],
                            "start_row": row, "end_row": -1, "marker_code": code,
                            "protocol_reasons": [] if active_run else ["missing_run_start"]}
        elif code == 20:
            if active_trial is None:
                warnings.append("orphan_end_marker_at_row_%d" % row)
            else:
                close_trial(row)
        elif code in RUN_ENDS:
            close_trial(row, "missing_trial_end_before_run_end")
            if active_run is None:
                warnings.append("orphan_run_end_at_row_%d" % row)
            elif active_run["run_id"] != RUN_ENDS[code]:
                active_run["protocol_reasons"].append("run_end_code_mismatch")
                close_run(row)
            else:
                close_run(row)
        else:
            warnings.append("unknown_marker_%s_at_row_%d" % (event["raw_code"], row))
            if active_trial is not None:
                active_trial["protocol_reasons"].append("unknown_marker_inside_action")
    close_trial(n_rows, "missing_trial_end_at_file_end")
    close_run(n_rows, "missing_run_end_at_file_end")
    run_by_instance = {r["run_instance"]: r for r in runs}
    for trial in trials:
        run = run_by_instance.get(trial["run_instance"])
        if run and run["protocol_reasons"]:
            trial["protocol_reasons"].extend(run["protocol_reasons"])
        if trial["end_row"] <= trial["start_row"]:
            trial["protocol_reasons"].append("unverified_action_end")
        trial["protocol_reasons"] = sorted(set(trial["protocol_reasons"]))
    for run in runs:
        subset = [t for t in trials if t["run_instance"] == run["run_instance"]]
        run["trial_count"] = len(subset)
        run["class_counts"] = dict(Counter(t["marker_code"] for t in subset))
    return trials, runs, warnings


def extract_features(window, zc_threshold_uv: float = 2.0, ssc_threshold_uv: float = 2.0):
    """输入 (采样点,4) 的已滤波窗口；返回按通道排列的 20 维特征。"""
    x = np.asarray(window, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 4 or len(x) < 3:
        raise ValueError("特征输入必须为 (至少3个采样点,4)。")
    if not np.isfinite(x).all():
        return np.full(20, np.nan, dtype=np.float64)
    diff = np.diff(x, axis=0)
    zc = ((np.signbit(x[:-1]) != np.signbit(x[1:])) & (np.abs(diff) >= zc_threshold_uv)).sum(axis=0)
    left = x[1:-1]-x[:-2]
    right = x[1:-1]-x[2:]
    ssc = ((left*right > 0) & (np.maximum(np.abs(left), np.abs(right)) >= ssc_threshold_uv)).sum(axis=0)
    matrix = np.stack([np.mean(np.abs(x), axis=0), np.sqrt(np.mean(x*x, axis=0)),
                       np.abs(diff).sum(axis=0), zc, ssc], axis=1)
    return matrix.reshape(20)


def padded_slice(values, start: int, stop: int):
    result = np.full((stop-start, 4), np.nan, dtype=np.float32)
    lo, hi = max(0, start), min(len(values), stop)
    if hi > lo:
        result[lo-start:hi-start] = values[lo:hi]
    return result


def inspect_window(raw, filtered, ready, breaks, start, stop, protocol_ok, config):
    reasons = []
    severity = 0
    length = stop-start
    inside = 0 <= start < stop <= len(raw)
    if not protocol_ok:
        reasons.append("protocol_review")
        severity = 2
    if not inside or len(filtered[start:stop]) != length:
        reasons.append("out_of_bounds")
        return "REVIEW", reasons, False, float("nan")
    xr, xf = raw[start:stop], filtered[start:stop]
    if not np.isfinite(xr).all() or not np.isfinite(xf).all():
        reasons.append("nonfinite_signal")
        severity = 2
    if breaks[start+1:stop].any():
        reasons.append("crosses_continuity_break")
        severity = 2
    if not ready[start:stop].all():
        reasons.append("filter_warmup_or_invalid")
        severity = 2
    corr = float("nan")
    if np.isfinite(xr).all():
        flats = np.flatnonzero(np.ptp(xr, axis=0) <= config["qc"]["flat_p2p_uv"])
        if len(flats):
            reasons.append("flat_" + "_".join("ch%d" % (i+1) for i in flats))
            severity = 2
        sat = config["qc"]["saturation_abs_uv"]
        if sat is not None and np.max(np.abs(xr)) >= sat:
            reasons.append("configured_saturation_threshold")
            severity = 2
    if np.isfinite(xf).all():
        warning = config["qc"]["warn_filtered_p2p_uv"]
        if warning > 0 and np.max(np.ptp(xf, axis=0)) > warning:
            reasons.append("large_filtered_p2p_warning_only")
            severity = max(severity, 1)
        if np.all(np.std(xf, axis=0) > 1e-12):
            corr = float(np.max(np.abs(np.corrcoef(xf, rowvar=False)[np.triu_indices(4, 1)])))
            if config["qc"]["corr_warning"] > 0 and corr > config["qc"]["corr_warning"]:
                reasons.append("high_channel_correlation_warning_only")
                severity = max(severity, 1)
    return ("PASS", "WARNING", "REVIEW")[severity], sorted(set(reasons)), severity < 2, corr


def trial_date(data, row: int, args) -> str:
    clocks = data["clocks"]
    if clocks is not None and re.match(r"^20\d{2}-\d{2}-\d{2}", clocks[row]):
        return str(clocks[row][:10])
    timestamp = data["timestamp"][row]
    if np.isfinite(timestamp):
        tz = timezone(timedelta(hours=args.utc_offset_hours))
        return datetime.fromtimestamp(float(timestamp), tz).strftime("%Y-%m-%d")
    return data["recording_date"]


def process_session(source, digest, data, config, args, output, class_names):
    fs = config["fs"]
    stem = safe_id(Path(source.name).stem)
    session_id = stem + "_" + digest[:10]
    session_dir = output/"sessions"/session_id
    session_dir.mkdir(parents=True, exist_ok=False)
    raw = data["raw"]
    breaks, issues, reconstructed = continuity(data["index"], data["timestamp"], config)
    filtered, ready = causal_filter_continuous(raw, breaks, config)
    events, event_warnings = marker_events(data["markers"])
    trials, runs, protocol_warnings = parse_protocol(events, len(raw))
    warnings = event_warnings + protocol_warnings
    if data["header_fs"] is None:
        warnings.append("missing_sample_rate_header_fs_from_cli_or_default_%s" % fs)
    if len(runs) != 4:
        warnings.append("expected_4_runs_observed_%d" % len(runs))
    for run in runs:
        if sum(run["class_counts"].get(c, 0) for c in ACTION_CODES) != 12 or any(run["class_counts"].get(c, 0) != 6 for c in ACTION_CODES):
            warnings.append("run_%d_counts_differ_from_12_target_trials_6_each_class" % run["run_id"])
    excluded = [dict(t, exclusion_reason="unsupported_scissors_marker12_not_rest") for t in trials if t["marker_code"] == 12]
    write_csv(session_dir/"excluded_actions.csv", excluded, ["run_id", "start_row", "end_row", "available_stop_row", "marker_code", "exclusion_reason"])
    excluded_mask = np.zeros(len(raw), dtype=bool)
    for item in excluded:
        excluded_mask[max(0, item["start_row"]):min(len(raw), item["available_stop_row"])] = True
    raw_trial_count = len(trials)
    trials = [t for t in trials if t["marker_code"] in ACTION_CODES]
    key = {"subject_id": args.subject, "session_id": session_id, "source_sha256": digest}
    windows, window_features, window_signals = [], [], []
    core_segments, baselines, full_segments, full_lengths = [], [], [], []
    rest_candidates = []
    gesture_labels = np.full(len(raw), -1, dtype=np.int16)
    gesture_labels[excluded_mask] = -2  # unsupported activity; never a rest class
    trial_index = np.full(len(raw), -1, dtype=np.int32)
    candidate_rest = np.zeros(len(raw), dtype=bool)
    n_core = config["core_length_samples"]
    width, hop = config["window_samples"], config["hop_samples"]
    for trial in trials:
        start, end = trial["start_row"], trial["end_row"]
        code = trial["marker_code"]
        label = ACTION_CODES.index(code)
        date = trial_date(data, start, args)
        uid = "%s::%s::%s::R%dI%d::T%03d" % (args.subject, date, session_id,
                  trial["run_id"], trial["run_instance"], trial["trial_in_run"])
        trial.update(key)
        trial.update({"trial_id": uid, "collection_date": date, "class_id": label,
                      "class_name": class_names[label], "source_file": source.display,
                      "protocol_ok": not trial["protocol_reasons"],
                      "duration_samples_s": (end-start)/fs if end >= start else float("nan"),
                      "duration_host_s": float(data["timestamp"][end]-data["timestamp"][start])
                          if 0 <= start < end < len(raw) else float("nan")})
        core_lo = start+config["core_start_samples"]
        core_hi = start+config["core_stop_samples"]
        core_inside_action = end >= core_hi and core_lo >= start
        protocol_ok = trial["protocol_ok"] and core_inside_action
        trial.update({"core_start_row": core_lo, "core_stop_row": core_hi,
                      "core_inside_verified_action": bool(core_inside_action),
                      "core_protocol_verified": bool(protocol_ok),
                      "gap_inside_action": bool(end > start and breaks[start+1:end].any())})
        core_segments.append(padded_slice(filtered, core_lo, core_hi).T)
        base_lo = start+config["baseline_start_samples"]
        base_hi = start+config["baseline_stop_samples"]
        baselines.append(padded_slice(filtered, base_lo, base_hi).T)
        base_qc, base_reasons, base_usable, _ = inspect_window(
            raw, filtered, ready, breaks, base_lo, base_hi, True, config)
        if base_hi > 0 and base_lo < len(raw) and gesture_labels[max(0, base_lo):min(len(raw), base_hi)].max(initial=-1) >= 0:
            base_reasons.append("overlaps_previous_verified_action")
            base_qc, base_usable = "REVIEW", False
        rest_candidates.append(dict(key, trial_id=uid, collection_date=date,
            start_row=base_lo, stop_row=base_hi, qc_status=base_qc,
            qc_reasons=";".join(base_reasons), numerical_usable=base_usable,
            verified_rest=False, meaning="candidate_only_verify_relaxation_before_rest_training"))
        if excluded_mask[max(0, base_lo):min(len(raw), base_hi)].any():
            base_reasons.append("overlaps_excluded_scissors")
            base_qc, base_usable = "REVIEW", False
            rest_candidates[-1].update(qc_status=base_qc, qc_reasons=";".join(base_reasons), numerical_usable=False)
        if base_usable:
            candidate_rest[max(0, base_lo):min(len(raw), base_hi)] = True
        trial["baseline_qc"] = base_qc
        trial["baseline_is_verified_rest"] = False
        # 原始动作完整保留，长度由真实起止 Marker 决定，未核实终点则保留可用范围并标记。
        full_stop = min(len(raw), max(start, trial["available_stop_row"]))
        full_segments.append(filtered[start:full_stop].astype(np.float32))
        full_lengths.append(full_stop-start)
        trial["full_numerical_usable"] = bool(trial["protocol_ok"] and full_stop > start
            and np.isfinite(filtered[start:full_stop]).all() and ready[start:full_stop].all()
            and not breaks[start+1:full_stop].any())
        trial["baseline_numerical_usable"] = bool(base_usable)
        if trial["protocol_ok"]:
            gesture_labels[start:end] = label
            trial_index[start:end] = trial["trial_in_session"]
        statuses, reasons_all = [], set(trial["protocol_reasons"])
        eligible = 0
        for number, offset in enumerate(range(0, n_core-width+1, hop), 1):
            lo, hi = core_lo+offset, core_lo+offset+width
            status, reasons, usable, corr = inspect_window(
                raw, filtered, ready, breaks, lo, hi, protocol_ok, config)
            signal = padded_slice(filtered, lo, hi)
            features = extract_features(signal, config["zc_threshold_uv"], config["ssc_threshold_uv"])
            record = dict(key, trial_id=uid, collection_date=date, run_id=trial["run_id"],
                run_instance=trial["run_instance"], trial_in_session=trial["trial_in_session"],
                trial_in_run=trial["trial_in_run"], window_in_trial=number,
                marker_code=code, class_id=label, class_name=class_names[label],
                start_row=lo, stop_row=hi, start_row_time_s=lo/fs, stop_row_time_s=hi/fs,
                qc_status=status, qc_reasons=";".join(reasons), usable_for_training=bool(usable),
                max_abs_channel_correlation=corr)
            windows.append(record)
            window_features.append(features.astype(np.float32))
            window_signals.append(signal.T)
            statuses.append(status)
            reasons_all.update(reasons)
            eligible += int(usable)
        ranks = {"PASS": 0, "WARNING": 1, "REVIEW": 2}
        trial["qc_status"] = max(statuses, key=lambda s: ranks[s]) if statuses else "REVIEW"
        if not protocol_ok:
            trial["qc_status"] = "REVIEW"
            reasons_all.add("core_or_protocol_not_verified")
        if trial["gap_inside_action"]:
            trial["qc_status"] = "REVIEW"
            reasons_all.add("continuity_break_inside_full_action")
        trial["qc_reasons"] = ";".join(sorted(reasons_all))
        trial["protocol_reasons"] = ";".join(trial["protocol_reasons"])
        trial["n_windows"] = len(statuses)
        trial["n_usable_windows"] = eligible
        trial["has_usable_windows"] = eligible > 0
        trial["core_numerical_usable"] = bool(protocol_ok and len(statuses) > 0 and eligible == len(statuses))
    for item in events:
        item.update(key)
        item["host_timestamp"] = float(data["timestamp"][item["row"]])
        item["row_time_s"] = item["row"]/fs
    for item in issues:
        item.update(key)
        item["inside_trial_ids"] = ";".join(t["trial_id"] for t in trials
            if t["start_row"] < item["after_row"] < t["end_row"])
    for item in runs:
        item.update(key)
        item["protocol_reasons"] = ";".join(item["protocol_reasons"])
        item["class_counts"] = json.dumps(item["class_counts"], sort_keys=True)
    # Marker=0 与未标注区间没有被填成静息类别；-1 始终表示未知/未标注。
    np.savez_compressed(session_dir/"continuous_4ch.npz",
        X_raw=raw.astype(np.float32), X_filtered=filtered.astype(np.float32),
        sample_index=data["index"], host_timestamp=data["timestamp"], marker_channel=data["markers"],
        row_time_s=np.arange(len(raw), dtype=np.float64)/fs,
        counter_time_s=reconstructed, break_before=breaks, filter_ready_mask=ready,
        gesture_class=gesture_labels, trial_in_session=trial_index,
        candidate_rest_mask=candidate_rest, excluded_activity_mask=excluded_mask, channel_columns=np.array(CHANNELS),
        schema=np.array(SCHEMA), fs=np.array(fs), source_sha256=np.array(digest))
    dt = np.diff(data["timestamp"])
    finite_dt = dt[np.isfinite(dt)]
    description = dict(key, source_file=source.display, source_name=source.name,
        recording_date=data["recording_date"], fs=fs, header_fs=data["header_fs"],
        board=data["board"], n_rows=len(raw), n_channels_used=4,
        row_duration_seconds=len(raw)/fs,
        first_host_timestamp=float(data["timestamp"][0]),
        host_timestamp_span_seconds=float(data["timestamp"][-1]-data["timestamp"][0]),
        host_dt_median_seconds=float(np.median(finite_dt)) if len(finite_dt) else None,
        host_dt_max_seconds=float(np.max(finite_dt)) if len(finite_dt) else None,
        host_timestamp_negative_count=int(np.sum(finite_dt < 0)),
        nonfinite_signal_rows=int(np.sum(~np.isfinite(raw).all(axis=1))),
        continuity_issue_count=len(issues), continuity_issues_in_actions=sum(bool(i["inside_trial_ids"]) for i in issues),
        raw_trial_count=raw_trial_count, excluded_scissors_trials=len(excluded),
        trial_count=len(trials), verified_complete_trial_count=sum(t["protocol_ok"] for t in trials),
        run_count=len(runs), class_counts=dict(Counter(t["marker_code"] for t in trials)),
        window_count=len(windows), window_qc_counts=dict(Counter(w["qc_status"] for w in windows)),
        usable_window_count=sum(w["usable_for_training"] for w in windows), warnings=warnings,
        continuous_file=str((session_dir/"continuous_4ch.npz").relative_to(output)))
    write_json(session_dir/"session_audit.json", description)
    return {"description": description, "trials": trials, "runs": runs, "events": events,
            "issues": issues, "windows": windows, "features": window_features,
            "window_signals": window_signals, "cores": core_segments, "baselines": baselines,
            "fulls": full_segments, "full_lengths": full_lengths, "rests": rest_candidates}


def assign_splits(trials, windows, descriptions, args):
    dates = sorted(set(t["collection_date"] for t in trials))
    sessions = list(dict.fromkeys(t["session_id"] for t in trials))
    mode = args.split_mode
    if mode == "auto":
        mode = "by-date" if len(dates) >= 2 else ("by-session" if len(sessions) >= 2 else "within-session")
    notes = []
    test_date, test_session = None, None
    if mode == "by-date":
        if len(dates) < 2:
            raise ValueError("by-date 至少需要两个实际采集日期；当前单文件请用 auto。")
        test_date = args.test_date or dates[-1]
        if test_date not in dates:
            raise ValueError("指定测试日期不在数据中：%s" % test_date)
    elif mode == "by-session":
        if len(sessions) < 2:
            raise ValueError("by-session 至少需要两个独立原始 Session。")
        info = {d["session_id"]: d for d in descriptions}
        def session_order(session):
            first = info[session].get("first_host_timestamp", float("nan"))
            return (info[session]["recording_date"], first if np.isfinite(first) else float("-inf"), session)
        test_session = max(sessions, key=session_order)
    lookup = {}
    for trial in trials:
        role = "unassigned"
        if trial["run_id"] > 0:
            if mode == "within-session":
                role = "test" if trial["run_id"] == args.test_run else "development"
            elif mode == "by-date":
                role = "test" if trial["collection_date"] == test_date else "development"
            elif mode == "by-session":
                role = "test" if trial["session_id"] == test_session else "development"
            elif mode == "none":
                role = "unassigned"
        if mode == "by-date":
            inner = args.subject + "::" + trial["collection_date"]
        elif mode == "by-session":
            inner = args.subject + "::" + trial["session_id"]
        else:
            inner = "%s::%s::R%dI%d" % (args.subject, trial["session_id"], trial["run_id"], trial["run_instance"])
        trial["split_role"] = role
        trial["inner_cv_group"] = inner
        lookup[trial["trial_id"]] = (role, inner)
    for window in windows:
        window["split_role"], window["inner_cv_group"] = lookup[window["trial_id"]]
    counts = Counter(t["split_role"] for t in trials)
    if mode != "none" and (counts["development"] == 0 or counts["test"] == 0):
        notes.append("没有同时得到开发与测试 Trial；检查 Run 边界及 --test-run，当前划分不能用于独立评估。")
    if mode == "within-session":
        notes.append("这是同日留 Run 评估，不能据此报告跨日或跨人准确率。")
    return {"mode": mode, "test_run": args.test_run if mode == "within-session" else None,
            "test_date": test_date, "test_session": test_session,
            "trial_counts": dict(counts), "notes": notes,
            "policy": "all_windows_of_a_trial_share_one_role_no_random_window_split"}


def text_array(records, key):
    return np.asarray([str(r.get(key, "")) for r in records], dtype=str)


def numeric_array(records, key, dtype=np.int64):
    return np.asarray([r[key] for r in records], dtype=dtype)


def save_all(output, parts, config, args, duplicate_sources):
    merge = lambda key: [item for part in parts for item in part[key]]
    trials, windows = merge("trials"), merge("windows")
    descriptions = [part["description"] for part in parts]
    split = assign_splits(trials, windows, descriptions, args)
    trial_columns = ["subject_id", "collection_date", "session_id", "run_id", "run_instance",
                     "trial_in_session", "trial_in_run", "trial_id", "marker_code", "class_id", "class_name",
                     "start_row", "end_row", "core_start_row", "core_stop_row", "split_role", "inner_cv_group",
                     "qc_status", "qc_reasons", "n_windows", "n_usable_windows"]
    write_csv(output/"trial_manifest.csv", trials, trial_columns)
    write_csv(output/"split_manifest.csv", trials, trial_columns)
    feature_list = merge("features")
    X = np.stack(feature_list).astype(np.float32) if feature_list else np.empty((0, 20), np.float32)
    features_frame = pd.DataFrame(windows)
    for i, name in enumerate(FEATURE_NAMES):
        features_frame[name] = X[:, i]
    if not windows:
        features_frame = pd.DataFrame(columns=["trial_id", "split_role", "qc_status", "usable_for_training"] + FEATURE_NAMES)
    features_frame.to_csv(output/"window_features.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    write_csv(output/"marker_events.csv", merge("events"), ["subject_id", "session_id", "row", "code", "pulse_samples"])
    write_csv(output/"continuity_issues.csv", merge("issues"), ["subject_id", "session_id", "after_row", "reason", "inferred_missing_samples", "inside_trial_ids"])
    write_csv(output/"run_manifest.csv", merge("runs"), ["subject_id", "session_id", "run_id", "run_instance", "start_row", "end_row", "trial_count", "class_counts"])
    write_csv(output/"rest_candidates.csv", merge("rests"), ["subject_id", "session_id", "trial_id", "start_row", "stop_row", "qc_status", "verified_rest"])
    signals = merge("window_signals")
    signal_windows = np.stack(signals).astype(np.float32) if signals else np.empty((0, 4, config["window_samples"]), np.float32)
    np.savez_compressed(output/"rps_windows.npz", X=X, X_features=X, X_signal=signal_windows,
        y=numeric_array(windows, "class_id"), marker_code=numeric_array(windows, "marker_code"),
        run_id=numeric_array(windows, "run_id"), trial_id=text_array(windows, "trial_id"),
        collection_date=text_array(windows, "collection_date"), session_id=text_array(windows, "session_id"),
        subject_id=text_array(windows, "subject_id"), split_role=text_array(windows, "split_role"),
        inner_cv_group=text_array(windows, "inner_cv_group"), qc_status=text_array(windows, "qc_status"),
        qc_reasons=text_array(windows, "qc_reasons"), usable_mask=numeric_array(windows, "usable_for_training", bool),
        start_row=numeric_array(windows, "start_row"), stop_row=numeric_array(windows, "stop_row"),
        feature_names=np.array(FEATURE_NAMES), class_names=np.array(args.class_names or ["石头", "布"]),
        schema=np.array(SCHEMA), fs=np.array(config["fs"]))
    n_core = config["core_length_samples"]
    n_base = config["baseline_stop_samples"]-config["baseline_start_samples"]
    cores, bases = merge("cores"), merge("baselines")
    full_lengths = np.asarray(merge("full_lengths"), dtype=np.int64)
    offsets = np.r_[0, np.cumsum(full_lengths)]
    fulls = merge("fulls")
    np.savez_compressed(output/"rps_segments.npz",
        X=np.stack(cores).astype(np.float32) if cores else np.empty((0, 4, n_core), np.float32),
        X_baseline=np.stack(bases).astype(np.float32) if bases else np.empty((0, 4, n_base), np.float32),
        X_full=np.concatenate(fulls, axis=0) if fulls else np.empty((0, 4), np.float32),
        full_lengths=full_lengths, full_offsets=offsets,
        y=numeric_array(trials, "class_id"), marker_code=numeric_array(trials, "marker_code"),
        run_id=numeric_array(trials, "run_id"), trial_id=text_array(trials, "trial_id"),
        subject_id=text_array(trials, "subject_id"), collection_date=text_array(trials, "collection_date"),
        session_id=text_array(trials, "session_id"), split_role=text_array(trials, "split_role"),
        inner_cv_group=text_array(trials, "inner_cv_group"), qc_status=text_array(trials, "qc_status"),
        core_protocol_verified=numeric_array(trials, "core_protocol_verified", bool),
        usable_mask=numeric_array(trials, "core_numerical_usable", bool),
        full_usable_mask=numeric_array(trials, "full_numerical_usable", bool),
        baseline_numerical_usable_mask=numeric_array(trials, "baseline_numerical_usable", bool),
        n_usable_windows=numeric_array(trials, "n_usable_windows"),
        schema=np.array(SCHEMA), fs=np.array(config["fs"]), channel_columns=np.array(CHANNELS))
    config["split"] = split
    config["command_line"] = sys.argv
    config["versions"] = {"python": platform.python_version(), "numpy": np.__version__,
                          "pandas": pd.__version__, "scipy": scipy.__version__}
    write_json(output/"preprocess_config.json", config)
    write_json(output/"label_mapping.json", {
        "marker_to_class_id": {str(c): i for i, c in enumerate(ACTION_CODES)},
        "class_id_to_name": {str(i): name for i, name in enumerate(args.class_names or ["石头", "布"])},
        "gesture_names_supplied_by_user": bool(args.class_names),
        "names_need_collection_config_verification": not bool(args.class_names),
        "marker_zero_is_rest": False, "excluded_marker_codes": [12], "class_id_to_marker": {"0": 11, "1": 13}, "run_boundary_markers_are_classes": False})
    summary = {"schema": SCHEMA, "script_version": VERSION, "subject_id": args.subject,
        "sources": descriptions, "duplicate_sources_skipped": duplicate_sources,
        "total_trials_retained": len(trials), "complete_protocol_trials": sum(t["protocol_ok"] for t in trials),
        "total_windows_retained": len(windows), "feature_dimension": 20,
        "class_counts_trials": dict(Counter(t["marker_code"] for t in trials)),
        "qc_trial_counts": dict(Counter(t["qc_status"] for t in trials)),
        "qc_window_counts": dict(Counter(w["qc_status"] for w in windows)),
        "usable_windows": sum(w["usable_for_training"] for w in windows),
        "split": split, "output_directory": str(output), "model_trained": False,
        "standard_scaler_fitted": False, "rest_labels_verified": False,
        "qc_thresholds_are_initial_fixed_candidates": True,
        "saturation_check_enabled": config["qc"]["saturation_enabled"]}
    write_json(output/"summary.json", summary)
    write_report(output, summary, trials, config)
    return summary


def html_table(headers, rows):
    escape = lambda x: html.escape(str(x))
    return "<table><thead><tr>" + "".join("<th>"+escape(x)+"</th>" for x in headers) + "</tr></thead><tbody>" + "".join(
        "<tr>"+"".join("<td>"+escape(x)+"</td>" for x in row)+"</tr>" for row in rows) + "</tbody></table>"


def write_report(output, summary, trials, config):
    escape = lambda x: html.escape(str(x))
    run_counts = Counter((t["session_id"], t["run_id"], t["marker_code"]) for t in trials)
    run_keys = sorted(set((t["session_id"], t["run_id"]) for t in trials))
    run_rows = [[s, r, run_counts[(s, r, 11)], run_counts[(s, r, 12)], run_counts[(s, r, 13)]] for s, r in run_keys]
    steps = [
        "读取 TXT 或 ZIP，按名称只选择 EXG Channel 0、1、2、3。后四个 EXG 占位列不参与任何质检或训练。",
        "检查 Sample Index 连续性及接收时间戳。批量到达的时间戳抖动不直接判为丢包；断点后重新初始化并预热。",
        "解析四个 Run 边界，11/13 配对后续 20；Marker=0 不作为静息标签。",
        "连续执行单向 SOS 滤波，按配置保留状态；不在每个窗口上重置，不使用双向滤波或全数据标准化。",
        "完整动作区间全部保存；默认用起点后 0.5 至 3.5 秒生成 200 ms 窗口，步长 100 ms。",
        "每路提取 MAV、RMS、WL、ZC、SSC，按通道顺序形成 20 维特征。噪声门槛是固定配置，未在测试集估计。",
        "保留全部目标 Trial/窗口并标记 PASS、WARNING、REVIEW；usable_mask 供后续训练选择，未删除原记录。",
        "按完整 Trial 和 Run/日期分组。StandardScaler 与模型留到训练阶段，在每个训练折中拟合。"
    ]
    blocks = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>OpenBCI 四通道预处理报告</title>",
        "<style>body{font-family:system-ui,'Microsoft YaHei',sans-serif;max-width:1120px;margin:36px auto;padding:0 22px;color:#18212b;line-height:1.65}h1,h2{color:#111}table{border-collapse:collapse;width:100%;margin:16px 0 24px;font-size:14px}td,th{border:1px solid #d5dce4;padding:9px 11px;text-align:left;overflow-wrap:anywhere}th{background:#e6edf5}tr:nth-child(even){background:#f8fafc}li{margin:7px 0}code{background:#edf1f5;padding:2px 5px}small{color:#536170}</style>",
        "<h1>OpenBCI 四通道肌电预处理报告</h1>",
        "<p>受试者：<b>%s</b>；保留 <b>%d</b> 个 Trial、<b>%d</b> 个窗口，每窗 <b>20</b> 维特征。</p>" %
            (escape(summary["subject_id"]), summary["total_trials_retained"], summary["total_windows_retained"]),
        "<p>此步骤未训练模型，也未拟合标准化参数。默认映射：石头11、布13。剪刀12单独审计，不用于分类训练或充当静息。</p>",
        "<h2>预处理流程</h2><ol>" + "".join("<li>"+escape(s)+"</li>" for s in steps)+"</ol>",
        "<h2>各 Run 的类别数量</h2>", html_table(["Session", "Run", "Marker 11", "Marker 12", "Marker 13"], run_rows),
        "<h2>质量标记与划分</h2>", html_table(["项目", "结果"], [
            ["Trial 质量标记", summary["qc_trial_counts"]], ["窗口质量标记", summary["qc_window_counts"]],
            ["建议可用于训练的窗口", summary["usable_windows"]], ["划分方式", summary["split"]["mode"]],
            ["完整 Trial 划分", summary["split"]["trial_counts"]],
            ["配置饱和阈值检查", "已启用" if summary["saturation_check_enabled"] else "未启用；须按实际单位与增益配置 --saturation-uv"]]),
        "<ul>"+"".join("<li>"+escape(x)+"</li>" for x in summary["split"]["notes"])+"</ul>",
        "<p>高相关和大幅值仅作 WARNING，不自动剔除。缺失数值、跨断点、预热、明显平直或协议异常为 REVIEW；全部记录仍保存。阈值为工程起点，需在开发数据核查。</p>",
        "<h2>连续性检查</h2>", html_table(["原文件", "行数", "断点或暂停", "位于正式动作内", "协议检查提示"],
            [[s["source_name"], s["n_rows"], s["continuity_issue_count"], s["continuity_issues_in_actions"], "; ".join(s["warnings"]) or "无"] for s in summary["sources"]]),
        "<h2>主要输出</h2>", html_table(["文件", "用途"], [
            ["window_features.csv / rps_windows.npz", "SVM 特征、短窗信号、标签、分组与质量掩码"],
            ["rps_segments.npz", "三秒动作段、候选基线、完整动作紧凑数组与 full_offsets"],
            ["sessions/各Session/continuous_4ch.npz", "全部四路连续原始与滤波信号，供实时回放"],
            ["trial_manifest.csv / split_manifest.csv", "完整 Trial 清单和按组划分"],
            ["rest_candidates.csv", "动作前基线候选；尚未确认放松，不能直接充当静息真值"],
            ["preprocess_config.json / label_mapping.json", "滤波系数、特征顺序、固定门槛、类别代码与软件版本"]]),
        "<h2>后续训练规则</h2><p>只用 development 选择算法与超参数，同一 Trial 所有重叠窗口保持同组。test 不参与标准化、阈值或模型选型。完成留出评估后可另外用全部 Trial 训练部署版；该部署版的准确率继续用新一天独立数据验证。</p>",
        "<small>schema: "+escape(SCHEMA)+"；脚本版本 "+escape(VERSION)+"</small></html>"]
    (output/"OPEN_REPORT.html").write_text("\n".join(blocks), encoding="utf-8")
    flow = "四通道预处理流程\n\n" + "\n".join("%d. %s" % (i+1, step) for i, step in enumerate(steps))
    flow += "\n\nNPZ说明\nX_full[full_offsets[i]:full_offsets[i+1]] 还原第 i 次完整动作，形状为(采样点,4)。\n"
    flow += "rps_segments 的 X 形状为(Trial数,4,核心段点数)；rps_windows 的 X 为20维特征，X_signal 为(窗口数,4,窗长)。\n"
    flow += "连续信号的 row_time_s 按保存行序计算；counter_time_s 仅按计数器重建并含不确定性；host_timestamp 为接收时戳。均未重采样或补造丢失样点。\n"
    (output/"PREPROCESS_FLOW.txt").write_text(flow, encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description="OpenBCI 四通道 RPS 因果预处理，保留全部 Trial。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "--input-path", dest="inputs", nargs="+", required=True,
                        help="原始 TXT/CSV、ZIP 或文件夹；可同时提供同一个人的多个文件")
    parser.add_argument("--subject", required=True, help="实际受试者编号；多输入必须来自同一个人")
    parser.add_argument("--output", "--output-dir", default="RP_BINARY4CH_PREPROCESS", help="输出父目录，每次自动新建独立子目录")
    parser.add_argument("--class-names", nargs=2, default=["石头", "布"], help="确认后提供，顺序为 Marker 11、13 的手势名称；默认不猜测")
    parser.add_argument("--fs", type=float, help="通常从文件头读取，缺失时默认250；显式值必须与头部一致")
    parser.add_argument("--input-unit", choices=["uv", "mv", "v"], default="uv", help="OpenBCI GUI 原始 EXG 通常为uv，内部统一转换到uV")
    parser.add_argument("--utc-offset-hours", type=float, default=8.0, help="仅缺少格式化时间/文件日期时的时间戳日期回退")
    parser.add_argument("--low-hz", type=float, default=20.0)
    parser.add_argument("--high-hz", type=float, default=100.0)
    parser.add_argument("--order", type=int, default=4, help="Butterworth 原型阶数；实际 SOS 系数一并保存")
    parser.add_argument("--notch-hz", type=float, default=50.0, help="0关闭陷波")
    parser.add_argument("--notch-q", type=float, default=30.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--core-start", type=float, default=0.5)
    parser.add_argument("--core-stop", type=float, default=3.5)
    parser.add_argument("--baseline-start", type=float, default=-2.2)
    parser.add_argument("--baseline-stop", type=float, default=-0.2)
    parser.add_argument("--window-ms", type=float, default=200.0)
    parser.add_argument("--hop-ms", type=float, default=100.0)
    parser.add_argument("--zc-threshold-uv", type=float, default=2.0, help="固定初始门槛，未根据全文件噪声估计")
    parser.add_argument("--ssc-threshold-uv", type=float, default=2.0)
    parser.add_argument("--flat-p2p-uv", type=float, default=0.05, help="实际四路中窗口峰峰值小于等于该值则REVIEW")
    parser.add_argument("--warn-p2p-uv", type=float, default=5000.0, help="滤波窗口大幅值仅WARNING；0关闭")
    parser.add_argument("--saturation-uv", type=float, default=None, help="按实际增益确认后指定原始幅值阈值；默认不冒充已做饱和检查")
    parser.add_argument("--corr-warning", type=float, default=0.99, help="四路相关性仅WARNING；0关闭")
    parser.add_argument("--counter-modulus", type=int, default=256)
    parser.add_argument("--host-pause-seconds", type=float, default=1.0, help="超过该主机暂停间隔则保守断开滤波；普通批量抖动不当丢包")
    parser.add_argument("--split-mode", choices=["auto", "within-session", "by-date", "by-session", "none"], default="auto",
                        help="auto：多日按日期；同日多文件按Session；单文件按Run留出")
    parser.add_argument("--test-run", type=int, default=4, help="单Session默认Run4测试，其余Run开发")
    parser.add_argument("--test-date", help="by-date的测试日YYYY-MM-DD，默认最后一个实际日期")
    return parser


def run(args):
    args.subject = safe_id(args.subject)
    if args.class_names and (any(not name.strip() for name in args.class_names)
                             or len(set(args.class_names)) != 2):
        raise ValueError("两个手势名称必须非空且互不相同，顺序对应11、13。")
    if args.flat_p2p_uv < 0 or args.warn_p2p_uv < 0:
        raise ValueError("质量幅值门槛不能为负数。")
    if not 0 <= args.corr_warning <= 1:
        raise ValueError("--corr-warning 应在0到1之间。")
    if args.saturation_uv is not None and args.saturation_uv <= 0:
        raise ValueError("--saturation-uv 应为正数。")
    if args.test_run not in (1, 2, 3, 4):
        raise ValueError("当前协议 --test-run 应为1、2、3或4。")
    sources = discover_sources(args.inputs)
    parent = Path(args.output).expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = parent/("RP_BINARY4CH_%s_%s" % (args.subject, stamp))
    output.mkdir(parents=True, exist_ok=False)
    log("[输出] %s" % output)
    parts = []
    duplicates = []
    seen = {}
    config = None
    names = args.class_names or ["石头", "布"]
    for i, source in enumerate(sources, 1):
        log("[读取 %d/%d] %s" % (i, len(sources), source.display))
        digest = source.fingerprint()
        if digest in seen:
            duplicates.append({"skipped": source.display, "same_as": seen[digest], "sha256": digest})
            log("[去重] 内容相同，跳过重复副本；不重复计算Trial。")
            continue
        data = read_source(source, args)
        candidate = make_config(args, data["fs"])
        if config is None:
            config = candidate
        elif not math.isclose(config["fs"], data["fs"]):
            raise ValueError("多个输入的采样率不一致，请分批预处理。")
        part = process_session(source, digest, data, config, args, output, names)
        parts.append(part)
        seen[digest] = source.display
        description = part["description"]
        log("[完成] %d行；4通道；%d Run；%d Trial；%d窗口；建议可用%d窗口" %
            (description["n_rows"], description["run_count"], description["trial_count"],
             description["window_count"], description["usable_window_count"]))
        del data
    if not parts or config is None:
        raise ValueError("没有处理到有效输入。")
    summary = save_all(output, parts, config, args, duplicates)
    log("[总计] 保留Trial=%d，保留窗口=%d，每窗特征=20" %
        (summary["total_trials_retained"], summary["total_windows_retained"]))
    log("[QC] Trial=%s；窗口=%s" % (summary["qc_trial_counts"], summary["qc_window_counts"]))
    log("[划分] %s：%s" % (summary["split"]["mode"], summary["split"]["trial_counts"]))
    log("[查看] %s" % (output/"OPEN_REPORT.html"))
    log("[下一步] 用development训练/选型，test只评估；本脚本未拟合StandardScaler或分类器。")
    return output, summary


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
        log("[错误] %s" % exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
