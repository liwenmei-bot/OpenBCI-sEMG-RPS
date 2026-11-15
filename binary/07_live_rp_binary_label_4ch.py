#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三分类原06/07自动检测流程的石头/布适配版：先放松，再动作，自动输出，最后人工标注。

石头 Marker11 -> class0；布 Marker13 -> class1。静息由活动门控处理。
Python >= 3.9；依赖 numpy pandas scipy scikit-learn joblib，实时另需 pylsl。
独立对照版本；载入现有石头/布二分类daily_model，不载入旧三分类模型。
默认legacy沿用该模型配套的离线活动参考、启动/释放门限与投票参数。
原06与三分类07的滤波、分窗、活动分数和状态机相同；此版本保留这些流程。
界面来自三分类07，仅保留两类事后标注；不要求提前选择实际动作。
示例：python 07_live_rp_binary_classic_4ch.py --run PIPELINE_DIRECTORY
运行前核对：增加 --check-only；只检查模型和门控，不连接设备。
截图试验：--activity-mode legacy --gate-profile trial；临时ON=7.2、OFF=7.0。
默认source仍使用门控文件原值。trial来自少量截图，不是经过独立验证的参数。
"""
from __future__ import annotations

import argparse

import hashlib

import html

import io

import json

import math

import platform

import re

import sys

import time

import traceback

from collections import deque

from datetime import datetime

from pathlib import Path

from typing import Any, Dict, List, Tuple

try:
    import joblib
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    from scipy.signal import sosfilt, sosfilt_zi
    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
except ImportError as exc:
    raise SystemExit("缺少依赖：%s\n请运行：python -m pip install numpy pandas scipy scikit-learn joblib" % exc)

VERSION = "1.1.0"

SCHEMA = "openbci_rp_binary_4ch_svm_continuous_replay_v1"

DAILY_SCHEMA = "openbci_rp_binary_4ch_svm_daily_v1"

CHANNELS = ["EXG Channel %d" % i for i in range(4)]

INPUT_SCHEMA = "openbci_rp_binary_4ch_preprocess_v1"

BASE_SCHEMA = "openbci_rp_binary_4ch_svm_v1"

FEATURE_NAMES = ["ch%d_%s" % (ch, f) for ch in range(1, 5) for f in ("MAV", "RMS", "WL", "ZC", "SSC")]

META_KEYS = ["y", "marker_code", "run_id", "trial_id", "collection_date", "session_id", "subject_id",
             "split_role", "inner_cv_group", "qc_status", "qc_reasons", "usable_mask", "start_row", "stop_row"]

CONTRACT_KEYS = ["schema", "fs", "n_channels", "channel_columns", "processing_unit", "filter_mode",
    "low_hz", "high_hz", "butter_prototype_order", "notch_hz", "notch_q", "sos",
    "filter_state_initialization", "warmup_samples", "core_start_samples", "core_stop_samples",
    "window_samples", "hop_samples", "feature_names", "feature_order", "wl_normalized",
    "zc_threshold_uv", "ssc_threshold_uv", "zc_definition", "ssc_definition", "qc"]

TRIAL_RULE = "majority_vote_then_mean_ovr_decision_score_then_smaller_class_id"

def require(condition, message):
    if not condition:
        raise ValueError(message)

def json_safe(value):
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray): return json_safe(value.tolist())
    if isinstance(value, np.generic): return json_safe(value.item())
    if isinstance(value, Path): return str(value)
    if isinstance(value, float) and not np.isfinite(value): return None
    return value

def write_json(path, value):
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def write_csv(path, frame):
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10g")

def log(message, output=None):
    line = "[%s] %s" % (datetime.now().strftime("%H:%M:%S"), message)
    print(line, flush=True)
    if output is not None:
        with (output / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_features(path):
    path = Path(path).expanduser().resolve()
    npz_path = path / "rps_windows.npz" if path.is_dir() else path
    require(npz_path.is_file() and npz_path.name == "rps_windows.npz", "请指向预处理子目录或 rps_windows.npz：%s" % path)
    root = npz_path.parent
    files = ["rps_windows.npz", "preprocess_config.json", "label_mapping.json", "trial_manifest.csv"]
    for name in files:
        require((root / name).is_file(), "缺少配套文件：%s" % (root / name))
    config = read_json(root / "preprocess_config.json")
    mapping = read_json(root / "label_mapping.json")
    with np.load(npz_path, allow_pickle=False) as z:
        require("schema" in z and str(z["schema"].item()) == INPUT_SCHEMA, "NPZ不是配套四通道预处理格式。")
        key = "X_features" if "X_features" in z else "X"
        require(key in z, "缺少20维特征。")
        X = np.asarray(z[key], dtype=np.float64)
        require(X.ndim == 2 and X.shape[1] == 20 and len(X) > 0, "特征必须为(窗口数,20)。")
        require("feature_names" in z and z["feature_names"].tolist() == FEATURE_NAMES, "NPZ特征顺序不匹配。")
        require("class_names" in z and z["class_names"].shape == (2,), "缺少两类名称。")
        names = z["class_names"].tolist()
        values = {}
        for key in META_KEYS:
            require(key in z and z[key].ndim == 1 and len(z[key]) == len(X), "字段缺失或长度不符：%s" % key)
            values[key] = z[key].copy()
        require("fs" in z and float(z["fs"].item()) == float(config.get("fs", -1)), "采样率不一致。")
    require(config.get("schema") == INPUT_SCHEMA and config.get("n_channels") == 4, "配置格式/通道数错误。")
    require(config.get("channel_columns") == ["EXG Channel %d" % i for i in range(4)], "必须只使用前四通道。")
    require(config.get("feature_names") == FEATURE_NAMES and config.get("filter_mode") == "causal_sos", "特征或因果滤波配置错误。")
    require(mapping.get("marker_to_class_id") == {"11": 0, "13": 1}, "Marker到类别的映射不匹配。")
    require([mapping.get("class_id_to_name", {}).get(str(i)) for i in range(2)] == names, "类别名称文件不一致。")
    meta = pd.DataFrame(values)
    for key in ("y", "marker_code", "run_id", "start_row", "stop_row"):
        v = pd.to_numeric(meta[key], errors="raise").to_numpy(dtype=float)
        require(np.isfinite(v).all() and np.equal(v, np.floor(v)).all(), "整数字段无效：%s" % key)
        meta[key] = v.astype(np.int64)
    for key in set(META_KEYS) - {"y", "marker_code", "run_id", "start_row", "stop_row", "usable_mask"}:
        meta[key] = meta[key].astype(str)
    require(set(meta.y.unique()) == {0, 1}, "窗口未覆盖两类，或包含未知类别。")
    require(np.array_equal(meta.marker_code.to_numpy(), np.take([11, 13], meta.y.to_numpy())), "Marker与类别编号不一致。")
    require(set(meta.usable_mask.unique()).issubset({True, False, 0, 1}), "usable_mask必须为布尔掩码。")
    meta["usable_mask"] = meta.usable_mask.astype(bool)
    require(set(meta.qc_status).issubset({"PASS", "WARNING", "REVIEW"}), "QC状态错误。")
    require(meta.subject_id.nunique() == 1, "必须是同一个人，不能混入多人数据。")
    require(meta.collection_date.str.fullmatch(r"20\d{2}-\d{2}-\d{2}").all(), "日期格式错误。")
    require(not meta.duplicated(["trial_id", "start_row", "stop_row"]).any(), "存在重复窗口。")
    require((meta.stop_row > meta.start_row).all(), "存在无效窗口范围。")
    trials = pd.read_csv(root / "trial_manifest.csv", keep_default_na=False, encoding="utf-8-sig",
                         dtype={"trial_id": str, "collection_date": str, "session_id": str, "subject_id": str})
    keys = ["trial_id", "class_id", "marker_code", "run_id", "subject_id", "session_id", "collection_date", "qc_status"]
    require(set(keys).issubset(trials.columns) and len(trials) > 0, "Trial清单缺少必要字段。")
    require(not trials.trial_id.duplicated().any(), "Trial编号重复。")
    require(trials.subject_id.nunique() == 1 and trials.subject_id.iloc[0] == meta.subject_id.iloc[0], "受试者不一致。")
    require(set(meta.trial_id).issubset(set(trials.trial_id)), "窗口无法对应Trial清单。")
    require(set(trials.qc_status).issubset({"PASS", "WARNING", "REVIEW"}), "Trial QC状态错误。")
    matched = trials.set_index("trial_id").loc[meta.trial_id.tolist()].reset_index(drop=True)
    for key in ("marker_code", "run_id", "subject_id", "session_id", "collection_date"):
        require(np.array_equal(meta[key].astype(str).to_numpy(), matched[key].astype(str).to_numpy()), "Trial/窗口不一致：%s" % key)
    require(np.array_equal(meta.y.to_numpy(), matched.class_id.to_numpy(dtype=int)), "Trial类别与窗口不一致。")
    require(set(trials.class_id.astype(int)).issubset({0, 1}), "Trial存在未知类别。")
    require(np.array_equal(trials.marker_code.to_numpy(dtype=int), np.take([11, 13], trials.class_id.to_numpy(dtype=int))), "Trial Marker与类别不一致。")
    finite = np.isfinite(X).all(axis=1)
    meta["finite_features"] = finite
    meta["model_eligible"] = finite & meta.usable_mask.to_numpy() & meta.qc_status.ne("REVIEW").to_numpy()
    meta["not_used_reason"] = [";".join((["nonfinite_features"] if not finite[i] else []) +
        (["preprocess_usable_false"] if not meta.usable_mask.iloc[i] else []) +
        (["qc_review"] if meta.qc_status.iloc[i] == "REVIEW" else [])) for i in range(len(meta))]
    return {"X": X, "meta": meta, "trials": trials, "config": config, "mapping": mapping,
            "root": root, "hashes": {name: file_sha256(root / name) for name in files}}

def check_contract(expected, actual):
    for key in CONTRACT_KEYS:
        require(key in expected and key in actual, "预处理配置缺少字段：%s" % key)
        if key == "sos":
            a, b = np.asarray(expected[key]), np.asarray(actual[key])
            equal = a.shape == b.shape and np.allclose(a, b, rtol=1e-12, atol=1e-12)
        else:
            equal = json_safe(expected[key]) == json_safe(actual[key])
        require(equal, "历史/当天预处理不一致：%s；请沿用基础模型的预处理参数。" % key)

def predict(pipeline, X):
    """Binary SVC: positive signed margin belongs to class 1 (paper)."""
    margin = np.asarray(pipeline.decision_function(X), dtype=float)
    require(margin.shape == (len(X),) and np.isfinite(margin).all(), "二分类SVM决策输出异常。")
    require(np.array_equal(pipeline.classes_, [0, 1]), "需要重新训练的石头/布二分类模型，不能直接载入三分类模型。")
    scores = np.column_stack([-margin, margin])
    return pipeline.classes_[(margin >= 0).astype(int)].astype(np.int64), scores


def check_pipeline(bundle):
    pipeline = bundle.get("pipeline")
    require(isinstance(pipeline, Pipeline) and isinstance(pipeline.named_steps.get("scaler"), StandardScaler)
            and isinstance(pipeline.named_steps.get("svm"), SVC), "模型必须是StandardScaler+SVC。")
    require(bundle.get("feature_names") == FEATURE_NAMES and bundle.get("class_ids") == [0, 1], "模型特征或类别不匹配。")
    require(np.array_equal(pipeline.classes_, [0, 1]) and pipeline["svm"].break_ties
            and pipeline["svm"].decision_function_shape == "ovr", "模型决策规则不匹配。")

def load_daily(path):
    path = Path(path).expanduser().resolve()
    require(path.is_file(), "找不到当天模型：%s" % path)
    bundle = joblib.load(path)
    require(isinstance(bundle, dict) and bundle.get("schema") == DAILY_SCHEMA
            and bundle.get("artifact_role") == "daily_calibrated", "请使用03输出的daily_model.joblib。")
    check_pipeline(bundle)
    require(bundle.get("calibration_selection_runs") == [1, 2]
            and bundle.get("independent_test_runs_not_used_for_fit") == [3, 4], "当天模型的训练/测试协议不匹配。")
    expected_runs = [1, 2] if bundle.get("calibration_applied") else []
    require(bundle.get("training", {}).get("today_runs_used_for_fit") == expected_runs,
            "当天模型训练Run声明异常，不能据此作独立评估。")
    return bundle, path

def audit_for(data):
    session = str(data["trials"].session_id.iloc[0])
    path = data["root"] / "sessions" / session / "session_audit.json"
    if not path.is_file():
        return {}
    value = read_json(path)
    require(value.get("session_id") == session, "Session审计文件编号不一致。")
    return value

def validate_protocol(data):
    tr = data["trials"]
    require(tr.collection_date.nunique() == 1 and tr.session_id.nunique() == 1, "请输入单日期、单Session的预处理目录。")
    require(set(tr.run_id.astype(int)) == {1, 2, 3, 4}, "需要完整4Run，不自动拼接或随机拆分窗口。")
    for run in (1, 2, 3, 4):
        group = tr[tr.run_id.eq(run)]
        require(len(group) == 12 and group.class_id.astype(int).value_counts().to_dict() == {0: 6, 1: 6},
                "Run%d需要12Trial、两类各6，请核查采集完整性。" % run)

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

class CausalWindowStream:
    """只接收信号/连续性。窗口日程与调用分块大小无关；Marker不属于此接口。"""
    def __init__(self, config):
        self.config = config
        self.width, self.hop = int(config["window_samples"]), int(config["hop_samples"])
        self.processor = CausalEMGFilter.from_config(config)
        self.position, self.segment_id, self.valid = 0, 0, False
        self.raw_buffer = np.empty((0, 4))
        self.filtered_buffer = np.empty((0, 4))
        self.next_end = None

    def reset(self, row):
        self.processor.reset()
        self.segment_id += 1
        self.raw_buffer = np.empty((0, 4))
        self.filtered_buffer = np.empty((0, 4))
        self.next_end = int(row) + int(self.config["warmup_samples"]) + self.width

    def push(self, samples, break_before=None):
        values = np.asarray(samples, dtype=np.float64)
        require(values.ndim == 2 and values.shape[1] == 4, "流输入必须为(N,4)。")
        flags = np.zeros(len(values), bool) if break_before is None else np.asarray(break_before, dtype=bool)
        require(flags.shape == (len(values),), "连续性标记长度错误。")
        finite = np.isfinite(values).all(axis=1)
        boundaries = np.unique(np.r_[0, len(values), np.flatnonzero(flags),
                                      np.flatnonzero(finite[1:] != finite[:-1]) + 1])
        frames, features = [], []
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            lo, hi = int(lo), int(hi)
            if hi <= lo: continue
            start, stop = self.position + lo, self.position + hi
            if not finite[lo]:
                self.valid = False
                self.processor.reset()
                self.raw_buffer = np.empty((0, 4))
                self.filtered_buffer = np.empty((0, 4))
                continue
            if flags[lo] or not self.valid:
                self.reset(start)
            self.valid = True
            filtered = self.processor.process(values[lo:hi])
            raw = np.concatenate([self.raw_buffer, values[lo:hi]])
            xf = np.concatenate([self.filtered_buffer, filtered])
            origin = start - len(self.raw_buffer)
            while self.next_end <= stop:
                end = int(self.next_end)
                a, b = end - self.width - origin, end - origin
                require(a >= 0 and b <= len(raw), "流窗口缓冲区错位。")
                rw = raw[a:b]
                # 与01的padded_slice/训练特征保持同样float32窗口精度。
                fw = xf[a:b].astype(np.float32)
                qc, reasons, usable, _ = inspect_window(rw, fw, np.ones(self.width, bool),
                    np.zeros(self.width, bool), 0, self.width, True, self.config)
                frame = {"start_row": end - self.width, "stop_row": end, "decision_row": end - 1,
                         "segment_id": self.segment_id, "qc_status": qc, "qc_reasons": ";".join(reasons),
                         "usable": bool(usable)}
                feature = extract_features(fw, self.config["zc_threshold_uv"], self.config["ssc_threshold_uv"])
                frames.append(frame); features.append(feature)
                self.next_end += self.hop
            self.raw_buffer = raw[-self.width:].copy()
            self.filtered_buffer = xf[-self.width:].copy()
        self.position += len(values)
        return frames, features

def load_signal(data, mode="auto"):
    session = str(data["trials"].session_id.iloc[0])
    path = data["root"] / "sessions" / session / "continuous_4ch.npz"
    require(path.is_file(), "缺少连续信号文件：%s；请保留01输出的sessions目录。" % path)
    with np.load(path, allow_pickle=False) as z:
        require(str(z["schema"].item()) == INPUT_SCHEMA and z["channel_columns"].tolist() == CHANNELS, "连续文件格式/通道错误。")
        raw = z["X_raw"].astype(np.float64)
        index, timestamps = z["sample_index"].copy(), z["host_timestamp"].copy()
        digest = str(z["source_sha256"].item())
        require(float(z["fs"].item()) == data["config"]["fs"], "连续文件采样率错误。")
        # 不读取marker_channel / gesture_class / candidate_rest_mask作为推理输入。
    require(raw.ndim == 2 and raw.shape[1] == 4 and len(index) == len(raw) == len(timestamps), "连续信号字段长度错误。")
    audit = audit_for(data)
    require(not audit.get("source_sha256") or audit["source_sha256"] == digest, "连续文件原始SHA与审计不一致。")
    original = Path(audit.get("source_file", ""))
    precision = "stored_float32_refiltered"
    if mode != "stored" and original.is_file() and original.suffix.lower() in {".txt", ".csv"}:
        require(file_sha256(original) == digest, "原TXT内容已变，不与预处理数据混用。")
        frame = pd.read_csv(original, comment="%", skipinitialspace=True, usecols=CHANNELS)
        values = frame[CHANNELS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        scale = {"uv": 1.0, "mv": 1000.0, "v": 1000000.0}[data["config"]["input_unit"]]
        require(values.shape == raw.shape, "原TXT行数与连续文件不一致。")
        raw = values * scale
        precision = "original_text_float64_verified_sha256"
    elif mode == "txt":
        raise ValueError("原始TXT无法定位；请保留审计中记录的原路径，或用 --raw-mode stored。")
    breaks, issues, _ = continuity(index, timestamps, data["config"])
    return {"raw": raw, "breaks": breaks, "issues": issues, "precision": precision,
            "source_sha256": digest, "fs": float(data["config"]["fs"])}

def collect_frames(signal, config, stop=None, chunk_samples=1000, output=None):
    n = len(signal["raw"]) if stop is None else min(len(signal["raw"]), int(stop))
    stream, frames, features = CausalWindowStream(config), [], []
    for lo in range(0, n, chunk_samples):
        hi = min(n, lo + chunk_samples)
        f, x = stream.push(signal["raw"][lo:hi], signal["breaks"][lo:hi])
        frames.extend(f); features.extend(x)
    require(len(frames) > 0, "没有完成预热的可用窗口。")
    return pd.DataFrame(frames), np.stack(features)

def classify_frames(frames, features, pipeline, output=None):
    predicted = np.full(len(frames), -1, dtype=np.int64)
    scores = np.full((len(frames), 2), np.nan)
    indices = np.flatnonzero(frames.usable.to_numpy() & np.isfinite(features).all(axis=1))
    for lo in range(0, len(indices), 256):
        part = indices[lo:lo + 256]
        predicted[part], scores[part] = predict(pipeline, features[part])
        if output is not None and lo % 2560 == 0:
            log("固定SVM推理：%d/%d 窗。" % (min(lo + len(part), len(indices)), len(indices)), output)
    frames = frames.copy()
    frames["predicted_class_id"] = predicted
    for k in range(2): frames["decision_%d" % k] = scores[:, k]
    return frames

def frames_in_intervals(frames, intervals):
    mask = np.zeros(len(frames), dtype=bool)
    starts, stops = frames.start_row.to_numpy(), frames.stop_row.to_numpy()
    for a, b in intervals:
        mask |= (starts >= int(a)) & (stops <= int(b))
    return mask

def candidate_intervals(data, trial_ids=None):
    path = data["root"] / "rest_candidates.csv"
    require(path.is_file(), "缺少rest_candidates.csv，无法构建源噪声候选参考。")
    rest = pd.read_csv(path, keep_default_na=False, encoding="utf-8-sig")
    require({"trial_id", "start_row", "stop_row", "qc_status"}.issubset(rest.columns), "基线候选清单字段不全。")
    if trial_ids is not None: rest = rest[rest.trial_id.isin(trial_ids)]
    rest = rest[rest.qc_status.isin(["PASS", "WARNING"])]
    if "numerical_usable" in rest:
        rest = rest[rest.numerical_usable.astype(str).str.lower().isin(["true", "1"])]
    return [(int(r.start_row), int(r.stop_row)) for r in rest.itertuples(index=False) if int(r.stop_row) > int(r.start_row)]

def run_intervals(data):
    path = data["root"] / "run_manifest.csv"
    require(path.is_file(), "缺少run_manifest.csv。")
    runs = pd.read_csv(path, keep_default_na=False, encoding="utf-8-sig")
    require(len(runs) == 4 and set(runs.run_id.astype(int)) == {1, 2, 3, 4}, "需要四个独立Run边界。")
    result = {}
    for r in runs.itertuples(index=False):
        a, b = int(r.start_row), int(r.end_row)
        require(0 <= a < b, "Run边界未核实。")
        result[int(r.run_id)] = (a, b + 1)
    return result

def activity_scores(features, noise_reference):
    rms = features[:, [1, 6, 11, 16]]
    normalized = rms / np.asarray(noise_reference)[None, :]
    return np.sort(normalized, axis=1)[:, -2:].mean(axis=1)

class GestureDetector:
    """只接收逐窗预测、活动分数、数值QC和分段号；没有Marker/标签参数。"""
    def __init__(self, params):
        self.params = params
        self.segment_id = None
        self.events = []
        self.active_event = None
        self.reset_state()

    def reset_state(self):
        self.armed, self.active, self.emitted = False, False, False
        self.on_count, self.off_count = 0, 0
        self.onset_candidate_row = None
        self.history = deque(maxlen=int(self.params["vote_windows"]))

    def close(self, row, reason):
        if self.active_event is not None:
            self.active_event["release_row"] = int(row)
            self.active_event["release_reason"] = reason
        self.active_event = None

    def step(self, row, segment_id, usable, activity, predicted, decision_scores):
        if segment_id != self.segment_id:
            self.close(row, "continuity_segment_change")
            self.reset_state(); self.segment_id = segment_id
        if not usable or predicted < 0 or not np.isfinite(activity):
            self.close(row, "signal_quality_review")
            self.reset_state()
            return "quality_lock", None
        low = activity <= self.params["off_threshold"]
        self.off_count = self.off_count + 1 if low else 0
        if self.off_count >= self.params["release_windows"]:
            self.close(row, "low_activity_release")
            self.active, self.emitted, self.armed = False, False, True
            self.on_count = 0; self.onset_candidate_row = None; self.history.clear()
            return "released", None
        if not self.armed:
            return "waiting_for_release", None
        if self.emitted:
            return "active_latched", None
        settle_samples = int(self.params.get("settle_samples", 0))
        if not self.active:
            if activity >= self.params["on_threshold"]:
                if self.on_count == 0: self.onset_candidate_row = int(row)
                self.on_count += 1
                if row - self.onset_candidate_row >= settle_samples:
                    self.history.append((int(predicted), np.asarray(decision_scores, dtype=float)))
                else:
                    self.history.clear()
            else:
                self.on_count = 0; self.onset_candidate_row = None; self.history.clear()
            if self.on_count < self.params["on_windows"]:
                return "quiet_or_candidate", None
            self.active = True
        else:
            # 释放滞回只用于保持事件锁定；它不能授权在低活动时发出新事件。
            if activity < self.params["on_threshold"]:
                self.history.clear()
                return "activity_below_on_waiting", None
            if row - self.onset_candidate_row < settle_samples:
                self.history.clear()
                return "settling", None
            self.history.append((int(predicted), np.asarray(decision_scores, dtype=float)))
        if activity < self.params["on_threshold"]:
            self.history.clear()
            return "activity_below_on_waiting", None
        if row - self.onset_candidate_row < settle_samples:
            self.history.clear()
            return "settling", None
        if len(self.history) < self.params["vote_windows"]:
            return "collecting_votes", None
        labels = np.asarray([x[0] for x in self.history])
        votes = np.bincount(labels, minlength=2)
        if votes.max() < self.params["minimum_votes"]:
            return "unstable", None
        means = np.mean([x[1] for x in self.history], axis=0)
        tied = np.flatnonzero(votes == votes.max())
        winner = int(tied[np.argmax(means[tied])])
        event = {"event_id": len(self.events) + 1, "decision_row": int(row),
                 "onset_candidate_row": self.onset_candidate_row, "class_id": winner,
                 "vote_agreement": float(votes[winner] / len(labels)),
                 "activity_score": float(activity), "release_row": None, "release_reason": "recording_ended_or_not_released"}
        self.events.append(event); self.active_event = event; self.emitted = True
        return "triggered", event

def detect(frames, scores, params, keep_trace=False):
    detector = GestureDetector(params)
    trace = []
    for i, frame in enumerate(frames.itertuples(index=False)):
        decision = np.array([frame.decision_0, frame.decision_1])
        state, event = detector.step(int(frame.decision_row), int(frame.segment_id), bool(frame.usable),
                                    scores[i], int(frame.predicted_class_id), decision)
        if keep_trace:
            trace.append({"decision_row": int(frame.decision_row), "activity_score": float(scores[i]),
                          "detector_state": state, "trigger_event_id": event["event_id"] if event else 0})
    return [dict(e) for e in detector.events], pd.DataFrame(trace)

def events_in_intervals(events, intervals):
    return [e for e in events if any(a <= e["decision_row"] < b for a, b in intervals)]

def score_trials(events, trials, fs, n_rows):
    records = []
    ordered = sorted(events, key=lambda e: e["decision_row"])
    for r in trials.itertuples(index=False):
        a, b = int(r.start_row), int(r.end_row)
        scorable = 0 <= a < b <= n_rows and int(r.core_stop_row) <= b
        inside = [e for e in ordered if a <= e["decision_row"] < b] if scorable else []
        first = inside[0] if inside else None
        preceding = [e for e in ordered if e["decision_row"] < a and "release_row" in e
                     and (e["release_row"] is None or not np.isfinite(e["release_row"]) or e["release_row"] >= a)]
        preceding = preceding[-1] if preceding else None
        predicted = int(first["class_id"]) if first else -1
        item = r._asdict()
        item.update({"marker_interval_scorable": bool(scorable), "n_triggers": len(inside),
            "first_predicted_class_id": predicted, "first_trigger_correct": bool(scorable and predicted == int(r.class_id)),
            "any_correct_trigger": any(e["class_id"] == int(r.class_id) for e in inside),
            "missed": bool(scorable and not inside), "wrong_first_class": bool(first and predicted != int(r.class_id)),
            "duplicate_triggers": max(0, len(inside) - 1),
            "first_event_id": int(first["event_id"]) if first else -1,
            "latency_from_marker_s": (first["decision_row"] - a) / fs if first else np.nan,
            "preceding_output_held_into_marker": bool(preceding),
            "preceding_event_id": int(preceding["event_id"]) if preceding else -1,
            "preceding_output_from_marker_s": (preceding["decision_row"] - a) / fs if preceding else np.nan,
            "score_reason": "" if scorable else "unverified_or_out_of_bounds_marker_interval"})
        records.append(item)
    result = pd.DataFrame(records)
    counts = result.class_id.astype(int).value_counts()
    correct = result[result.first_trigger_correct].class_id.astype(int).value_counts()
    recalls = [int(correct.get(k, 0)) / int(counts[k]) if int(counts.get(k, 0)) else None for k in range(2)]
    cm = np.zeros((2, 3), dtype=int)
    for r in result.itertuples(index=False): cm[int(r.class_id), int(r.first_predicted_class_id) if r.first_predicted_class_id >= 0 else 2] += 1
    latencies = result.loc[result.first_trigger_correct, "latency_from_marker_s"].to_numpy(dtype=float)
    score = {"n_trials_retained": len(result), "n_marker_intervals_scorable": int(result.marker_interval_scorable.sum()),
        "correct_first_triggers": int(result.first_trigger_correct.sum()),
        "first_trigger_accuracy_all_trials": float(result.first_trigger_correct.mean()) if len(result) else None,
        "first_trigger_balanced_recall": float(np.mean([v for v in recalls if v is not None])) if len(result) else None,
        "missed_trials": int(result.missed.sum()), "wrong_first_class_trials": int(result.wrong_first_class.sum()),
        "trials_with_duplicate_triggers": int(result.duplicate_triggers.gt(0).sum()),
        "duplicate_trigger_count": int(result.duplicate_triggers.sum()), "class_recall": recalls,
        "preceding_output_held_into_trial_count": int(result.preceding_output_held_into_marker.sum()),
        "strict_misses_with_preceding_held_output": int((result.missed & result.preceding_output_held_into_marker).sum()),
        "confusion_stone_paper_no_trigger": cm.tolist(),
        "median_correct_trigger_latency_from_marker_s": float(np.median(latencies)) if len(latencies) else None,
        "p90_correct_trigger_latency_from_marker_s": float(np.quantile(latencies, .9)) if len(latencies) else None}
    return result, score

def excluded_intervals(data):
    session = str(data["trials"].session_id.iloc[0])
    path = data["root"] / "sessions" / session / "excluded_actions.csv"
    if not path.is_file():
        return []
    rows = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    return [(int(r.start_row), int(r.available_stop_row)) for r in rows.itertuples(index=False)
            if int(r.available_stop_row) > int(r.start_row)]


def choose_detector(source, frames, features, fs, n_rows, output, settle_candidates_ms=(0, 200, 400, 600, 800)):
    trials = source["trials"]
    run1 = trials[trials.run_id.eq(1)]
    run2 = trials[trials.run_id.eq(2)]
    baseline1 = candidate_intervals(source, set(run1.trial_id))
    base_mask = frames_in_intervals(frames, baseline1) & frames.usable.to_numpy()
    core1 = [(int(t.core_start_row), int(t.core_stop_row)) for t in run1.itertuples(index=False)]
    act_mask = frames_in_intervals(frames, core1) & frames.usable.to_numpy()
    require(base_mask.sum() >= 20 and act_mask.sum() >= 20, "源Run1基线候选或动作参考不足20窗，无法设置活动门控。")
    rms = features[:, [1, 6, 11, 16]]
    noise = np.maximum(np.median(rms[base_mask], axis=0), 1e-3)
    scores = activity_scores(features, noise)
    noise99 = float(np.quantile(scores[base_mask], .99))
    noise50 = float(np.median(scores[base_mask]))
    action10 = float(np.quantile(scores[act_mask], .1))
    thresholds = sorted(set(max(1.2, noise99 * scale) for scale in (1.0, 1.25, 1.5)))
    run_bounds = run_intervals(source)
    validation_bound = [run_bounds[2]]
    baseline2 = candidate_intervals(source, set(run2.trial_id))
    waits = sorted(set(float(value) for value in settle_candidates_ms))
    require(len(waits) > 0 and all(np.isfinite(value) and value >= 0 for value in waits), "等待时间必须为有限非负毫秒数。")
    candidates, best_rank, best = [], None, None
    for on in thresholds:
        off = min(on * .9, max(noise50 * 1.05, min(noise99 * .9, on * .65)))
        settings = [(width, minimum, wait) for width, minimum in ((3, 3), (5, 4)) for wait in waits]
        for width, minimum, wait in settings:
            params = {"on_threshold": float(on), "off_threshold": float(off), "on_windows": 2,
                      "release_windows": 3, "vote_windows": width, "minimum_votes": minimum,
                      "settle_seconds": wait / 1000, "settle_samples": int(round(wait * fs / 1000)),
                      "emit_requires_current_activity_above_on": True,
                      "vote_history_excludes_settling_and_low_activity": True,
                      "noise_reference_rms_uv": noise.tolist(), "activity_score": "mean_of_top2_channel_rms_to_noise_ratios",
                      "initial_state_requires_observed_release": True}
            events, _ = detect(frames, scores, params)
            validation_events = events_in_intervals(events, validation_bound)
            val_trials, metrics_val = score_trials(validation_events, run2, fs, n_rows)
            baseline_triggers = len(events_in_intervals(validation_events, baseline2))
            unsupported = excluded_intervals(source)
            outside = len([e for e in validation_events if not any(a <= e["decision_row"] < b for a, b in unsupported) and not any(int(t.start_row) <= e["decision_row"] < int(t.end_row) for t in run2.itertuples(index=False))])
            row = {"candidate_id": len(candidates) + 1, "on_threshold": on, "off_threshold": off,
                   "vote_windows": width, "minimum_votes": minimum,
                   "settle_ms": wait,
                   "correct_first_triggers": metrics_val["correct_first_triggers"],
                   "trial_count": len(run2), "first_trigger_accuracy": metrics_val["first_trigger_accuracy_all_trials"],
                   "missed_trials": metrics_val["missed_trials"], "wrong_first_trials": metrics_val["wrong_first_class_trials"],
                   "duplicate_triggers": metrics_val["duplicate_trigger_count"], "baseline_candidate_triggers": baseline_triggers,
                   "outside_marker_triggers": outside,
                   "median_latency_from_marker_s": metrics_val["median_correct_trigger_latency_from_marker_s"]}
            candidates.append(row)
            latency = row["median_latency_from_marker_s"]
            rank = (row["correct_first_triggers"], -row["duplicate_triggers"], -baseline_triggers, -outside,
                    -(latency if latency is not None else float("inf")), on, -wait, -width)
            if best_rank is None or rank > best_rank:
                best_rank, best = rank, dict(params)
                best["selected_candidate_id"] = row["candidate_id"]
    candidate_frame = pd.DataFrame(candidates)
    candidate_frame["selected"] = candidate_frame.candidate_id.eq(best["selected_candidate_id"])
    write_csv(output / "source_gate_validation.csv", candidate_frame)
    best.update({"noise_reference_source": "source_Run1_candidate_baselines_not_verified_rest",
        "baseline_reference_verified_rest": False, "source_baseline_frame_count": int(base_mask.sum()),
        "source_action_frame_count": int(act_mask.sum()), "noise_score_q99": noise99, "action_score_q10": action10,
        "activity_reference_overlap_warning": action10 <= noise99,
        "settle_candidates_ms": waits,
        "selection_scope": "source_Run1_reference_Run2_gate_selection_no_target_data"})
    return best
import csv
import queue
import threading

LIVE_VERSION = "1.0.1-rp-classic-tune"
LIVE_SCHEMA = "openbci_rp_binary_4ch_svm_live_v1"
STATE_NAMES = {
    "waiting_for_release": "请先放松手臂", "released": "已放松，等待动作",
    "quiet_or_candidate": "等待动作稳定", "collecting_votes": "正在稳定判决",
    "settling": "等待动作稳定", "unstable": "类别尚未稳定",
    "activity_below_on_waiting": "活动不足，等待稳定", "active_latched": "已识别，请放松后再做下一次",
    "triggered": "已识别，请放松后再做下一次", "quality_lock": "信号异常，请检查后放松",
    "warmup": "滤波预热中", "no_data": "数据中断，等待恢复", "stale": "数据积压，暂停输出",
    "connected": "已连接，正在接收数据", "connecting": "正在查找LSL数据流", "stopped": "已停止",
    "resyncing": "正在重新订阅LSL，等待新数据",
    "rest_calibrating": "正在测量静息，请保持手臂放松；暂不识别",
    "rest_calibration_failed": "静息测量未通过，请放松后点击“重新测静息”",
}


def activity_indicator(activity, params, state=None):
    """显示只解释固定门限，不改变门控；条的50%位置为ON，数值不是概率。"""
    on, off = float(params["on_threshold"]), float(params["off_threshold"])
    maximum = max(1., on*2)
    if activity is None or not np.isfinite(activity):
        return {"value": 0., "maximum": maximum, "fraction": 0., "zone": "unknown",
            "text": "等待有效活动数据"}
    activity = float(activity)
    if activity <= off:
        zone, explanation = "below_off", "低于释放门限，需连续%d窗" % params["release_windows"]
    elif activity < on:
        zone, explanation = "hysteresis", "滞回区：高于释放门限、低于启动门限"
    else:
        zone, explanation = "above_on", "达到启动门限，仍需等待与投票"
    if state in {"active_latched", "triggered"}:
        explanation = "本次已输出，等待持续放松后释放"
    elif state == "waiting_for_release":
        explanation = "请先持续放松，解除锁定后才允许下一次输出"
    elif state == "collecting_votes":
        explanation = "正在累积类别投票"
    elif state == "unstable":
        explanation = "活动已达到门限，但类别投票尚未稳定"
    elif state == "settling":
        explanation = "活动已达到门限，正在等待动作稳定"
    elif state in {"quality_lock", "warmup", "no_data", "resyncing"}:
        explanation = "当前暂停输出，请查看识别状态"
    if state is not None:
        explanation = {"below_off": "低于释放门限", "hysteresis": "滞回区", "above_on": "高于启动门限"}[zone]+"；"+explanation
    return {"value": min(maximum, max(0., activity)), "maximum": maximum,
        "fraction": min(1., max(0., activity/maximum)), "zone": zone,
        "text": "活动 %.2f ｜ 释放≤%.3f ｜ 启动≥%.3f ｜ %s" % (activity, off, on, explanation)}


class ReceiveHealth:
    """接收端5秒滚动速率；LSL数据年龄不冒充板卡端到端延迟。"""
    def __init__(self):
        self.points, self.total = deque(), 0

    def update(self, count, clock):
        self.total += int(count)
        self.points.append((float(clock), self.total))
        while len(self.points) > 2 and self.points[1][0] <= clock-5.:
            self.points.popleft()
        duration = clock-self.points[0][0]
        return (self.total-self.points[0][1])/duration if duration >= .5 else None


def open_live_inlet(lsl, selected, timeout):
    """每次订阅创建全新的inlet；打开失败也主动关闭，不复用旧对象。"""
    inlet = lsl.StreamInlet(selected, max_buflen=2, recover=False, processing_flags=lsl.proc_clocksync)
    try:
        inlet.open_stream(timeout=timeout)
        return inlet
    except Exception:
        try: inlet.close_stream()
        except Exception: pass
        raise


def flush_live_inlet(inlet):
    operation = getattr(inlet, "flush", None)
    if operation is None: return 0, False
    return max(0, int(operation())), True


def inlet_queue_size(inlet):
    try:
        operation = getattr(inlet, "samples_available", None)
        return max(0, int(operation())) if operation is not None else None
    except Exception:
        # 诊断读取失败不改变识别；实际接收异常仍由pull_chunk报告。
        return None


def make_output(parent, prefix, subject=""):
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in str(subject))[:60]
    target = Path(parent).expanduser().resolve() / (prefix + ("_" + safe if safe else "") + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    target.mkdir(parents=True, exist_ok=False)
    return target


def validate_detector(bundle, model_path, params):
    require(params.get("schema") == SCHEMA, "门控配置格式错误，请使用05 v1.1或本脚本prepare-only生成的配置。")
    require(params.get("script_version") == "1.1.0", "当前实时引擎对应05 v1.1.0，请重新生成匹配的门控配置。")
    require(params.get("model_sha256") == file_sha256(model_path), "门控配置和当前模型不是同一个组合，请用该模型重新准备门控。")
    check_contract(bundle["preprocess_config"], params.get("preprocess_config", {}))
    require(params.get("source_hashes") == bundle.get("today_hashes"), "门控源数据和模型的校准数据不一致。")
    require(params.get("source_calibration_runs") == [1, 2] and params.get("target_used_for_selection") is False,
            "门控必须仅由源Run1+2确定。")
    require(params.get("emit_requires_current_activity_above_on") is True and
            params.get("vote_history_excludes_settling_and_low_activity") is True,
            "缺少v1.1的低活动保护设置。")
    noise = np.asarray(params.get("noise_reference_rms_uv", []), dtype=float)
    require(noise.shape == (4,) and np.isfinite(noise).all() and (noise > 0).all(), "四通道噪声参考无效。")
    on, off = float(params["on_threshold"]), float(params["off_threshold"])
    require(np.isfinite([on, off]).all() and on > off >= 0, "门控阈值无效。")
    for key in ("on_windows", "release_windows", "vote_windows", "minimum_votes", "settle_samples"):
        value = params.get(key)
        require(isinstance(value, (int, np.integer)) and value >= (0 if key == "settle_samples" else 1), "门控整数参数无效：" + key)
    require(params["vote_windows"] / 2 < params["minimum_votes"] <= params["vote_windows"], "投票设置无效。")
    fs = float(bundle["preprocess_config"]["fs"])
    require(np.isfinite(params.get("settle_seconds", np.nan)) and params["settle_seconds"] >= 0 and
            int(round(params["settle_seconds"] * fs)) == params["settle_samples"], "等待秒数和采样点不一致。")
    for key in ("window_samples", "hop_samples"):
        require(params.get(key) == bundle["preprocess_config"][key], "门控窗口配置不一致：" + key)


def prepare_detector(args):
    bundle, model_path = load_daily(args.model)
    sidecar = model_path.parent / "calibration_config.json"
    saved = read_json(sidecar) if sidecar.is_file() else {}
    source_path = args.source_data or saved.get("today_directory")
    require(source_path, "找不到校准预处理目录，请指定 --source-data，或保留模型旁calibration_config.json。")
    source = load_features(source_path)
    validate_protocol(source)
    check_contract(bundle["preprocess_config"], source["config"])
    require(source["hashes"] == bundle.get("today_hashes"), "源数据与当天模型记录的校准哈希不匹配。")
    require(str(source["trials"].subject_id.iloc[0]) == str(bundle["subject_id"]) and
            str(source["trials"].collection_date.iloc[0]) == bundle["calibration_date"], "校准受试者或日期不一致。")
    output = make_output(args.output, "RPS_LIVE_SETUP", bundle["subject_id"])
    digest = file_sha256(model_path)
    log("只读取校准记录的Run1+2及之前数据；SVM和标准化器保持冻结。", output)
    signal = load_signal(source, args.raw_mode)
    stop = run_intervals(source)[2][1]
    frames, features = collect_frames(signal, source["config"], stop, 1000)
    frames = classify_frames(frames, features, bundle["pipeline"], output)
    params = choose_detector(source, frames, features, float(source["config"]["fs"]), len(signal["raw"]), output,
                             args.settle_candidates_ms)
    params.update({"schema": SCHEMA, "script_version": "1.1.0", "prepared_by_live_script": LIVE_VERSION,
        "model_sha256": digest, "source_hashes": source["hashes"], "source_calibration_runs": [1, 2],
        "target_used_for_selection": False, "source_limit_exclusive_row": stop,
        "window_samples": source["config"]["window_samples"], "hop_samples": source["config"]["hop_samples"],
        "class_names": bundle["class_names"], "preprocess_config": source["config"],
        "calibration_date": bundle["calibration_date"], "raw_precision": signal["precision"]})
    validate_detector(bundle, model_path, params)
    require(file_sha256(model_path) == digest, "模型文件意外发生变化。")
    write_json(output / "replay_detector_config.json", params)
    log("门控准备完成：ON=%.3f，OFF=%.3f，等待%.3f秒，%d窗至少%d票。" %
        (params["on_threshold"], params["off_threshold"], params["settle_seconds"],
         params["vote_windows"], params["minimum_votes"]), output)
    log("实时配置：%s" % (output / "replay_detector_config.json"), output)
    return output


class TransportGuard:
    """LSL只提供收到的数据；无原始计数器时不能证明板卡端无丢样。"""
    def __init__(self, fs, gap_seconds=.2, counter_modulus=256):
        self.fs, self.gap_seconds, self.modulus = float(fs), float(gap_seconds), int(counter_modulus)
        self.last_stamp, self.last_counter = None, None

    def inspect(self, stamps, counters=None):
        stamps = np.asarray(stamps, dtype=float)
        require(stamps.ndim == 1, "LSL时间戳必须为一维。")
        flags, invalid, issues = np.zeros(len(stamps), bool), np.zeros(len(stamps), bool), []
        if counters is not None:
            counters = np.asarray(counters, dtype=float)
            require(counters.shape == stamps.shape, "计数器长度不一致。")
        for i, stamp in enumerate(stamps):
            reasons = []
            dt = stamp - self.last_stamp if self.last_stamp is not None else np.nan
            if not np.isfinite(stamp):
                reasons.append("invalid_lsl_timestamp"); invalid[i] = True
            elif self.last_stamp is None:
                flags[i] = True
            elif dt < -self.gap_seconds:
                reasons.append("large_lsl_timestamp_regression"); invalid[i] = True
            elif dt <= 0:
                # GUI用每批发送时刻倒推样本时间；相邻批次可短暂重叠。
                # LSL传递顺序用于窗口日程，小幅时间抖动仅记录，不冒充板卡乱序。
                issues.append({"offset": i, "reason": "lsl_batch_timestamp_jitter_warning_only", "timestamp_dt_s": dt})
            elif dt > self.gap_seconds:
                reasons.append("lsl_timestamp_gap")
            if counters is not None:
                counter = counters[i]
                good = np.isfinite(counter) and counter == round(counter) and 0 <= counter < self.modulus
                if not good:
                    reasons.append("invalid_original_sample_counter"); invalid[i] = True
                elif self.last_counter is not None and int((counter - self.last_counter) % self.modulus) != 1:
                    reasons.append("original_sample_counter_discontinuity")
                self.last_counter = counter if good else None
            if reasons:
                flags[i] = True
                issues.append({"offset": i, "reason": ";".join(reasons), "timestamp_dt_s": dt})
            self.last_stamp = float(stamp) if np.isfinite(stamp) else None
        return flags, invalid, issues


def live_gate_options(args):
    return {"mode": getattr(args, "activity_mode", "legacy"),
        "rest_seconds": float(getattr(args, "rest_seconds", 8.)),
        "minimum_active_channels": int(getattr(args, "min_active_channels", 2)),
        "on_z": float(getattr(args, "activity_on_z", 6.)),
        "off_z": float(getattr(args, "activity_off_z", 3.))}


def runtime_detector_params(params, args):
    result = dict(params)
    options = live_gate_options(args)
    profile = getattr(args, "gate_profile", "source")
    explicit = {key: getattr(args, attr, None) for key, attr in [
        ("on_threshold", "gate_on"), ("off_threshold", "gate_off"),
        ("settle_seconds", "gate_settle_seconds"), ("vote_windows", "gate_vote_windows"),
        ("minimum_votes", "gate_min_votes"), ("release_windows", "gate_release_windows")]}
    require(profile in {"source", "trial"}, "未知门控参数方案。")
    legacy_thresholds = profile == "trial" or any(explicit[key] is not None for key in ["on_threshold", "off_threshold"])
    require(not legacy_thresholds or options["mode"] == "legacy",
        "trial及--gate-on/--gate-off采用legacy分数，不能与robust评分混用。")
    if options["mode"] == "robust":
        result.update({"on_threshold": options["on_z"], "off_threshold": options["off_z"],
            "activity_score": "kth_largest_channel_robust_rms_z",
            "minimum_active_channels": options["minimum_active_channels"],
            "runtime_activity_reference": "live_user_instructed_rest",
            "source_gate_activity_thresholds_used": False,
            "source_noise_reference_usage": "legacy_comparison_score_only"})
    overrides = ({"on_threshold": 7.2, "off_threshold": 7.0, "settle_seconds": .2,
        "vote_windows": 5, "minimum_votes": 4, "release_windows": 5} if profile == "trial" else {})
    overrides.update({key: value for key, value in explicit.items() if value is not None})
    if overrides:
        original = {key: params.get(key) for key in ["on_threshold", "off_threshold", "settle_seconds",
            "vote_windows", "minimum_votes", "release_windows"]}
        result.update(overrides)
        if "settle_seconds" in overrides:
            fs = float(params.get("preprocess_config", {}).get("fs", 0))
            require(np.isfinite(fs) and fs > 0, "修改等待时间需要门控配置中有效的采样率。")
            result["settle_samples"] = int(round(float(result["settle_seconds"]) * fs))
        result.update({"runtime_gate_profile": profile,
            "runtime_gate_overrides": dict(overrides), "offline_detector_settings_original": original,
            "runtime_gate_settings_independently_validated": False,
            "runtime_gate_uses_preselected_class": False})
        if "on_threshold" in overrides or "off_threshold" in overrides:
            result["source_gate_activity_thresholds_used"] = False
    on, off = float(result["on_threshold"]), float(result["off_threshold"])
    require(np.isfinite([on, off]).all() and on > off >= 0, "必须满足：启动门限 > 放松门限 >= 0。")
    require(np.isfinite(result["settle_seconds"]) and result["settle_seconds"] >= 0, "稳定等待时间必须为有限非负数。")
    for key in ["on_windows", "vote_windows", "minimum_votes", "release_windows"]:
        value = result[key]
        require(isinstance(value, (int, np.integer)) and value >= 1, "门控窗口参数必须为正整数：" + key)
    require(result["vote_windows"]/2 < result["minimum_votes"] <= result["vote_windows"],
        "投票必须满足：窗口数的一半 < 最少票数 <= 窗口数。")
    return result



class RestActivityGate:
    """只观察当前/过去RMS；无类别或按钮标签输入，不拟合分类器，不自动更新已接受的静息参考。"""
    def __init__(self, fs, width, seconds=8., minimum_channels=2, on_z=6., off_z=3.):
        require(np.isfinite([fs, seconds, on_z, off_z]).all() and fs > 0 and seconds > 0,
                "静息门控时间参数无效。")
        require(isinstance(minimum_channels, int) and 1 <= minimum_channels <= 4 and on_z > off_z >= 0,
                "静息门控通道数或门限无效。")
        self.fs, self.width, self.seconds = float(fs), int(width), float(seconds)
        self.minimum_channels, self.on_z, self.off_z = minimum_channels, float(on_z), float(off_z)
        self.history, self.revision, self.change_token = [], 0, 0
        self.samples, self.first_row, self.last_row = [], None, None
        self.profile, self.error, self.state = None, "", "idle"
        self.partial_resets = 0
        self.begin("startup", 0)

    def begin(self, reason, row):
        if self.state == "collecting" and self.samples:
            self.history.append({"revision": self.revision, "status": "aborted", "reason": reason,
                "first_decision_row": self.first_row, "last_decision_row": self.last_row,
                "window_count": len(self.samples)})
        self.revision += 1
        self.reason, self.request_row = reason, int(row)
        self.state, self.profile, self.error = "collecting", None, ""
        self.samples, self.first_row, self.last_row = [], None, None
        self.partial_resets = 0
        self.change_token += 1

    def reset_partial(self):
        if self.state == "collecting" and self.samples:
            self.samples, self.first_row, self.last_row = [], None, None
            self.partial_resets += 1

    def elapsed(self):
        return 0. if self.first_row is None else min(self.seconds, (self.last_row-self.first_row+self.width)/self.fs)

    def _finish(self):
        data = np.asarray(self.samples, dtype=float)
        median = np.median(data, axis=0)
        mad = np.median(np.abs(data-median), axis=0)
        # 尺度下限避免极安静或量化信号MAD接近0时，微小变化造成巨大分数。
        scale = np.maximum.reduce([1.4826*mad, .15*median, np.full(4, .01)])
        half = len(data)//2
        first, last = np.median(data[:half], axis=0), np.median(data[half:], axis=0)
        drift = np.maximum(first, last)/np.maximum(np.minimum(first, last), 1e-6)
        spread = (np.quantile(data, .9, axis=0)-np.quantile(data, .1, axis=0))/np.maximum(median, 1e-6)
        bad = np.flatnonzero((median < .01) | (drift > 1.5) | (spread > 1.5))
        accepted = len(bad) == 0
        attempt = {"revision": self.revision, "status": "accepted" if accepted else "rejected",
            "reason": self.reason, "request_row": self.request_row,
            "first_decision_row": self.first_row, "last_decision_row": self.last_row,
            "valid_span_seconds": self.elapsed(), "window_count": len(data), "partial_resets": self.partial_resets,
            "median_rms_uv": median.tolist(), "mad_rms_uv": mad.tolist(), "robust_scale_uv": scale.tolist(),
            "on_rms_uv": (median+self.on_z*scale).tolist(), "off_rms_uv": (median+self.off_z*scale).tolist(),
            "half_to_half_median_ratio": drift.tolist(), "relative_p90_p10_spread": spread.tolist(),
            "rejected_channels_one_based": (bad+1).tolist(),
            "rest_is_user_instructed_not_independently_verified": True}
        self.history.append(attempt)
        self.profile = attempt if accepted else None
        self.state = "ready" if accepted else "failed"
        self.error = "" if accepted else "CH%s 静息波动较大或接近平直，请检查信号并放松后重新测量" % ",".join(str(x+1) for x in bad)
        self.samples = []
        self.change_token += 1

    def observe(self, rms, row, usable):
        rms = np.asarray(rms, dtype=float)
        valid = bool(usable and rms.shape == (4,) and np.isfinite(rms).all() and (rms >= 0).all())
        if not valid:
            self.reset_partial()
            return None, None
        if self.state == "collecting":
            self.first_row = int(row) if self.first_row is None else self.first_row
            self.last_row = int(row)
            self.samples.append(rms.copy())
            if self.elapsed()+1e-9 >= self.seconds and len(self.samples) >= 20:
                self._finish()
        if self.state != "ready": return None, None
        z = np.maximum(0., (rms-np.asarray(self.profile["median_rms_uv"]))/np.asarray(self.profile["robust_scale_uv"]))
        # 第k强通道决定分数；k=2时，单独一路无论多大都不能授权新输出。
        return float(np.sort(z)[-self.minimum_channels]), z

    def snapshot(self):
        return {"mode": "robust", "state": self.state, "revision": self.revision,
            "elapsed_seconds": self.elapsed(), "required_seconds": self.seconds, "error": self.error,
            "minimum_active_channels": self.minimum_channels,
            "on_threshold": self.on_z, "off_threshold": self.off_z,
            "on_rms_uv": None if self.profile is None else list(self.profile["on_rms_uv"]),
            "off_rms_uv": None if self.profile is None else list(self.profile["off_rms_uv"])}

    def report(self):
        return {"schema": "openbci_rp_binary_live_rest_gate_v1", "status": self.snapshot(), "history": self.history,
            "score_definition": "kth_largest(max(0,(RMS-median)/max(1.4826*MAD,0.15*median,0.01uV)))",
            "reference_update_policy": "startup_or_explicit_recalibrate_rest_only_then_frozen",
            "minimum_active_channels": self.minimum_channels, "on_z": self.on_z, "off_z": self.off_z,
            "stability_rules": {"maximum_half_median_ratio": 1.5, "maximum_relative_p90_p10_spread": 1.5,
                "minimum_channel_median_rms_uv": .01},
            "settings_are_starting_values_not_validated_on_user_independent_data": True,
            "classifier_updated": False}


class LiveEngine:
    """保持原窗口、特征、SVM与投票；可显式替换活动评分，不接收动作Marker或按钮标签。"""
    def __init__(self, bundle, params, names=None, max_data_age=None, activity_gate=None):
        self.bundle, self.params = bundle, params
        self.config = bundle["preprocess_config"]
        self.stream = CausalWindowStream(self.config)
        self.detector = GestureDetector(params)
        self.names = list(names or bundle["class_names"])
        self.max_data_age = max_data_age
        self.activity_gate = activity_gate
        self.last_state, self.last_activity, self.last_qc = "warmup", None, None

    def interrupt(self, reason, row=None):
        row = max(0, self.stream.position - 1) if row is None else int(row)
        self.detector.close(row, reason)
        self.detector.reset_state(); self.detector.segment_id = None
        self.stream.valid = False
        if self.activity_gate is not None: self.activity_gate.reset_partial()
        self.last_state, self.last_activity = reason, None

    def recalibrate_rest(self):
        require(self.activity_gate is not None, "请用 --activity-mode robust 启用静息测量。")
        self.interrupt("rest_calibrating")
        self.activity_gate.begin("manual_button", self.stream.position)

    def process(self, values, stamps, flags=None, emission_clock=None):
        values, stamps = np.asarray(values, dtype=float), np.asarray(stamps, dtype=float)
        require(values.ndim == 2 and values.shape[1] == 4 and stamps.shape == (len(values),), "实时输入形状错误。")
        flags = np.zeros(len(values), bool) if flags is None else np.asarray(flags, dtype=bool)
        require(flags.shape == (len(values),), "连续性标记形状错误。")
        finite = np.isfinite(values).all(axis=1)
        boundaries = np.unique(np.r_[0, len(values), np.flatnonzero(flags),
                                    np.flatnonzero(finite[1:] != finite[:-1]) + 1])
        result = []
        origin = self.stream.position
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            lo, hi = int(lo), int(hi)
            if hi <= lo: continue
            if flags[lo] or not finite[lo]:
                self.interrupt("quality_lock" if not finite[lo] else "warmup", origin + lo)
            frames, features = self.stream.push(values[lo:hi], flags[lo:hi])
            if not frames: continue
            features = np.asarray(features)
            scores, predicted = np.full((len(frames), 2), np.nan), np.full(len(frames), -1, dtype=int)
            eligible = np.array([f["usable"] for f in frames]) & np.isfinite(features).all(axis=1)
            if eligible.any():
                predicted[eligible], scores[eligible] = predict(self.bundle["pipeline"], features[eligible])
            activities = activity_scores(features, self.params["noise_reference_rms_uv"])
            for i, frame in enumerate(frames):
                stamp = float(stamps[frame["decision_row"] - origin])
                now = float(emission_clock()) if emission_clock else stamp
                timely = self.max_data_age is None or (-.25 <= now-stamp <= self.max_data_age)
                activity, z = float(activities[i]), None
                if self.activity_gate is not None:
                    activity, z = self.activity_gate.observe(features[i, [1, 6, 11, 16]], frame["decision_row"],
                        frame["usable"] and timely)
                if self.activity_gate is not None and self.activity_gate.state != "ready":
                    self.detector.close(frame["decision_row"], "rest_calibration")
                    self.detector.reset_state()
                    state, event = ("rest_calibration_failed" if self.activity_gate.state == "failed" else "rest_calibrating"), None
                    if not frame["usable"]: state = "quality_lock"
                else:
                    state, event = self.detector.step(frame["decision_row"], frame["segment_id"],
                        frame["usable"] and timely, np.nan if activity is None else activity, int(predicted[i]), scores[i])
                if not timely: state = "stale"
                if event is not None:
                    event.update({"class_name": self.names[event["class_id"]], "marker_code_for_display": [11, 13][event["class_id"]],
                        "decision_lsl_time_s": stamp, "emitted_local_clock_s": now,
                        "data_age_at_emission_s": now - stamp, "wall_time": datetime.now().isoformat(timespec="milliseconds")})
                    if self.activity_gate is not None:
                        event.update({"activity_mode": "robust", "rest_calibration_revision": self.activity_gate.revision,
                            "minimum_active_channels": self.activity_gate.minimum_channels})
                record = dict(frame)
                record.update({"decision_lsl_time_s": stamp, "processed_local_clock_s": now,
                    "data_age_s": now - stamp, "predicted_class_id": int(predicted[i]),
                    "activity_score": activity, "detector_state": state,
                    "output_suppressed_for_data_age": not timely,
                    "trigger_event_id": event["event_id"] if event else -1})
                record.update({"decision_%d" % k: scores[i, k] for k in range(2)})
                record.update(dict(zip(FEATURE_NAMES, features[i])))
                if self.activity_gate is not None:
                    record.update({"activity_mode": "robust", "legacy_activity_score": float(activities[i]),
                        "rest_calibration_revision": self.activity_gate.revision, "rest_calibration_state": self.activity_gate.state,
                        "channels_above_on": None if z is None else int(np.sum(z >= self.activity_gate.on_z)),
                        "channels_above_off": None if z is None else int(np.sum(z > self.activity_gate.off_z))})
                    record.update({"activity_z_ch%d" % (k+1): None if z is None else float(z[k]) for k in range(4)})
                result.append(record)
                self.last_state, self.last_activity, self.last_qc = state, activity, frame["qc_status"]
        return result

    def snapshot(self):
        event = self.detector.active_event
        result = {"state": self.last_state, "activity": self.last_activity, "qc": self.last_qc,
            "gesture": event["class_name"] if event else "—", "event_count": len(self.detector.events),
            "samples": self.stream.position}
        if self.activity_gate is not None: result["activity_gate"] = self.activity_gate.snapshot()
        return result


class SnapshotStore:
    """可重建的CSV/JSON快照：Windows文件占用只延后保存，不中断采样，无阻塞重试。"""
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.lock = threading.RLock()
        self.contents, self.pending, self.warned = {}, {}, {}
        self.recoveries, self.recovery_hashes = {}, {}

    def _warn(self, name, error):
        now = time.monotonic()
        if now-self.warned.get(name, -float("inf")) < 15.: return
        self.warned[name] = now
        message = "保存暂缓：%s 被占用或无替换权限；识别继续，稍后重试。请关闭正在打开该文件的程序。%s" % (name, error)
        try: log(message, self.output)
        except OSError: print(message, flush=True)

    def text(self, name, contents):
        with self.lock:
            self.contents[name] = contents
            path = self.output/name
            temporary = path.with_suffix(".tmp"+path.suffix)
            try:
                with temporary.open("w", encoding="utf-8", newline="") as stream: stream.write(contents)
                temporary.replace(path)
            except PermissionError as exc:
                # 临时文件若已完成写入就保留；内存也保留最新版本，不循环sleep或丢弃识别状态。
                self.pending[name] = str(exc)
                self._warn(name, exc)
                return False
            self.pending.pop(name, None)
            return True

    def csv(self, name, fields, rows):
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
        return self.text(name, "\ufeff"+buffer.getvalue())

    def json(self, name, value):
        return self.text(name, json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False))

    def status(self):
        with self.lock:
            unrecovered = [name for name in self.pending if self.recovery_hashes.get(name) !=
                hashlib.sha256(self.contents[name].encode("utf-8")).hexdigest()]
            return {"primary_pending_files": sorted(self.pending), "unrecovered_files": sorted(unrecovered),
                "recovery_files": dict(self.recoveries), "errors": dict(self.pending)}

    def retry_pending(self):
        with self.lock:
            for name in list(self.pending): self.text(name, self.contents[name])
            return self.status()

    def recover_pending(self):
        """停止/关窗时为仍被占用的快照另存唯一恢复文件；不覆盖用户正在查看的文件。"""
        with self.lock:
            for name in list(self.pending):
                contents = self.contents[name]
                digest = hashlib.sha256(contents.encode("utf-8")).hexdigest()
                if self.recovery_hashes.get(name) == digest: continue
                path = self.output/name
                target = path.with_name(path.stem+"_recovery_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f")+path.suffix)
                try:
                    with target.open("x", encoding="utf-8", newline="") as stream: stream.write(contents)
                except OSError as exc:
                    self._warn(name, exc)
                    continue
                self.recoveries[name], self.recovery_hashes[name] = str(target), digest
                try: log("原文件仍无法替换，最新内容已另存：%s" % target, self.output)
                except OSError: print("最新内容已另存：%s" % target, flush=True)
            return self.status()


class PosthocLabels:
    """识别后的人工标签。仅订阅事件，不向LiveEngine提供任何输入。"""
    FIELDS = ["record_id", "source", "event_id", "actual_class_id", "actual_class_name",
        "actual_marker_code", "predicted_class_id", "predicted_class_name", "result",
        "event_emitted_local_clock_s", "event_wall_time", "annotation_local_clock_s", "annotation_wall_time"]

    def __init__(self, names, clock=None):
        require(len(names) == 2 and len(set(names)) == 2, "两类显示名称必须互不相同。")
        self.names, self.clock = list(names), clock or time.monotonic
        self.lock = threading.RLock()
        self.events, self.records, self.labelled = {}, [], set()
        self.last_save, self.dirty = -float("inf"), True
        self.snapshots, self.session_summary = None, None

    def storage_for(self, output):
        with self.lock:
            if self.snapshots is None: self.snapshots = SnapshotStore(output)
            require(self.snapshots.output == Path(output).resolve(), "标注记录不能混入另一个运行目录。")
            return self.snapshots

    def observe(self, events):
        with self.lock:
            for event in events:
                event_id = int(event["event_id"])
                if event_id in self.events: continue
                require(int(event["class_id"]) in (0, 1), "事件类别无效。")
                self.events[event_id] = dict(event)
                self.dirty = True

    def pending(self):
        with self.lock:
            return [dict(self.events[k]) for k in sorted(self.events) if k not in self.labelled]

    def annotate(self, event_id, actual_class_id):
        with self.lock:
            require(actual_class_id in (0, 1), "实际动作类别无效。")
            require(event_id in self.events, "找不到该识别事件。")
            require(event_id not in self.labelled, "该事件已经标注，不能重复计数。")
            event = self.events[event_id]
            row = self._row(actual_class_id)
            predicted = int(event["class_id"])
            row.update({"source": "event_label", "event_id": int(event_id), "predicted_class_id": predicted,
                "predicted_class_name": self.names[predicted],
                "result": "correct" if predicted == actual_class_id else "wrong",
                "event_emitted_local_clock_s": event.get("emitted_local_clock_s"),
                "event_wall_time": event.get("wall_time", "")})
            self.records.append(row); self.labelled.add(int(event_id)); self.dirty = True
            return dict(row)

    def record_miss(self, actual_class_id):
        with self.lock:
            require(actual_class_id in (0, 1), "实际动作类别无效。")
            require(not self.pending(), "请先确认待标注事件，避免把已经输出的动作记成漏检。")
            row = self._row(actual_class_id)
            row.update({"source": "manual_miss", "event_id": None, "predicted_class_id": None,
                "predicted_class_name": None, "result": "missed", "event_emitted_local_clock_s": None,
                "event_wall_time": ""})
            self.records.append(row); self.dirty = True
            return dict(row)

    def _row(self, actual):
        return {"record_id": len(self.records)+1, "actual_class_id": int(actual),
            "actual_class_name": self.names[actual], "actual_marker_code": [11, 13][actual],
            "annotation_local_clock_s": float(self.clock()),
            "annotation_wall_time": datetime.now().isoformat(timespec="milliseconds")}

    def report(self):
        with self.lock:
            rows = [dict(row) for row in self.records]
            correct = sum(r["result"] == "correct" for r in rows)
            wrong = sum(r["result"] == "wrong" for r in rows)
            misses = sum(r["result"] == "missed" for r in rows)
            matrix = [[0]*3 for _ in range(2)]
            for row in rows:
                matrix[row["actual_class_id"]][row["predicted_class_id"] if row["predicted_class_id"] is not None else 2] += 1
            per_class = [{"class_id": k, "class_name": self.names[k], "record_count": sum(matrix[k]),
                "correct": matrix[k][k], "accuracy": matrix[k][k]/sum(matrix[k]) if sum(matrix[k]) else None}
                for k in range(2)]
            summary = {"schema": "openbci_rp_binary_posthoc_event_labels_v1", "label_source": "user_confirmation_after_prediction",
                "labels_used_for_inference": False, "detector_reset_on_label": False,
                "recorded_actions": len(rows), "correct": correct, "wrong": wrong, "manual_misses": misses,
                "record_based_accuracy": correct/len(rows) if rows else None,
                "labelled_output_events": correct+wrong,
                "labelled_event_accuracy": correct/(correct+wrong) if correct+wrong else None,
                "total_output_events": len(self.events), "unlabelled_output_events": len(self.events)-len(self.labelled),
                "per_class": per_class, "confusion_matrix_columns": self.names+["漏检"], "confusion_matrix": matrix,
                "accuracy_scope": "仅已标注事件及手动记录漏检；未标注事件不进入分母。",
                "truth_note": "依赖用户如实确认实际动作；不使用预测结果自动生成真值。漏检需手动记录。",
                "manual_miss_note": "若随后才出现该动作输出，请勿再次把同一次动作标注为新的一次。",
                "automatic_miss_rate": None, "rest_false_trigger_rate": None, "muscle_onset_latency_s": None}
            return {"summary": summary, "records": rows}

    def _update_session(self, summary, evaluation):
        summary["posthoc_evaluation"] = evaluation
        summary["accuracy"] = evaluation["record_based_accuracy"]
        summary["ground_truth_note"] = evaluation["accuracy_scope"]+evaluation["truth_note"]
        return summary

    def save(self, output, force=False):
        # 按钮回调与接收线程共用写锁，单击后立即落盘；不会用旧快照覆盖新标签。
        with self.lock:
            now = time.monotonic()
            storage = self.storage_for(output)
            if not force and ((not self.dirty and not storage.pending) or now-self.last_save < 1.): return
            storage.retry_pending()
            report = self.report()
            storage.csv("action_labels.csv", self.FIELDS, report["records"])
            summary = report["summary"]
            storage.csv("label_confusion_matrix.csv", ["actual_class"]+summary["confusion_matrix_columns"],
                [dict(zip(["actual_class"]+summary["confusion_matrix_columns"], [self.names[k]]+row))
                    for k, row in enumerate(summary["confusion_matrix"])])
            storage.json("label_accuracy_summary.json", summary)
            if self.session_summary is not None:
                self._update_session(self.session_summary, summary)
                self.session_summary["snapshot_persistence"] = storage.status()
                storage.json("session_summary.json", self.session_summary)
            self.last_save, self.dirty = now, bool(storage.pending)
            report["persistence"] = storage.status()
            return report

    def finalize(self, output, summary, events):
        with self.lock:
            self.observe(events)
            self.session_summary = summary
            report = self.save(output, force=True)
            return report


class LiveLog:
    def __init__(self, output, config, snapshots=None):
        self.output, self.files, self.writers = Path(output), {}, {}
        self.snapshots = snapshots or SnapshotStore(output)
        self.last_event_snapshot = -float("inf")
        self.counts = {"received_samples": 0, "windows": 0, "transport_issues": 0, "discarded_stale_samples": 0,
                       "qc_windows": {"PASS": 0, "WARNING": 0, "REVIEW": 0}}
        self.last_flush = time.monotonic()
        write_json(self.output / "live_configuration.json", config)
        self.open_csv("raw_samples.csv", ["received_row", "lsl_time_s", "received_local_clock_s", "original_sample_counter",
            "ch1_uv", "ch2_uv", "ch3_uv", "ch4_uv", "break_before", "timestamp_invalid", "discarded_stale"])
        self.open_csv("window_trace.csv", ["start_row", "stop_row", "decision_row", "segment_id", "qc_status", "qc_reasons", "usable",
            "decision_lsl_time_s", "processed_local_clock_s", "data_age_s", "predicted_class_id", "decision_0", "decision_1",
            "activity_score", "detector_state", "output_suppressed_for_data_age", "trigger_event_id"] + FEATURE_NAMES +
            ["activity_mode", "legacy_activity_score", "rest_calibration_revision", "rest_calibration_state",
             "channels_above_on", "channels_above_off"] + ["activity_z_ch%d" % k for k in range(1, 5)])
        self.open_csv("transport_issues.csv", ["received_row", "reason", "timestamp_dt_s", "wall_time"])
        self.open_csv("transport_health.csv", ["wall_time", "received_samples", "local_clock_s",
            "last_sample_lsl_time_s", "lsl_age_s", "receive_rate_hz_5s", "inlet_queued_samples", "stream_uid"])
        self.last_health_clock = None

    def open_csv(self, name, fields):
        stream = (self.output / name).open("w", encoding="utf-8-sig", newline="")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); self.files[name], self.writers[name] = stream, writer

    def raw(self, raw, stamps, arrival, flags, invalid, stale, counters=None):
        start = self.counts["received_samples"]
        writer = self.writers["raw_samples.csv"]
        for i, row in enumerate(raw):
            writer.writerow(dict(zip(writer.fieldnames, [start+i, stamps[i], arrival,
                None if counters is None else counters[i], *row, bool(flags[i]), bool(invalid[i]), bool(stale[i])])))
        self.counts["received_samples"] += len(raw)
        self.counts["discarded_stale_samples"] += int(np.sum(stale))

    def windows(self, records):
        self.writers["window_trace.csv"].writerows(records)
        self.counts["windows"] += len(records)
        for row in records: self.counts["qc_windows"][row["qc_status"]] += 1

    def issue(self, row, reason, dt=None):
        self.writers["transport_issues.csv"].writerow({"received_row": int(row), "reason": reason,
            "timestamp_dt_s": dt, "wall_time": datetime.now().isoformat(timespec="milliseconds")})
        self.counts["transport_issues"] += 1

    def health(self, arrival, stamp, rate, queued, uid):
        if self.last_health_clock is not None and arrival-self.last_health_clock < 1.: return
        self.writers["transport_health.csv"].writerow({"wall_time": datetime.now().isoformat(timespec="milliseconds"),
            "received_samples": self.counts["received_samples"], "local_clock_s": arrival,
            "last_sample_lsl_time_s": stamp, "lsl_age_s": arrival-stamp, "receive_rate_hz_5s": rate,
            "inlet_queued_samples": queued, "stream_uid": uid})
        self.last_health_clock = arrival

    def events(self, events, force=False):
        now = time.monotonic()
        if not force and now-self.last_event_snapshot < 1.: return
        fields = ["event_id", "decision_row", "onset_candidate_row", "class_id", "class_name", "marker_code_for_display",
            "vote_agreement", "activity_score", "release_row", "release_reason", "decision_lsl_time_s",
            "emitted_local_clock_s", "data_age_at_emission_s", "wall_time", "activity_mode",
            "rest_calibration_revision", "minimum_active_channels"]
        self.snapshots.csv("events.csv", fields, events)
        self.last_event_snapshot = now

    def flush(self, force=False):
        if force or time.monotonic() - self.last_flush >= 1:
            for stream in self.files.values(): stream.flush()
            self.last_flush = time.monotonic()

    def close(self):
        error = None
        for stream in self.files.values():
            if stream.closed: continue
            try: stream.flush()
            except OSError as exc: error = error or exc
            finally:
                try: stream.close()
                except OSError as exc: error = error or exc
        if error: raise error


def import_lsl():
    try:
        import pylsl
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("LSL组件无法加载：%s\n在bci环境运行：python -m pip install --upgrade pylsl" % exc) from exc
    return pylsl


def stream_record(info):
    return {"name": info.name(), "type": info.type(), "channels": info.channel_count(),
        "nominal_srate": info.nominal_srate(), "source_id": info.source_id(), "uid": info.uid()}


def select_stream(streams, name, expected_fs, channels, stream_type=None, source_id=None, counter_channel=None):
    matched = [s for s in streams if s.name() == name and (not stream_type or s.type() == stream_type)
               and (not source_id or s.source_id() == source_id)]
    require(len(matched) == 1, "需要恰好一个名称为%s的LSL流，实际%d个。先运行 --list-streams；重名时指定 --source-id。" % (name, len(matched)))
    selected = matched[0]
    require(abs(float(selected.nominal_srate()) - expected_fs) < 1e-6,
        "LSL采样率%sHz与模型%sHz不一致，请把板卡/GUI采样率设为模型值。" % (selected.nominal_srate(), expected_fs))
    used = list(channels) + ([] if counter_channel is None else [counter_channel])
    require(max(used) <= selected.channel_count(), "选择的通道超出LSL数据通道数。")
    return selected


def run_receiver(args, bundle, model_path, params, output, stop, publish, shared, lsl=None):
    names = list(args.class_names or bundle["class_names"])
    labels = shared.get("posthoc_labels")
    snapshots = labels.storage_for(output) if labels is not None else SnapshotStore(output)
    controls = shared.setdefault("live_controls", queue.Queue())
    options = live_gate_options(args)
    effective_params = runtime_detector_params(params, args)
    gate = (RestActivityGate(float(bundle["preprocess_config"]["fs"]), bundle["preprocess_config"]["window_samples"],
        options["rest_seconds"], options["minimum_active_channels"], options["on_z"], options["off_z"])
        if options["mode"] == "robust" else None)
    engine, journal, inlet = LiveEngine(bundle, effective_params, names, args.max_data_age, activity_gate=gate), None, None
    health, last_rms, receive_rate = ReceiveHealth(), None, None
    fs = float(bundle["preprocess_config"]["fs"])
    start_clock, last_data, last_console = time.monotonic(), None, 0.0
    detector_path = Path(args.detector).expanduser().resolve()
    model_digest, config_digest = None, None
    guard = TransportGuard(fs, args.timestamp_gap_seconds, bundle["preprocess_config"]["counter_modulus"])
    failure, interrupted = None, False
    saved_gate_token = None
    try:
        model_digest, config_digest = file_sha256(model_path), file_sha256(detector_path)
        lsl = lsl or import_lsl()
        publish({"state": "connecting", "gesture": "—", "event_count": 0, "samples": 0})
        streams = lsl.resolve_streams(wait_time=args.resolve_seconds)
        selected = select_stream(streams, args.stream_name, fs, args.channels, args.stream_type, args.source_id, args.counter_channel)
        metadata = stream_record(selected)
        inlet = open_live_inlet(lsl, selected, args.resolve_seconds)
        local_config = {"schema": LIVE_SCHEMA, "script_version": LIVE_VERSION, "replay_engine_version": "1.1.0",
            "source_engine": "original_three_class_06_automatic_gate_and_07_posthoc_ui",
            "model_path": str(model_path), "model_sha256": model_digest, "detector_path": str(Path(args.detector).resolve()),
            "detector_sha256": config_digest, "subject_id": bundle["subject_id"], "calibration_date": bundle["calibration_date"],
            "same_calendar_date_as_calibration": bundle["calibration_date"] == datetime.now().strftime("%Y-%m-%d"),
            "stream": metadata, "channels_one_based": args.channels, "input_unit": args.input_unit,
            "counter_channel_one_based": args.counter_channel, "original_board_counter_available": args.counter_channel is not None,
            "continuity_scope": "optional_original_counter_plus_lsl_timestamps_and_reception_stall",
            "board_packet_loss_fully_observable": False, "clock_processing": "LSL_proc_clocksync_only_no_dejitter",
            "timestamp_gap_seconds": args.timestamp_gap_seconds, "max_data_age_seconds": args.max_data_age,
            "stall_seconds": args.stall_seconds, "class_names": names, "class_ids": [0, 1],
            "marker_codes_for_display_only": [11, 13], "gui_output_required": "TimeSeriesRaw_unfiltered_uV",
            "preprocess_config": bundle["preprocess_config"], "frozen_detector": params,
            "activity_gate_options": options, "runtime_detector_settings": effective_params,
            "input_detector_file_modified": False,
            "activity_reference_update_policy": "explicit_rest_measurement_then_frozen" if gate is not None else "input_file_frozen",
            "classifier_updated_during_session": False, "gesture_markers_used_for_inference": False,
            "accuracy_available_without_external_ground_truth": False,
            "posthoc_label_buttons_enabled": labels is not None,
            "manual_labels_used_for_inference": False, "detector_reset_on_label": False,
            "data_age_is_not_muscle_onset_or_hardware_end_to_end_latency": True,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                         "sklearn": sklearn.__version__, "pylsl": getattr(lsl, "__version__", "unknown")}}
        journal = LiveLog(output, local_config, snapshots=snapshots)
        journal.events([])
        if gate is not None:
            snapshots.json("rest_calibration.json", gate.report())
            saved_gate_token = gate.change_token
        if labels is not None: labels.save(output, force=True)
        dropped, flush_supported = flush_live_inlet(inlet)
        local_config.update({"startup_inlet_flush_supported": flush_supported, "startup_inlet_discarded_samples": dropped,
            "flush_scope": "Python_inlet_only_not_GUI_or_board", "resubscribe_does_not_reset_board": True,
            "connection_history": [{"reason": "startup", "wall_time": datetime.now().isoformat(timespec="milliseconds"),
                "stream": metadata, "flushed_inlet_samples": dropped}]})
        snapshots.json("live_configuration.json", local_config)
        if dropped: journal.issue(0, "startup_inlet_flush_discarded_%d_samples" % dropped)
        log("启动订阅已就绪，清理Python接收缓存%d点；此操作不能清空GUI或板卡上游队列。" % dropped, output)
        log("已连接：%s；%sHz；LSL共%d通道，读取%s。" % (metadata["name"], fs, metadata["channels"], args.channels), output)
        log("保持手臂放松，等预热和释放状态完成后再做动作；每次动作后放松。", output)
        if gate is not None:
            log("稳健活动门控：预热后测%.1f秒静息，至少%d路增强才启动；测量期间不输出，完成后参考冻结。" %
                (gate.seconds, gate.minimum_channels), output)
        if args.counter_channel is None:
            log("此流未提供原始Sample Index；连续性检查依据LSL时间戳和接收中断。", output)
        if not local_config["same_calendar_date_as_calibration"]:
            log("当前使用%s校准模型，适合连接联调；本日效果需用本日数据验证。" % bundle["calibration_date"], output)
        last_data = time.monotonic()
        while not stop.is_set():
            if args.duration and time.monotonic() - start_clock >= args.duration: break
            requested_commands = set()
            while True:
                try: requested_commands.add(controls.get_nowait())
                except queue.Empty: break
            if "resubscribe" in requested_commands:
                engine.interrupt("warmup")
                publish({**engine.snapshot(), "state": "resyncing", "rms_uv": None})
                journal.events(engine.detector.events, force=True)
                old_inlet, inlet = inlet, None
                try: old_inlet.close_stream()
                except Exception: journal.issue(engine.stream.position, "old_inlet_close_warning_before_resubscribe")
                finally: old_inlet = None
                # 重新发现同名同源流并创建新inlet；不把按钮操作当作观察到真实放松。
                streams = lsl.resolve_streams(wait_time=args.resolve_seconds)
                selected = select_stream(streams, args.stream_name, fs, args.channels, args.stream_type,
                    args.source_id or metadata["source_id"], args.counter_channel)
                metadata = stream_record(selected)
                inlet = open_live_inlet(lsl, selected, args.resolve_seconds)
                dropped, supported = flush_live_inlet(inlet)
                guard = TransportGuard(fs, args.timestamp_gap_seconds, bundle["preprocess_config"]["counter_modulus"])
                health, last_rms, receive_rate = ReceiveHealth(), None, None
                local_config["connection_history"].append({"reason": "manual_resubscribe",
                    "wall_time": datetime.now().isoformat(timespec="milliseconds"), "stream": metadata,
                    "flushed_inlet_samples": dropped, "flush_supported": supported})
                local_config["stream"] = metadata
                snapshots.json("live_configuration.json", local_config)
                journal.issue(engine.stream.position, "manual_lsl_resubscribe_flushed_%d_samples" % dropped)
                log("LSL已重新订阅，清理接收缓存%d点；重新预热并观察到放松后才允许输出。" % dropped, output)
                last_data, interrupted = time.monotonic(), False
            if "recalibrate_rest" in requested_commands and gate is not None:
                engine.recalibrate_rest()
                journal.events(engine.detector.events, force=True)
                snapshots.json("rest_calibration.json", gate.report())
                saved_gate_token = gate.change_token
                journal.issue(engine.stream.position, "user_requested_rest_calibration_revision_%d" % gate.revision)
                publish({**engine.snapshot(), "rms_uv": None})
                log("已开始重新测静息：请持续放松%.1f秒；旧事件已结束，已有人工标注保留。" % gate.seconds, output)
            samples, stamps = inlet.pull_chunk(timeout=.1, max_samples=max(25, int(fs // 5)))
            tick = time.monotonic()
            if not stamps:
                if tick - last_data >= args.stall_seconds and not interrupted:
                    interrupted = True
                    engine.interrupt("no_data")
                    last_rms = None
                    journal.issue(engine.stream.position, "reception_stall")
                    journal.events(engine.detector.events)
                    publish(engine.snapshot())
                    log("数据中断，已暂停输出；恢复后重新预热。", output)
                journal.flush()
                if labels is not None: labels.save(output)
                continue
            matrix, timestamps = np.asarray(samples, dtype=float), np.asarray(stamps, dtype=float)
            require(matrix.ndim == 2 and len(matrix) == len(timestamps) and matrix.shape[1] == metadata["channels"], "LSL数据形状异常。")
            arrival = float(lsl.local_clock())
            flags, invalid, issues = guard.inspect(timestamps, None if args.counter_channel is None else matrix[:, args.counter_channel-1])
            if interrupted: flags[0] = True
            interrupted = False
            raw = matrix[:, np.asarray(args.channels)-1] * {"uv": 1., "mv": 1000., "v": 1000000.}[args.input_unit]
            counters = None if args.counter_channel is None else matrix[:, args.counter_channel-1]
            stale = np.isfinite(timestamps) & ((arrival - timestamps > args.max_data_age) | (timestamps - arrival > .25))
            for item in issues:
                journal.issue(engine.stream.position + item["offset"], item["reason"], item["timestamp_dt_s"])
            if stale.any():
                journal.issue(engine.stream.position + int(np.flatnonzero(stale)[0]), "stale_or_future_timestamp_samples_suppressed")
            journal.raw(raw, timestamps, arrival, flags, invalid, stale, counters)
            clean = raw.copy(); clean[invalid | stale] = np.nan
            records = engine.process(clean, timestamps, flags, lsl.local_clock)
            if gate is not None and saved_gate_token != gate.change_token:
                snapshots.json("rest_calibration.json", gate.report())
                saved_gate_token = gate.change_token
                if gate.state == "ready":
                    log("静息测量完成，参考已冻结；各路释放/启动门限(μV)：%s。仍需观察到持续放松后才允许输出。" %
                        "；".join("CH%d %.2f/%.2f" % (k+1, gate.profile["off_rms_uv"][k], gate.profile["on_rms_uv"][k])
                            for k in range(4)), output)
                elif gate.state == "failed": log("静息测量未通过：%s。请放松后点击“重新测静息”。" % gate.error, output)
            if records: last_rms = [float(records[-1]["ch%d_RMS" % k]) for k in range(1, 5)]
            receive_rate = health.update(len(raw), arrival)
            queued = inlet_queue_size(inlet)
            journal.health(arrival, float(timestamps[-1]), receive_rate, queued, metadata["uid"])
            if labels is not None:
                labels.observe(engine.detector.events)
                labels.save(output)
            if stale[-1]: engine.last_state = "stale"
            journal.windows(records); journal.events(engine.detector.events); journal.flush()
            last_data = tick
            snapshot = engine.snapshot(); snapshot["data_age"] = float(arrival - timestamps[-1])
            snapshot["output_directory"] = str(output)
            snapshot["pending_snapshot_files"] = snapshots.status()["primary_pending_files"]
            snapshot.update({"rms_uv": last_rms, "receive_rate_hz_5s": receive_rate,
                "inlet_queued_samples": queued, "last_sample_lsl_time_s": float(timestamps[-1]),
                "published_local_clock_s": float(lsl.local_clock()), "stream_uid": metadata["uid"]})
            publish(snapshot)
            emitted = [r for r in records if r["trigger_event_id"] >= 0]
            for record in emitted:
                event = engine.detector.events[record["trigger_event_id"]-1]
                log("识别：%s（显示Marker %d）；稳定票%d/%d；事件%d。" %
                    (event["class_name"], event["marker_code_for_display"], round(event["vote_agreement"]*params["vote_windows"]),
                     params["vote_windows"], event["event_id"]), output)
            if args.no_ui and tick - last_console >= 2:
                log("%s；活动=%s；累计输出%d次。" % (STATE_NAMES.get(snapshot["state"], snapshot["state"]),
                    "—" if snapshot["activity"] is None else "%.2f" % snapshot["activity"], snapshot["event_count"]), output)
                last_console = tick
    except Exception as exc:
        failure = "%s: %s" % (type(exc).__name__, exc)
        shared["exit_code"] = 1
        log("实时接收停止：" + failure, output)
        publish({"state": "error", "message": failure, "gesture": "—"})
    finally:
        try:
            engine.interrupt("stopped")
            if inlet is not None:
                try: inlet.close_stream()
                except Exception: pass
                finally: inlet = None
            try: unchanged = bool(model_digest and config_digest and file_sha256(model_path) == model_digest and file_sha256(detector_path) == config_digest)
            except OSError: unchanged = False
            if not unchanged:
                failure = (failure + "; " if failure else "") + "模型或门控文件在会话中发生改变"
                shared["exit_code"] = 1
            summary = {"schema": LIVE_SCHEMA, "script_version": LIVE_VERSION, "status": "error" if failure else "completed",
                "error": failure, "output_directory": str(output), "elapsed_wall_seconds": time.monotonic()-start_clock,
                "event_count": len(engine.detector.events), "accuracy": None, "rest_false_trigger_rate": None,
                "ground_truth_note": "实时输入没有动作和真实静息标签，不自动计算准确率或静息误触发率。",
                "model_and_detector_files_unchanged": unchanged, "counters": journal.counts if journal else {}}
            summary["lsl_connection_count"] = len(local_config.get("connection_history", [])) if journal is not None else 0
            summary["activity_mode"] = options["mode"]
            if gate is not None:
                snapshots.json("rest_calibration.json", gate.report())
                summary["rest_calibration_status"] = gate.snapshot()
            if journal:
                try: journal.events(engine.detector.events, force=True); journal.flush(True)
                finally: journal.close()
            if labels is not None:
                report = labels.finalize(output, summary, engine.detector.events)
                stats = report["summary"]
                log("人工标注：已记录%d次，正确%d，错误%d，手动漏检%d，尚未标注事件%d。" %
                    (stats["recorded_actions"], stats["correct"], stats["wrong"], stats["manual_misses"],
                     stats["unlabelled_output_events"]), output)
            snapshots.recover_pending()
            summary["snapshot_persistence"] = snapshots.status()
            snapshots.json("session_summary.json", summary)
            persistence = snapshots.recover_pending()
            shared["snapshot_persistence"] = persistence
            if persistence["unrecovered_files"]:
                log("部分快照尚未保存成功，请保留窗口并解除文件占用：%s" % ", ".join(persistence["unrecovered_files"]), output)
            elif persistence["primary_pending_files"]:
                log("主快照暂未更新；最新内容已写入恢复文件，详见session_summary.json及上方路径。", output)
            else:
                log("日志已保存：%s" % output, output)
        except Exception as exc:
            shared["exit_code"], shared["save_error"] = 1, "%s: %s" % (type(exc).__name__, exc)
            print("[保存未完成] %s；请保留当前输出目录。" % shared["save_error"], flush=True)
            if journal:
                try: journal.close()
                except OSError: pass
        finally:
            shared["finished"] = True
            stop.set()


def show_window(args, bundle, params, output, stop, updates, shared, worker):
    try:
        import tkinter as tk
        from tkinter import ttk
        root = tk.Tk()
    except Exception as exc:
        raise RuntimeError("无法打开显示窗口：%s。可添加 --no-ui 在终端显示。" % exc) from exc
    names = list(args.class_names or bundle["class_names"])
    labels = shared.get("posthoc_labels")
    if labels is None:
        labels = PosthocLabels(names, clock=import_lsl().local_clock)
        shared["posthoc_labels"] = labels
    controls = shared.setdefault("live_controls", queue.Queue())
    options = live_gate_options(args)
    display_params = runtime_detector_params(params, args)
    gate_label = ("截图临时试验" if args.gate_profile == "trial" else ("自定义" if display_params.get("runtime_gate_overrides") else ("当前静息" if options["mode"] == "robust" else "门控文件原值")))
    root.title("OpenBCI 实时识别 · %s" % LIVE_VERSION)
    root.geometry("940x850"); root.minsize(850, 780)
    root.configure(bg="#f2f5fa")
    body = ttk.Frame(root, padding=14); body.pack(fill="both", expand=True)
    ttk.Label(body, text="四通道 sEMG · 石头/布识别（旧流程对照版）", font=("Microsoft YaHei UI", 18, "bold")).pack(anchor="w")
    ttk.Label(body, text="%s  |  校准日期 %s  |  %s Hz  |  程序 %s" %
        (bundle["subject_id"], bundle["calibration_date"], bundle["preprocess_config"]["fs"], LIVE_VERSION)).pack(anchor="w", pady=(5, 8))
    gesture, state, detail = tk.StringVar(value="—"), tk.StringVar(value="正在查找LSL数据流"), tk.StringVar(value="请先启动GUI中的原始时间序列LSL输出")
    ttk.Label(body, text="本次锁定的识别结果（放松后释放）", foreground="#56647a").pack()
    ttk.Label(body, textvariable=gesture, font=("Microsoft YaHei UI", 30, "bold"), foreground="#164ab6").pack(pady=4)
    ttk.Label(body, textvariable=state, font=("Microsoft YaHei UI", 13)).pack()
    ttk.Label(body, textvariable=detail, wraplength=840).pack(pady=5)
    caption = ("稳健活动分数 · 至少%d/4路持续增强才启动（50%%位置为启动门限；不是识别概率）" %
        options["minimum_active_channels"] if options["mode"] == "robust" else
        "肌电活动条 · 相对校准噪声的分数（不是识别概率；50%位置为启动门限）")
    ttk.Label(body, text=caption, foreground="#56647a", wraplength=840).pack(anchor="w")
    progress = ttk.Progressbar(body, length=560, mode="determinate", maximum=max(1., display_params["on_threshold"]*2))
    progress.pack(fill="x", pady=6)
    activity_text = tk.StringVar(value="等待有效活动数据")
    rms_text = tk.StringVar(value="分类窗口四通道RMS：—")
    gate_text = tk.StringVar(value="启动后请持续放松，等待静息测量完成" if options["mode"] == "robust" else "使用离线校准的活动参考")
    ttk.Label(body, textvariable=activity_text, wraplength=840).pack(anchor="w")
    ttk.Label(body, textvariable=rms_text, wraplength=840, foreground="#56647a").pack(anchor="w", pady=(2, 4))
    ttk.Label(body, textvariable=gate_text, wraplength=840, foreground="#56647a").pack(anchor="w", pady=(0, 4))
    ttk.Label(body, text="本次门控：%s | 稳定等待%.2f秒 | %d窗至少%d票 | 放松确认%d窗" % (gate_label, display_params["settle_seconds"], display_params["vote_windows"], display_params["minimum_votes"], display_params["release_windows"]), wraplength=840, foreground="#56647a").pack(anchor="w", pady=(0, 4))
    panel = ttk.LabelFrame(body, text="识别结束后：选择你刚才实际做的动作", padding=12)
    panel.pack(fill="x", pady=6)
    pending_text = tk.StringVar(value="等待第一次识别输出……")
    ttk.Label(panel, textvariable=pending_text, font=("Microsoft YaHei UI", 13, "bold"), wraplength=740).pack(anchor="w", pady=4)
    row = ttk.Frame(panel); row.pack(fill="x", pady=7)
    target, last_click = [None], [-float("inf")]
    feedback = tk.StringVar(value="做一次手势 → 放松 → 点实际动作保存；标注完成后再做下一次。")
    label_buttons = []

    def save_result(actual_id, event_id=None, missed=False):
        if time.monotonic()-last_click[0] < .4: return False
        try:
            record = labels.record_miss(actual_id) if missed else labels.annotate(event_id, actual_id)
            saved = labels.save(output, force=True)
            last_click[0] = time.monotonic()
            result = {"correct":"正确", "wrong":"错误", "missed":"漏检"}[record["result"]]
            pending = saved["persistence"]["primary_pending_files"]
            feedback.set("%s：%s实际%s，识别%s，%s。" %
                ("已记录，部分文件保存暂缓" if pending else "已保存", "事件%d · " % event_id if event_id is not None else "",
                 names[actual_id], record["predicted_class_name"] or "无输出", result))
            refresh_labels()
            return True
        except Exception as exc:
            feedback.set("保存未完成：%s。标注记录会保留，请检查输出目录。" % exc)
            return False

    for k, name in enumerate(names):
        row.columnconfigure(k, weight=1)
        button = ttk.Button(row, text="实际是%s" % name, command=lambda k=k: save_result(k, target[0]))
        button.grid(row=0, column=k, sticky="ew", padx=4)
        label_buttons.append(button)

    def open_miss_dialog():
        if labels.pending():
            feedback.set("请先标注待确认的识别事件，避免重复记录同一次动作。")
            return
        dialog = tk.Toplevel(root)
        dialog.title("记录一次漏检")
        dialog.transient(root); dialog.grab_set()
        inner = ttk.Frame(dialog, padding=18); inner.pack(fill="both", expand=True)
        ttk.Label(inner, text="刚才确实做了动作，但没有新的识别输出。\n请选择实际动作；这次会计入漏检。",
            wraplength=440).pack(pady=(0, 10))
        for k, name in enumerate(names):
            def choose(k=k):
                if save_result(k, missed=True): dialog.destroy()
            ttk.Button(inner, text=name, command=choose).pack(side="left", padx=6)
    miss_button = ttk.Button(panel, text="本次没有输出：记录一次漏检", command=open_miss_dialog)
    miss_button.pack(anchor="w", pady=5)
    ttk.Label(panel, textvariable=feedback, wraplength=740, foreground="#1655a0").pack(anchor="w", pady=5)
    stats_panel = ttk.LabelFrame(body, text="已标注记录的统计", padding=10)
    stats_panel.pack(fill="x", pady=6)
    accuracy, counts, classes = tk.StringVar(), tk.StringVar(), tk.StringVar()
    ttk.Label(stats_panel, textvariable=accuracy, font=("Microsoft YaHei UI", 17, "bold")).pack(anchor="w")
    ttk.Label(stats_panel, textvariable=counts, wraplength=740).pack(anchor="w", pady=4)
    ttk.Label(stats_panel, textvariable=classes, wraplength=740).pack(anchor="w", pady=2)
    ttk.Label(body, text="准确率=正确÷（正确+错误+手动记录漏检）。未标注事件不计入；没有输出的动作需手动记录。",
        wraplength=760, foreground="#56647a").pack(anchor="w", pady=4)
    ttk.Label(body, text="每次动作后放松；标注仅用于统计，不会改变识别结果。", wraplength=760).pack(anchor="w", pady=3)
    ttk.Label(body, text="输出目录：" + str(output), wraplength=760, foreground="#56647a").pack(anchor="w", pady=3)
    storage_notice = tk.StringVar(value="")
    ttk.Label(body, textvariable=storage_notice, wraplength=760, foreground="#ad5c00").pack(anchor="w", pady=2)
    footer = ttk.Frame(body); footer.pack(fill="x", side="bottom", pady=5)
    def request_resubscribe():
        if not stop.is_set() and controls.empty():
            controls.put_nowait("resubscribe")
            resubscribe_button.configure(state="disabled")
            state.set("已请求重新订阅LSL；将重新预热并等待放松")
    resubscribe_button = ttk.Button(footer, text="重新同步LSL（板卡保持连接）", command=request_resubscribe)
    resubscribe_button.pack(side="left")
    def request_rest_calibration():
        if options["mode"] == "robust" and not stop.is_set():
            controls.put_nowait("recalibrate_rest")
            rest_button.configure(state="disabled")
            state.set("已请求重新测静息；请保持手臂放松")
    rest_button = ttk.Button(footer, text="重新测静息（请先放松）", command=request_rest_calibration,
        state="normal" if options["mode"] == "robust" else "disabled")
    rest_button.pack(side="left", padx=6)
    stop_button = ttk.Button(footer, text="停止接收并保存", command=stop.set)
    stop_button.pack(side="right")
    root.protocol("WM_DELETE_WINDOW", stop.set)

    def refresh_labels():
        pending = labels.pending()
        target[0] = pending[0]["event_id"] if pending else None
        if pending:
            current = pending[0]
            pending_text.set("待标注事件 #%d：识别为【%s】 · 共%d条待标注" %
                (current["event_id"], names[int(current["class_id"])], len(pending)))
        else: pending_text.set("暂无待标注事件。做完一次手势、出现输出后，再点实际动作。")
        enabled = bool(pending) and time.monotonic()-last_click[0] >= .4
        for button in label_buttons: button.configure(state="normal" if enabled else "disabled")
        miss_button.configure(state="disabled" if pending else "normal")
        summary = labels.report()["summary"]
        value = summary["record_based_accuracy"]
        accuracy.set("已标注动作正确率：%s" % ("—" if value is None else "%.2f%%（%d/%d）" %
            (value*100, summary["correct"], summary["recorded_actions"])))
        counts.set("已记录%d次  |  正确%d  |  错误%d  |  手动漏检%d  |  待标注%d" %
            (summary["recorded_actions"], summary["correct"], summary["wrong"], summary["manual_misses"], summary["unlabelled_output_events"]))
        classes.set("  |  ".join("%s %d/%d" % (r["class_name"], r["correct"], r["record_count"]) for r in summary["per_class"]))

    last_snapshot = {}
    def poll():
        nonlocal last_snapshot
        latest = None
        while True:
            try: latest = updates.get_nowait()
            except queue.Empty: break
        if latest is not None: last_snapshot = latest
        if last_snapshot:
            current = last_snapshot
            status = current.get("state", "connecting")
            stamp = current.get("last_sample_lsl_time_s")
            age = float(labels.clock())-float(stamp) if stamp is not None else None
            display_stale = age is not None and not (-.25 <= age <= args.max_data_age)
            state.set("接收画面已过时，等待新数据；可尝试重新同步LSL" if display_stale and not stop.is_set() else
                STATE_NAMES.get(status, current.get("message", status)))
            gesture.set("—" if display_stale else current.get("gesture", "—"))
            indicator = activity_indicator(None if display_stale else current.get("activity"), display_params, status)
            progress["maximum"] = indicator["maximum"]
            progress["value"] = indicator["value"]
            activity_text.set(indicator["text"])
            gate_status = current.get("activity_gate")
            if gate_status is not None and not display_stale:
                if gate_status["state"] == "collecting":
                    activity_text.set("静息测量 %.1f / %.1f 秒：保持放松，暂不输出手势" %
                        (gate_status["elapsed_seconds"], gate_status["required_seconds"]))
                    gate_text.set("前臂保持支撑，手自然放松；测量结束后仍需观察到持续放松")
                elif gate_status["state"] == "failed":
                    activity_text.set("静息测量未通过，识别暂停")
                    gate_text.set(gate_status["error"])
                else:
                    gate_text.set("各路释放/启动门限（μV）："+" | ".join("CH%d %.2f/%.2f" %
                        (k+1, gate_status["off_rms_uv"][k], gate_status["on_rms_uv"][k]) for k in range(4)))
            elif gate_status is not None and display_stale:
                gate_text.set("接收画面已过时，等待新数据")
            rms = current.get("rms_uv")
            rms_text.set("分类窗口RMS（μV）："+("—" if rms is None or display_stale else
                "  |  ".join("CH%d %.2f" % (k+1, value) for k, value in enumerate(rms))))
            if status == "error": detail.set(current.get("message", "运行失败；请查看终端"))
            else:
                rate, queued = current.get("receive_rate_hz_5s"), current.get("inlet_queued_samples")
                detail.set("输出%d次 | 接收%d点 | QC %s | LSL年龄%s | 5秒接收率%s | 缓存%s点" %
                    (current.get("event_count", 0), current.get("samples", 0), current.get("qc") or "—",
                     "—" if age is None else "%.0f ms" % (age*1000),
                     "—" if rate is None else "%.1f Hz" % rate, "—" if queued is None else queued))
            resubscribe_button.configure(state="disabled" if stop.is_set() or not controls.empty() or
                status in {"connecting", "resyncing"} else "normal")
            rest_button.configure(state="disabled" if options["mode"] != "robust" or stop.is_set() or
                not controls.empty() or status in {"connecting", "resyncing"} else "normal")
        refresh_labels()
        persistence = labels.storage_for(output).status()
        pending_files = persistence["primary_pending_files"]
        if pending_files:
            storage_notice.set("保存暂缓：%s。关闭占用文件的程序后重试；停止时会尝试另存恢复文件。" % ", ".join(pending_files))
        else: storage_notice.set("")
        if shared.get("finished"):
            state.set("接收已停止；请查看终端报错" if shared.get("exit_code", 0) else
                "接收已停止；部分文件有占用，见保存提示" if pending_files else "接收已停止；记录已保存，可继续标注待确认事件")
            if pending_files:
                labels.save(output, force=False)
            stop_button.configure(text="关闭", command=root.destroy)
            root.protocol("WM_DELETE_WINDOW", root.destroy)
        root.after(100, poll)
    refresh_labels()
    worker.start(); root.after(100, poll)
    try: root.mainloop()
    except KeyboardInterrupt: stop.set(); root.destroy()
    finally:
        stop.set(); worker.join(timeout=6)
        labels.save(output, force=True)
        storage = labels.storage_for(output)
        storage.recover_pending()
        if labels.session_summary is not None:
            labels.session_summary["snapshot_persistence"] = storage.status()
            storage.json("session_summary.json", labels.session_summary)
            storage.recover_pending()


def resolve_run_inputs(args):
    """Resolve paths only. Never fit, prepare, rewrite or relabel input files."""
    if not args.run:
        return
    require(not args.model and not args.detector, "--run与--model/--detector请选择一种提供方式，避免混用模型。")
    require(not args.prepare_only, "--run用于现成模型的实时检测；准备门控请显式提供--model。")
    directory = Path(args.run).expanduser().resolve()
    state = read_json(directory / "pipeline_state.json")
    require(state.get("schema") == "openbci_rp_binary_pipeline_v1", "不是石头/布二分类流程目录。")
    for key, attr in [("daily_model", "model"), ("detector", "detector")]:
        value = state.get(key)
        require(isinstance(value, str) and value, "流程状态缺少" + key)
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = directory / path
        path = path.resolve()
        require(path.is_file(), "找不到文件：" + str(path))
        setattr(args, attr, str(path))


def print_input_check(args, bundle, model_path, params):
    runtime = runtime_detector_params(params, args)
    config = bundle["preprocess_config"]
    result = {"script_version": LIVE_VERSION, "check_only": True,
        "source_engine": "original_three_class_06_automatic_gate_and_07_posthoc_ui",
        "model": str(model_path), "model_sha256": file_sha256(model_path),
        "detector": str(Path(args.detector).expanduser().resolve()),
        "class_names": list(args.class_names), "class_ids": [0, 1],
        "marker_codes_for_display_only": [11, 13],
        "model_calibration_date": bundle["calibration_date"], "activity_mode": args.activity_mode,
        "fs": config["fs"], "window_samples": config["window_samples"],
        "hop_samples": config["hop_samples"],
        "thresholds": {key: runtime[key] for key in ["on_threshold", "off_threshold", "settle_seconds",
            "vote_windows", "minimum_votes", "release_windows"]},
        "labels_used_for_inference": False, "classifier_updated": False,
        "runtime_gate_profile": runtime.get("runtime_gate_profile", "source"),
        "runtime_gate_overrides": runtime.get("runtime_gate_overrides", {})}
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2, allow_nan=False))


def build_live_parser():
    p = argparse.ArgumentParser(description="OpenBCI四通道SVM实时识别；识别后点击实际动作保存标签与正确率。Python>=3.9。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run", help="现有二分类pipeline_state.json所在目录；与--model/--detector二选一")
    p.add_argument("--check-only", action="store_true", help="只检查现有模型和门控，不打开界面或连接LSL")
    p.add_argument("--model", help="03生成的daily_model.joblib")
    p.add_argument("--detector", help="匹配该模型的replay_detector_config.json（05 v1.1或本脚本prepare-only生成）")
    p.add_argument("--prepare-only", action="store_true", help="只从模型的源Run1+2准备门控，不连接设备、不使用测试数据")
    p.add_argument("--source-data", help="prepare-only时的校准预处理目录；默认读取模型旁calibration_config.json")
    p.add_argument("--raw-mode", choices=["auto", "stored", "txt"], default="auto")
    p.add_argument("--settle-candidates-ms", nargs="+", type=float, default=[0, 200, 400, 600, 800], help="仅准备门控时用源Run2选择")
    p.add_argument("--list-streams", action="store_true", help="只列出LSL流，不需要模型和门控")
    p.add_argument("--stream-name", default="rps_raw", help="与GUI Networking中填写的LSL Name一致")
    p.add_argument("--stream-type", help="可选，限制LSL Type；GUI可填EMG")
    p.add_argument("--source-id", help="有重名流时指定精确source_id")
    p.add_argument("--channels", nargs=4, type=int, default=[1, 2, 3, 4], help="LSL中四个肌电通道，从1开始计数")
    p.add_argument("--counter-channel", type=int, help="可选，仅当LSL确实另外包含原始Sample Index时指定；GUI TimeSeriesRaw通常不包含")
    p.add_argument("--input-unit", choices=["uv", "mv", "v"], default="uv", help="OpenBCI GUI原始时间序列默认微伏")
    p.add_argument("--class-names", nargs=2, default=["石头", "布"], help="按Marker11、13顺序给按钮和识别显示命名；本项目已确认默认映射")
    p.add_argument("--activity-mode", choices=["legacy", "robust"], default="legacy",
        help="legacy沿用离线门控；robust启用启动静息测量、逐通道稳健门限和多通道确认（新方案需独立验证）")
    p.add_argument("--gate-profile", choices=["source", "trial"], default="source",
        help="source沿用门控文件；trial为截图临时试验：ON7.2/OFF7.0，等待0.2秒，5窗4票、5窗放松")
    p.add_argument("--gate-on", type=float, help="覆盖本次legacy活动启动门限；必须大于放松门限")
    p.add_argument("--gate-off", type=float, help="覆盖本次legacy活动放松门限；必须小于启动门限")
    p.add_argument("--gate-settle-seconds", type=float, help="覆盖本次动作稳定等待秒数，非负")
    p.add_argument("--gate-vote-windows", type=int, help="覆盖本次类别投票窗口数")
    p.add_argument("--gate-min-votes", type=int, help="覆盖本次最少同类票数，须严格超过窗口数一半")
    p.add_argument("--gate-release-windows", type=int, help="覆盖本次连续低活动释放窗口数")
    p.add_argument("--rest-seconds", type=float, default=8., help="robust模式预热后连续测量静息的秒数，范围5~30")
    p.add_argument("--min-active-channels", type=int, default=2, help="robust模式需要增强的通道数，1~4；默认2防止单路极值触发")
    p.add_argument("--activity-on-z", type=float, default=6., help="robust模式启动门限：偏离静息中位数的稳健尺度倍数")
    p.add_argument("--activity-off-z", type=float, default=3., help="robust模式释放门限，必须小于启动门限")
    p.add_argument("--no-ui", action="store_true", help="只用终端显示")
    p.add_argument("--duration", type=float, default=0, help="接收秒数；0表示直到关闭窗口或Ctrl+C")
    p.add_argument("--resolve-seconds", type=float, default=3, help="LSL发现和连接超时秒数")
    p.add_argument("--timestamp-gap-seconds", type=float, default=.2, help="LSL时间戳超过此间隔则重置；不把一般GUI批量时间抖动当作确定丢包")
    p.add_argument("--max-data-age", type=float, default=.75, help="积压超过此秒数的样本留日志但不用于输出动作")
    p.add_argument("--stall-seconds", type=float, default=1, help="连续收不到数据超过此秒数时暂停并重置")
    p.add_argument("--output", default="RP_BINARY_CLASSIC_LIVE_LABEL", help="输出父目录，每次自动新建子目录")
    return p


def main():
    args = build_live_parser().parse_args()
    try:
        require(not (args.check_only and args.prepare_only), "--check-only和--prepare-only不能同时使用。")
        if not args.list_streams: resolve_run_inputs(args)
        require(len(set(args.channels)) == 4 and min(args.channels) >= 1, "四个输入通道必须为互不相同的正整数。")
        require(args.counter_channel is None or (args.counter_channel > 0 and args.counter_channel not in args.channels), "计数器通道必须独立于四个肌电通道。")
        require(all(np.isfinite(x) and x > 0 for x in [args.resolve_seconds, args.timestamp_gap_seconds,
                args.max_data_age, args.stall_seconds]) and np.isfinite(args.duration) and args.duration >= 0, "时间参数无效。")
        require(len(set(args.class_names)) == 2 and not set(args.class_names) & {"actual_class", "漏检"},
                "两类名称必须互不相同，且不能使用保留名称actual_class或漏检。")
        require(np.isfinite(args.rest_seconds) and 5 <= args.rest_seconds <= 30 and
                1 <= args.min_active_channels <= 4 and np.isfinite([args.activity_on_z, args.activity_off_z]).all()
                and args.activity_on_z > args.activity_off_z >= 0, "静息测量须为5~30秒，通道数须为1~4，启动门限须大于释放门限且非负。")
        if args.list_streams:
            lsl = import_lsl()
            streams = lsl.resolve_streams(wait_time=args.resolve_seconds)
            for stream in streams: print(json.dumps(stream_record(stream), ensure_ascii=False))
            if not streams: print("没有找到LSL流：在GUI的Networking选择LSL、TimeSeriesRaw并启动输出。")
            return 0
        require(args.model, "请提供 --model；查看流时可单独使用 --list-streams。")
        if args.prepare_only:
            require(args.gate_profile == "source" and all(getattr(args, key) is None for key in ["gate_on", "gate_off", "gate_settle_seconds", "gate_vote_windows", "gate_min_votes", "gate_release_windows"]), "实时覆盖参数不用于prepare-only；请准备原门控后在实时命令中指定。")
            require(not args.detector, "prepare-only会生成门控，不需要 --detector。")
            prepare_detector(args); return 0
        require(args.detector, "请提供 --detector，或先用 --prepare-only 准备门控。")
        bundle, model_path = load_daily(args.model)
        params = read_json(Path(args.detector).expanduser())
        validate_detector(bundle, model_path, params)
        runtime_detector_params(params, args)  # Validate before creating a window/thread/output.
        if args.check_only:
            print_input_check(args, bundle, model_path, params)
            return 0
        output = make_output(args.output, "RP_BINARY_CLASSIC_LIVE_LABEL", bundle["subject_id"])
        stop, updates, shared = threading.Event(), queue.Queue(maxsize=1), {"exit_code": 0, "finished": False}
        def publish(value):
            try: updates.get_nowait()
            except queue.Empty: pass
            try: updates.put_nowait(value)
            except queue.Full: pass
        worker = threading.Thread(target=run_receiver, args=(args, bundle, model_path, params, output, stop, publish, shared), daemon=True)
        if args.no_ui:
            worker.start()
            try:
                while worker.is_alive(): worker.join(timeout=.2)
            except KeyboardInterrupt: stop.set(); worker.join(timeout=6)
        else:
            show_window(args, bundle, params, output, stop, updates, shared, worker)
        return int(shared.get("exit_code", 0))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print("[错误] %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
