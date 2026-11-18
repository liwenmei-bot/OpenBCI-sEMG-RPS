#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenBCI 4CH 石头/布：TKE/RMS三组门控对照，先动作后标注。

Python >=3.9。安装：python -m pip install numpy scipy scikit-learn joblib pylsl
单文件完整程序；读取现有二分类 daily_model.joblib 与匹配的门控，不拟合SVM。
示例：python 08_live_rp_tke_compare_4ch.py --run YOUR_PIPELINE_DIRECTORY
自检：python 08_live_rp_tke_compare_4ch.py --self-test
仅核查模型：追加 --check-only；离线审计：--replay LOG_DIRECTORY。

流程：静息30秒 -> 自由做石头/布各至少3次、每次4秒后标注 -> 冻结门限
      -> 自由动作自动识别；可记录4秒测试窗口并在结束后标注 -> 静息测试60秒。
开始测试只记录评估边界，不启动、不重置门控，也不预选类别。
标签不会传入SVM或正式测试检测器。未输出的测试也可标注并计入漏检。
A=原RMS参考/门限，B=当天RMS，C=当天TKE；三者共享窗口、特征、SVM和投票规则。
TKE为过去/当前三点因果计算：abs(x[t-1]**2-x[t-2]*x[t])，保留跨包状态。
它相对中心采样点晚一采样点，在250Hz为4ms，另有窗口/投票/传输延迟。
本程序是工程对照试验，不保证TKE优于RMS；按钮到输出时间不是肌肉起始延迟。
论文参考：https://doi.org/10.3389/fnbot.2021.711047
核心滤波、特征和事件状态机沿用原项目的classic流程。
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import html
import io
import json
import math
import queue
import sys
import threading
import time
import traceback
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import joblib
    import numpy as np
    import scipy
    import sklearn
    from scipy.signal import sosfilt, sosfilt_zi
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
except ImportError as exc:
    raise SystemExit('缺少依赖：%s\n请运行：python -m pip install numpy scipy scikit-learn joblib pylsl' % exc)

APP_VERSION = '1.0.0-tke-rms-compare'
APP_SCHEMA = 'openbci_rp_tke_compare_v1'
ARM_NAMES = {'A': '原RMS', 'B': '当天RMS', 'C': '当天TKE'}
NAMES = ['石头', '布']
SCHEMA = "openbci_rp_binary_4ch_svm_continuous_replay_v1"

DAILY_SCHEMA = "openbci_rp_binary_4ch_svm_daily_v1"

FEATURE_NAMES = ["ch%d_%s" % (ch, f) for ch in range(1, 5) for f in ("MAV", "RMS", "WL", "ZC", "SSC")]

CONTRACT_KEYS = ["schema", "fs", "n_channels", "channel_columns", "processing_unit", "filter_mode",
    "low_hz", "high_hz", "butter_prototype_order", "notch_hz", "notch_q", "sos",
    "filter_state_initialization", "warmup_samples", "core_start_samples", "core_stop_samples",
    "window_samples", "hop_samples", "feature_names", "feature_order", "wl_normalized",
    "zc_threshold_uv", "ssc_threshold_uv", "zc_definition", "ssc_definition", "qc"]

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

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

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

def dump_json(path, value):
    """Only write new snapshots; live journals stay append-only (no events.tmp rename)."""
    path = Path(path)
    text = json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)
    path.write_text(text, encoding='utf-8')


def unique_directory(parent, prefix):
    path = Path(parent).expanduser().resolve() / (prefix + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    path.mkdir(parents=True, exist_ok=False)
    return path


def top_two(values, reference):
    values, reference = np.asarray(values, float), np.asarray(reference, float)
    require(reference.shape == (4,) and np.isfinite(reference).all() and (reference > 0).all(), '静息参考必须为四个正数。')
    return float(np.sort(values / reference)[-2:].mean())


class TKEOperator:
    """One-sample delayed causal TKE. No future sample, no per-packet restart."""
    def __init__(self):
        self.tail = np.empty((0, 4), float)

    def reset(self):
        self.tail = np.empty((0, 4), float)

    def process(self, values):
        values = np.asarray(values, float)
        require(values.ndim == 2 and values.shape[1] == 4, 'TKE输入需为(N,4)。')
        if not len(values):
            return values.copy()
        history = len(self.tail)
        joined = np.concatenate([self.tail, values])
        result = np.full(values.shape, np.nan)
        if len(joined) >= 3:
            energy = np.abs(joined[1:-1] ** 2 - joined[:-2] * joined[2:])
            first = max(0, 2 - history)
            result[first:] = energy[max(0, history - 2):]
        self.tail = joined[-2:].copy()
        return result


class FeatureStream:
    """Original classic window schedule/features, plus synchronized causal TKE."""
    def __init__(self, config):
        self.config = config
        self.width, self.hop = int(config['window_samples']), int(config['hop_samples'])
        self.processor = CausalEMGFilter.from_config(config)
        self.tke = TKEOperator()
        self.position, self.segment_id, self.valid = 0, 0, False
        self.raw_buffer = np.empty((0, 4))
        self.filtered_buffer = np.empty((0, 4))
        self.energy_buffer = np.empty((0, 4))
        self.next_end = None

    def reset(self, row):
        self.processor.reset()
        self.tke.reset()
        self.segment_id += 1
        self.raw_buffer = np.empty((0, 4))
        self.filtered_buffer = np.empty((0, 4))
        self.energy_buffer = np.empty((0, 4))
        self.next_end = int(row) + int(self.config['warmup_samples']) + self.width

    def interrupt(self):
        self.valid = False
        self.processor.reset()
        self.tke.reset()

    def push(self, samples, break_before=None):
        values = np.asarray(samples, float)
        require(values.ndim == 2 and values.shape[1] == 4, '流输入需为(N,4)。')
        flags = np.zeros(len(values), bool) if break_before is None else np.asarray(break_before, bool)
        require(flags.shape == (len(values),), '连续性标记长度错误。')
        finite = np.isfinite(values).all(axis=1)
        boundaries = np.unique(np.r_[0, len(values), np.flatnonzero(flags), np.flatnonzero(finite[1:] != finite[:-1]) + 1])
        frames = []
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            lo, hi = int(lo), int(hi)
            if hi <= lo:
                continue
            start, stop = self.position + lo, self.position + hi
            if not finite[lo]:
                self.interrupt()
                self.raw_buffer = np.empty((0, 4))
                self.filtered_buffer = np.empty((0, 4))
                self.energy_buffer = np.empty((0, 4))
                continue
            if flags[lo] or not self.valid:
                self.reset(start)
            self.valid = True
            filtered = self.processor.process(values[lo:hi])
            energy = self.tke.process(filtered)
            raw = np.concatenate([self.raw_buffer, values[lo:hi]])
            xf = np.concatenate([self.filtered_buffer, filtered])
            et = np.concatenate([self.energy_buffer, energy])
            origin = start - len(self.raw_buffer)
            while self.next_end <= stop:
                end = int(self.next_end)
                a, b = end - self.width - origin, end - origin
                require(a >= 0 and b <= len(raw), '窗口缓冲区错位。')
                fw = xf[a:b].astype(np.float32)
                status, reasons, usable, _ = inspect_window(raw[a:b], fw, np.ones(self.width, bool),
                    np.zeros(self.width, bool), 0, self.width, True, self.config)
                feature = extract_features(fw, self.config['zc_threshold_uv'], self.config['ssc_threshold_uv'])
                tke_mean = np.mean(et[a:b], axis=0)
                frames.append({'start_row': end - self.width, 'stop_row': end, 'decision_row': end - 1,
                    'segment_id': self.segment_id, 'qc_status': status, 'qc_reasons': ';'.join(reasons),
                    'usable': bool(usable), 'features': feature, 'rms': feature[[1, 6, 11, 16]],
                    'tke': tke_mean, 'tke_usable': bool(np.isfinite(tke_mean).all())})
                self.next_end += self.hop
            self.raw_buffer = raw[-self.width:].copy()
            self.filtered_buffer = xf[-self.width:].copy()
            self.energy_buffer = et[-self.width:].copy()
        self.position += len(values)
        return frames


class ComparisonEngine:
    """Inference accepts only signal/timestamps; no trial truth or gesture hints."""
    def __init__(self, bundle, original_params, max_data_age=.75):
        self.bundle = bundle
        self.config = bundle['preprocess_config']
        self.stream = FeatureStream(self.config)
        self.original = copy.deepcopy(original_params)
        self.max_data_age = float(max_data_age)
        self.setup = None
        self.detectors = {}
        self.events = []
        self.latest = None

    def activate(self, setup):
        require(self.setup is None, '正式测试已经冻结；请新开程序进行下一轮校准。')
        self.setup = copy.deepcopy(setup)
        self.detectors = {arm: GestureDetector(self.setup['arms'][arm]['params']) for arm in ARM_NAMES}

    def interrupt(self, reason):
        self.stream.interrupt()
        for detector in self.detectors.values():
            detector.close(max(0, self.stream.position - 1), reason)
            detector.reset_state()
            detector.segment_id = None

    def process(self, samples, stamps, flags, clock):
        origin = self.stream.position
        stamps = np.asarray(stamps, float)
        frames = self.stream.push(samples, flags)
        if not frames:
            return []
        features = np.asarray([f['features'] for f in frames])
        eligible = np.array([f['usable'] for f in frames]) & np.isfinite(features).all(axis=1)
        predictions, scores = np.full(len(frames), -1, int), np.full((len(frames), 2), np.nan)
        if eligible.any():
            predictions[eligible], scores[eligible] = predict(self.bundle['pipeline'], features[eligible])
        for i, frame in enumerate(frames):
            row = frame['decision_row']
            stamp = float(stamps[row - origin])
            now = float(clock(row))
            timely = bool(np.isfinite(stamp) and -.25 <= now - stamp <= self.max_data_age)
            frame.update({'decision_lsl_time_s': stamp, 'processed_local_clock_s': now,
                'data_age_s': now - stamp, 'timely': timely, 'predicted_class_id': int(predictions[i]),
                'scores': scores[i], 'arms': {}})
            for arm in ARM_NAMES:
                profile = self.setup['arms'][arm] if self.setup else None
                metric = frame['tke'] if arm == 'C' else frame['rms']
                reference = profile['reference'] if profile else (self.original['noise_reference_rms_uv'] if arm == 'A' else None)
                activity = top_two(metric, reference) if reference is not None else None
                state, event = 'calibration', None
                if self.setup and row >= int(self.setup['activation_row']):
                    usable = frame['usable'] and timely and (frame['tke_usable'] if arm == 'C' else True)
                    state, event = self.detectors[arm].step(row, frame['segment_id'], usable,
                        float(activity), int(predictions[i]), scores[i])
                    if not timely:
                        state = 'stale'
                    if event is not None:
                        event.update({'arm': arm, 'class_name': NAMES[event['class_id']],
                            'decision_lsl_time_s': stamp, 'emitted_local_clock_s': now,
                            'data_age_s': now - stamp, 'config_id': self.setup['config_id']})
                        self.events.append(event)
                frame['arms'][arm] = {'activity': activity, 'state': state,
                    'event_id': event['event_id'] if event else None,
                    'current_class_id': self.detectors[arm].active_event['class_id']
                        if self.setup and self.detectors[arm].active_event else None}
            self.latest = frame
        return frames


def events_for_attempt(events, attempt, arm):
    return [e for e in events if e['arm'] == arm and
        attempt['start_clock_s'] <= e['emitted_local_clock_s'] < attempt['end_clock_s']]


def calibration_quality(records, interval, seconds, config):
    selected = [r for r in records if interval['start_clock_s'] <= r['processed_local_clock_s'] < interval['end_clock_s']]
    good = [r for r in selected if r['usable'] and r['tke_usable'] and r['timely']]
    expected = seconds * float(config['fs']) / int(config['hop_samples'])
    require(len(good) >= .85 * expected, '有效数据不足85%%，请检查数据流并重新测量。')
    require(len({r['segment_id'] for r in selected}) == 1, '校准期间有连续性重置，请重新测量。')
    return good


def make_reference(rest_records, metric):
    reference = np.median(np.asarray([r[metric] for r in rest_records]), axis=0)
    require(reference.shape == (4,) and np.isfinite(reference).all() and (reference > 1e-12).all(), '静息参考过小或无效，请检查四通道信号。')
    return reference


def choose_thresholds(records, rest_interval, attempts, metric, reference, source_params):
    """Gate selection uses detection/miss/duplicate/latency, never correct-class scores."""
    usable = np.array([r['usable'] and r['timely'] and (r['tke_usable'] if metric == 'tke' else True) for r in records])
    activity = np.array([top_two(r[metric], reference) for r in records])
    clocks = np.array([r['processed_local_clock_s'] for r in records])
    rest = usable & (clocks >= rest_interval['start_clock_s']) & (clocks < rest_interval['end_clock_s'])
    require(rest.sum() >= 30, '缺少有效静息校准窗口。')
    stable = np.zeros(len(records), bool)
    for trial in attempts:
        stable |= (clocks >= trial['start_clock_s'] + .5) & (clocks < trial['end_clock_s'] - .3)
    active = activity[stable & usable]
    require(len(active) >= 12, '有效动作窗口不足，无法选择门限。')
    r = activity[rest]
    candidates = np.unique(np.r_[np.quantile(r, [.95, .975, .99, .995]),
        np.quantile(active, [.05, .1, .2, .35, .5, .7]),
        (np.quantile(r, .99) + np.quantile(active, [.1, .25])) / 2])
    candidates = candidates[np.isfinite(candidates) & (candidates > np.median(r))]
    require(len(candidates), '活动与静息无法形成有效候选门限。')
    best, best_key, trials = None, None, []
    for on in candidates:
        offs = np.unique(np.r_[np.quantile(r, [.75, .9, .95]), on * np.array([.5, .65, .8])])
        for off in offs:
            if not (0 < off < on):
                continue
            params = copy.deepcopy(source_params)
            params.update({'on_threshold': float(on), 'off_threshold': float(off)})
            detector = GestureDetector(params)
            outputs = []
            for i, record in enumerate(records):
                _, event = detector.step(record['decision_row'], record['segment_id'], bool(usable[i]),
                    float(activity[i]), record['predicted_class_id'], record['scores'])
                if event:
                    outputs.append({'emitted_local_clock_s': float(clocks[i])})
            rest_false = sum(rest_interval['start_clock_s'] <= e['emitted_local_clock_s'] < rest_interval['end_clock_s'] for e in outputs)
            misses = {0: [], 1: []}
            delays, duplicates = [], 0
            for attempt in attempts:
                found = [e for e in outputs if attempt['start_clock_s'] <= e['emitted_local_clock_s'] < attempt['end_clock_s']]
                misses[attempt['actual_class_id']].append(int(not found))
                duplicates += max(0, len(found) - 1)
                if found:
                    delays.append(found[0]['emitted_local_clock_s'] - attempt['start_clock_s'])
            balanced_miss = float(np.mean([np.mean(misses[k]) for k in [0, 1]]))
            p90 = float(np.quantile(delays, .9)) if delays else 1e9
            key = (int(rest_false > 0), balanced_miss, duplicates, rest_false, p90, -float(on))
            candidate = {'on_threshold': float(on), 'off_threshold': float(off),
                'rest_false_events': rest_false, 'balanced_miss_rate': balanced_miss,
                'misses_by_class': {str(k): int(sum(v)) for k, v in misses.items()},
                'duplicate_events': duplicates, 'button_to_output_p90_s': None if not delays else p90}
            trials.append(candidate)
            if best_key is None or key < best_key:
                best_key, best = key, candidate
    require(best is not None, '没有可用的门限组合。')
    selected = copy.deepcopy(source_params)
    selected.update({key: best[key] for key in ['on_threshold', 'off_threshold']})
    return selected, {'selected': best, 'candidate_count': len(trials), 'candidates': trials,
        'uses_correct_class_to_select_thresholds': False,
        'selection_scope': 'rest_and_posthoc_labelled_calibration_attempts_only',
        'independent_validation': False}


class Experiment:
    """Owns evaluation labels. Never passed into ComparisonEngine.process/step."""
    def __init__(self, config, rest_seconds=30., action_seconds=4.):
        self.config = config
        self.rest_seconds, self.action_seconds = float(rest_seconds), float(action_seconds)
        self.records = []
        self.attempts, self.rest_tests = [], []
        self.rest_interval = None
        self.active = None
        self.pending = None
        self.frozen = False
        self.phase = 'need_rest'
        self.message = '数据预热后，保持自然放松，点击“测静息”。'
        self.flow = []
        self.next_id = 1

    def note(self, kind, now, **fields):
        item = {'kind': kind, 'clock_s': float(now), **fields}
        self.flow.append(item)
        return item

    def start(self, kind, now):
        require(self.active is None and self.pending is None, '请先完成当前记录及事后标注。')
        if kind == 'rest_calibration':
            require(not self.frozen and not self.attempts, '已有动作校准记录；需要重测时请重新启动程序。')
            duration = self.rest_seconds
        elif kind == 'calibration':
            require(not self.frozen and self.rest_interval is not None, '请先完成静息校准。')
            duration = self.action_seconds
        elif kind == 'test':
            require(self.frozen, '请先冻结门限。')
            duration = self.action_seconds
        elif kind == 'rest_test':
            require(self.frozen, '请先冻结门限。')
            duration = 60.
        else:
            raise ValueError('未知记录类型。')
        self.active = {'attempt_id': self.next_id, 'stage': kind, 'start_clock_s': float(now),
            'end_clock_s': float(now + duration), 'duration_s': float(duration),
            'actual_class_id': None, 'valid': True, 'invalid_reason': None}
        self.next_id += 1
        self.phase = kind
        self.message = '持续放松。' if kind.startswith('rest') else '现在自由做石头或布，保持至倒计时结束。'
        self.note('interval_start', now, interval=copy.deepcopy(self.active))

    def observe(self, records):
        if not self.frozen:
            self.records.extend(records)

    def tick(self, now):
        if self.active is None or now < self.active['end_clock_s']:
            return
        interval, self.active = self.active, None
        self.note('interval_end', now, interval=copy.deepcopy(interval))
        if interval['stage'] == 'rest_calibration':
            try:
                calibration_quality(self.records, interval, self.rest_seconds, self.config)
                self.rest_interval = interval
                self.phase = 'calibration_ready'
                self.message = '静息已记录。请自由做石头/布各至少3次，每次结束后确认实际动作。'
            except ValueError as exc:
                self.rest_interval = None
                self.phase = 'need_rest'
                self.message = str(exc)
                self.note('rest_calibration_rejected', now, reason=str(exc))
        elif interval['stage'] == 'rest_test':
            self.rest_tests.append(interval)
            self.phase = 'testing'
            self.message = '静息测试结束，结果已保存。'
        else:
            interval['quality_coverage'] = self.coverage(interval)
            if interval['quality_coverage'] < .85:
                interval.update({'valid': False, 'invalid_reason': 'timely_usable_window_coverage_below_85_percent'})
            self.pending = interval
            self.phase = 'label_pending'
            self.message = '本次记录结束。请点击刚才实际做的动作；没有输出也要标注。'

    def coverage(self, interval):
        frames = getattr(self, 'all_test_records', []) if self.frozen else self.records
        count = sum(r['usable'] and r['timely'] and interval['start_clock_s'] <= r['processed_local_clock_s'] < interval['end_clock_s'] for r in frames)
        expected = interval['duration_s'] * float(self.config['fs']) / int(self.config['hop_samples'])
        return min(1., count / max(1., expected))

    def label(self, actual, now):
        require(self.pending is not None, '只能在记录窗口结束后标注。')
        require(actual in [0, 1], '实际动作只能是石头或布。')
        item = self.pending
        item['actual_class_id'] = int(actual)
        item['label_clock_s'] = float(now)
        item['label_source'] = 'user_confirmation_after_capture_and_predictions'
        self.attempts.append(item)
        self.pending = None
        self.note('posthoc_label', now, attempt=copy.deepcopy(item))
        self.phase = 'testing' if self.frozen else 'calibration_ready'
        self.message = '已保存实际%s。请放松，再记录下一次。' % NAMES[actual]
        return item

    def cancel(self, now, reason='user_cancelled_or_action_not_completed'):
        item = self.pending or self.active
        require(item is not None, '当前没有可作废的记录。')
        item.update({'valid': False, 'invalid_reason': reason})
        self.note('invalid_interval', now, interval=copy.deepcopy(item))
        self.pending = self.active = None
        self.phase = 'testing' if self.frozen else ('calibration_ready' if self.rest_interval else 'need_rest')
        self.message = '本次已作废并保留审计记录。'

    def freeze(self, engine, model_sha, now):
        require(not self.frozen and self.active is None and self.pending is None, '当前不能冻结门限。')
        require(self.rest_interval is not None, '请先测静息。')
        valid = [a for a in self.attempts if a['stage'] == 'calibration' and a['valid']]
        counts = Counter(a['actual_class_id'] for a in valid)
        require(counts[0] >= 3 and counts[1] >= 3, '有效校准动作需石头、布各至少3次。')
        rest = calibration_quality(self.records, self.rest_interval, self.rest_seconds, self.config)
        rms_ref, tke_ref = make_reference(rest, 'rms'), make_reference(rest, 'tke')
        b, report_b = choose_thresholds(self.records, self.rest_interval, valid, 'rms', rms_ref, engine.original)
        c, report_c = choose_thresholds(self.records, self.rest_interval, valid, 'tke', tke_ref, engine.original)
        setup = {'schema': APP_SCHEMA, 'script_version': APP_VERSION, 'created_at': datetime.now().isoformat(),
            'model_sha256': model_sha, 'preprocess_config': self.config,
            'activation_row': engine.stream.position, 'calibration_date': datetime.now().strftime('%Y-%m-%d'),
            'rest_interval': self.rest_interval, 'calibration_attempts': valid,
            'selection_reports': {'B': report_b, 'C': report_c},
            'arms': {'A': {'metric': 'rms', 'reference': engine.original['noise_reference_rms_uv'], 'params': copy.deepcopy(engine.original)},
                'B': {'metric': 'rms', 'reference': rms_ref.tolist(), 'params': b},
                'C': {'metric': 'tke', 'reference': tke_ref.tolist(), 'params': c}},
            'reference_policy': 'frozen_for_entire_test_no_online_adaptation',
            'labels_used_for_svm_fit': False, 'test_labels_used_for_gate_selection': False,
            'latency_scope': 'record_button_to_output_not_muscle_onset'}
        setup['config_id'] = hashlib.sha256(json.dumps(json_safe(setup), sort_keys=True).encode()).hexdigest()
        engine.activate(setup)
        self.frozen = True
        self.phase = 'testing'
        self.all_test_records = []
        self.note('thresholds_frozen', now, config_id=setup['config_id'], activation_row=setup['activation_row'])
        self.message = '门限已冻结。请先放松，再做动作；“记录一次测试”只记时间，不控制识别。'
        return setup


def evaluate(attempts, rest_tests, events, records, fs, hop):
    valid = [a for a in attempts if a['stage'] == 'test' and a.get('actual_class_id') in [0, 1] and a['valid']]
    all_test = [a for a in attempts if a['stage'] == 'test']
    result = {'evaluation_scope': 'posthoc_labelled_button_intervals',
        'test_attempts_labelled': len(all_test), 'valid_test_attempts': len(valid),
        'quality_invalid_test_attempts': len(all_test) - len(valid),
        'true_muscle_onset_latency_s': None, 'rest_labels': 'operator_instructed_continuous_relaxation', 'arms': {}}
    for arm in ARM_NAMES:
        counts = Counter()
        per_class = {k: Counter() for k in [0, 1]}
        delays, release_delays, confusion = [], [], np.zeros((2, 3), int)
        rows = []
        for attempt in valid:
            truth = attempt['actual_class_id']
            outputs = events_for_attempt(events, attempt, arm)
            prediction = outputs[0]['class_id'] if outputs else None
            outcome = 'miss' if prediction is None else ('correct' if prediction == truth else 'wrong')
            counts[outcome] += 1
            per_class[truth][outcome] += 1
            counts['duplicates'] += max(0, len(outputs) - 1)
            confusion[truth, prediction if prediction is not None else 2] += 1
            delay = outputs[0]['emitted_local_clock_s'] - attempt['start_clock_s'] if outputs else None
            if delay is not None:
                delays.append(delay)
            next_start = min([a['start_clock_s'] for a in all_test if a['start_clock_s'] >= attempt['end_clock_s']] + [float('inf')])
            released = [r['processed_local_clock_s'] for r in records
                if attempt['end_clock_s'] <= r['processed_local_clock_s'] < next_start
                and r.get('arms', {}).get(arm, {}).get('state') == 'released']
            release_delay = min(released) - attempt['end_clock_s'] if released else None
            if release_delay is not None:
                release_delays.append(release_delay)
            rows.append({'attempt_id': attempt['attempt_id'], 'truth': truth, 'first_prediction': prediction,
                'outcome': outcome, 'output_count': len(outputs), 'button_to_output_s': delay,
                'task_window_end_to_release_s': release_delay})
        rest_events, rest_valid_seconds, rest_wall_seconds = 0, 0., 0.
        for rest in rest_tests:
            if not rest.get('valid', True):
                continue
            rest_events += len(events_for_attempt(events, rest, arm))
            rest_wall_seconds += rest['duration_s']
            n = sum(r['usable'] and r['timely'] and (r['tke_usable'] if arm == 'C' else True)
                and rest['start_clock_s'] <= r['processed_local_clock_s'] < rest['end_clock_s'] for r in records)
            rest_valid_seconds += min(rest['duration_s'], n * hop / fs)
        rest_coverage = rest_valid_seconds / rest_wall_seconds if rest_wall_seconds else None
        result['arms'][arm] = {'name': ARM_NAMES[arm], 'correct': counts['correct'], 'wrong': counts['wrong'],
            'misses': counts['miss'], 'duplicate_outputs': counts['duplicates'],
            'attempt_success_rate': counts['correct'] / len(valid) if valid else None,
            'labelled_output_accuracy': counts['correct'] / len(delays) if delays else None,
            'miss_rate': counts['miss'] / len(valid) if valid else None,
            'button_to_first_output_median_s': float(np.median(delays)) if delays else None,
            'button_to_first_output_p90_s': float(np.quantile(delays, .9)) if delays else None,
            'task_window_end_to_release_median_s': float(np.median(release_delays)) if release_delays else None,
            'release_unconfirmed_attempts': len(valid) - len(release_delays),
            'rest_false_events': rest_events, 'rest_observed_seconds': rest_wall_seconds,
            'rest_usable_seconds': rest_valid_seconds, 'rest_quality_coverage': rest_coverage,
            'rest_false_events_per_minute': rest_events * 60 / rest_valid_seconds
                if rest_valid_seconds > 0 and rest_coverage >= .85 else None,
            'per_class': {str(k): {'name': NAMES[k], 'n': sum(per_class[k][x] for x in ['correct', 'wrong', 'miss']),
                **{x: per_class[k][x] for x in ['correct', 'wrong', 'miss']}} for k in [0, 1]},
            'confusion_matrix_columns': NAMES + ['漏检'], 'confusion_matrix': confusion.tolist(), 'attempts': rows,
            'total_output_events': sum(e['arm'] == arm for e in events)}
    return result


class Journal:
    def __init__(self, output):
        self.output = Path(output)
        self.handles = {}
        self.writers = {}
        self.raw_count = 0
        self.last_flush = 0.
        self.event_count = self.flow_count = 0

    def csv_rows(self, name, rows, fields=None):
        if not rows:
            return
        if name not in self.writers:
            h = (self.output / name).open('x', encoding='utf-8-sig', newline='')
            writer = csv.DictWriter(h, fieldnames=fields or list(rows[0]))
            writer.writeheader()
            self.handles[name], self.writers[name] = h, writer
        self.writers[name].writerows(rows)

    def jsonl(self, name, item):
        if name not in self.handles:
            self.handles[name] = (self.output / name).open('x', encoding='utf-8')
        self.handles[name].write(json.dumps(json_safe(item), ensure_ascii=False, allow_nan=False) + '\n')

    def raw(self, values, stamps, arrival, flags, invalid, stale):
        rows = []
        for i, value in enumerate(values):
            rows.append({'row': self.raw_count + i, 'lsl_timestamp_s': float(stamps[i]),
                'received_local_clock_s': float(arrival),
                **{'ch%d_uv' % (k + 1): float(value[k]) for k in range(4)},
                'break_before': int(flags[i]), 'transport_invalid': int(invalid[i]), 'stale_input': int(stale[i])})
        self.csv_rows('raw_input.csv', rows)
        self.raw_count += len(values)

    def windows(self, records):
        rows = []
        for r in records:
            item = {k: r[k] for k in ['start_row', 'stop_row', 'decision_row', 'segment_id', 'qc_status',
                'qc_reasons', 'usable', 'tke_usable', 'decision_lsl_time_s', 'processed_local_clock_s',
                'data_age_s', 'timely', 'predicted_class_id']}
            item.update(dict(zip(FEATURE_NAMES, r['features'])))
            item.update({'tke_ch%d_uv2' % (k + 1): float(r['tke'][k]) for k in range(4)})
            item.update({'score_0': float(r['scores'][0]), 'score_1': float(r['scores'][1])})
            for arm, state in r['arms'].items():
                item.update({arm + '_' + k: v for k, v in state.items()})
            rows.append(item)
        self.csv_rows('windows.csv', rows)

    def sync(self, engine, experiment):
        while self.event_count < len(engine.events):
            self.jsonl('events.jsonl', copy.deepcopy(engine.events[self.event_count]))
            self.event_count += 1
        while self.flow_count < len(experiment.flow):
            self.jsonl('evaluation_journal.jsonl', experiment.flow[self.flow_count])
            self.flow_count += 1
        self.flush()

    def flush(self, force=False):
        if force or time.monotonic() - self.last_flush >= .5:
            for h in self.handles.values():
                h.flush()
            self.last_flush = time.monotonic()

    def close(self):
        self.flush(True)
        for handle in self.handles.values():
            handle.close()


STATE_TEXT = {'calibration': '校准中', 'released': '已放松，等待动作',
    'waiting_for_release': '请先放松，尚未解除锁定', 'quiet_or_candidate': '等待动作达到启动条件',
    'settling': '动作稳定等待', 'collecting_votes': '正在投票', 'unstable': '类别投票未稳定',
    'triggered': '已识别，请放松', 'active_latched': '本次已输出，等待放松',
    'activity_below_on_waiting': '活动未达到启动门限', 'quality_lock': '信号质控暂停', 'stale': '过时或未来数据，暂停输出'}


def write_report(output, session, setup, experiment, engine, status='completed', failure=None):
    records = getattr(experiment, 'all_test_records', [])
    result = evaluate(experiment.attempts, experiment.rest_tests, engine.events, records,
        float(engine.config['fs']), int(engine.config['hop_samples']))
    result.update({'schema': APP_SCHEMA, 'script_version': APP_VERSION, 'status': status, 'failure': failure,
        'model_sha256': session['model_sha256'], 'model_file_unchanged': file_sha256(Path(session['model_path'])) == session['model_sha256'],
        'config_id': setup['config_id'] if setup else None,
        'pending_unlabelled_attempts': int(experiment.pending is not None),
        'unfinished_interval': experiment.active, 'attempts_including_calibration': experiment.attempts,
        'invalidated_intervals': [x['interval'] for x in experiment.flow if x['kind'] == 'invalid_interval'],
        'rest_test_intervals': experiment.rest_tests,
        'limits': ['按钮到输出时间不等于真实肌肉起始延迟。',
            '静息真值依赖操作者按提示放松，未用独立EMG/动作传感器确认。',
            'LSL时间戳和Python缓存不能证明GUI/板卡上游物理同步。',
            '若当天模型包含本次测试动作的训练数据，不属于独立分类验证。',
            '校准动作与测试动作分别统计；不依据正式测试标签更新参数。']})
    dump_json(Path(output) / 'summary.json', result)
    head = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>TKE/RMS门控对照</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;line-height:1.7}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse}small{color:#555}</style>'
    body = ['<h1>石头/布：TKE与RMS对照</h1>', '<p>版本：%s；状态：%s</p>' % (APP_VERSION, html.escape(status)),
        '<p>按动作总正确率包括漏检；仅已标注且数据质量有效的测试进入分母。校准结果不计入测试。</p>',
        '<table><tr><th>方案</th><th>正确/总数</th><th>错误</th><th>漏检</th><th>重复输出</th><th>按钮到输出中位数</th><th>90分位</th><th>静息误输出/分</th></tr>']
    def fmt(value):
        return '—' if value is None else '%.3f' % value
    for arm, m in result['arms'].items():
        body.append('<tr><td>%s</td><td>%d/%d</td><td>%d</td><td>%d</td><td>%d</td><td>%s s</td><td>%s s</td><td>%s</td></tr>' %
            (ARM_NAMES[arm], m['correct'], result['valid_test_attempts'], m['wrong'], m['misses'], m['duplicate_outputs'],
             fmt(m['button_to_first_output_median_s']), fmt(m['button_to_first_output_p90_s']), fmt(m['rest_false_events_per_minute'])))
    body.append('</table><h2>解释与限制</h2><ul>' + ''.join('<li>%s</li>' % html.escape(x) for x in result['limits']) + '</ul>')
    if setup:
        body.append('<h2>冻结门限</h2><table><tr><th>方案</th><th>ON</th><th>OFF</th><th>参考</th></tr>')
        for arm, p in setup['arms'].items():
            body.append('<tr><td>%s</td><td>%.5g</td><td>%.5g</td><td>%s</td></tr>' %
                (ARM_NAMES[arm], p['params']['on_threshold'], p['params']['off_threshold'], html.escape(str(p['reference']))))
        body.append('</table><p>门限仅来自静息和动作校准；正式测试期间冻结。</p>')
    if failure:
        body.append('<pre>' + html.escape(str(failure)) + '</pre>')
    body.append('<p>逐窗记录：windows.csv；原始输入：raw_input.csv；输出：events.jsonl；标注审计：evaluation_journal.jsonl。</p></html>')
    (Path(output) / 'OPEN_REPORT.html').write_text(head + ''.join(body), encoding='utf-8')
    return result


def load_inputs(args):
    model, detector = args.model, args.detector
    if args.run:
        require(not model and not detector, '--run与--model/--detector二选一。')
        directory = Path(args.run).expanduser().resolve()
        state = read_json(directory / 'pipeline_state.json')
        require(state.get('schema') == 'openbci_rp_binary_pipeline_v1', '不是配套石头/布二分类流程目录。')
        resolved = []
        for key in ['daily_model', 'detector']:
            require(isinstance(state.get(key), str) and state[key], '流程记录缺少：' + key)
            path = Path(state[key]).expanduser()
            resolved.append(path if path.is_absolute() else directory / path)
        model, detector = resolved
    require(model and detector, '请提供--run流程目录，或同时提供--model与--detector。')
    bundle, model_path = load_daily(model)
    detector_path = Path(detector).expanduser().resolve()
    original = read_json(detector_path)
    validate_detector(bundle, model_path, original)
    config = bundle['preprocess_config']
    require(config['filter_mode'] == 'causal_sos' and int(config['n_channels']) == 4, '需要四通道因果滤波模型。')
    require(config['processing_unit'] == 'uV' and config['feature_names'] == FEATURE_NAMES, '模型单位或特征顺序不匹配。')
    require(config.get('wl_normalized') is False, '当前实现对应未归一化WL，请核对训练配置。')
    require(np.isclose(config['fs'], 250.), '当前试验版本仅支持250Hz，请勿强行修改采样率。')
    return bundle, model_path, original, detector_path


def get_lsl():
    try:
        import pylsl
        return pylsl
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError('无法加载pylsl：%s\n请在bci环境运行：python -m pip install pylsl' % exc) from exc


def describe_stream(s):
    return {'name': s.name(), 'type': s.type(), 'channels': s.channel_count(),
        'nominal_srate': s.nominal_srate(), 'source_id': s.source_id(), 'uid': s.uid()}


def connect_inlet(lsl, args, fs):
    streams = lsl.resolve_streams(wait_time=args.resolve_seconds)
    matching = [s for s in streams if s.name() == args.stream_name
        and (not args.stream_type or s.type().strip() == args.stream_type.strip())
        and (not args.source_id or s.source_id() == args.source_id)]
    require(len(matching) == 1, '需要唯一匹配LSL流，当前%d个。请检查GUI是否发送，并用--list-streams核对；重名时指定--source-id。' % len(matching))
    selected = matching[0]
    require(abs(float(selected.nominal_srate()) - fs) < 1e-6, 'LSL采样率与模型不一致。')
    need = max(args.channels + ([args.counter_channel] if args.counter_channel else []))
    require(selected.channel_count() >= need, '流的通道数不足。')
    inlet = lsl.StreamInlet(selected, max_buflen=2, recover=False, processing_flags=lsl.proc_clocksync)
    inlet.open_stream(timeout=args.resolve_seconds)
    if hasattr(inlet, 'flush'):
        inlet.flush()
    else:
        for _ in range(20):
            _, stamps = inlet.pull_chunk(timeout=0., max_samples=1024)
            if not stamps:
                break
    return inlet, describe_stream(selected)


def snapshot_for(engine, experiment, health, now):
    latest = engine.latest
    result = {'phase': experiment.phase, 'message': experiment.message,
        'active': copy.deepcopy(experiment.active), 'pending': copy.deepcopy(experiment.pending),
        'calibration_counts': dict(Counter(a['actual_class_id'] for a in experiment.attempts
            if a['stage'] == 'calibration' and a['valid'])), 'frozen': experiment.frozen,
        'rest_ready': experiment.rest_interval is not None, 'clock_s': now,
        'received_rows': engine.stream.position, 'health': copy.deepcopy(health), 'arms': {},
        'rms': None if latest is None else latest['rms'].tolist(),
        'tke': None if latest is None else latest['tke'].tolist(),
        'qc': None if latest is None else latest['qc_status'], 'pending_predictions': {}}
    if experiment.pending:
        for arm in ARM_NAMES:
            outputs = events_for_attempt(engine.events, experiment.pending, arm)
            result['pending_predictions'][arm] = outputs[0]['class_id'] if outputs else None
    for arm in ARM_NAMES:
        profile = engine.setup['arms'][arm] if engine.setup else None
        state = latest['arms'][arm] if latest else {'state': 'calibration', 'activity': None, 'current_class_id': None}
        result['arms'][arm] = {**state, 'on': profile['params']['on_threshold'] if profile else None,
            'off': profile['params']['off_threshold'] if profile else None,
            'count': sum(e['arm'] == arm for e in engine.events)}
    # Evaluation uses previously confirmed labels; it has no path back to inference.
    result['evaluation'] = evaluate(experiment.attempts, experiment.rest_tests, engine.events,
        getattr(experiment, 'all_test_records', []), float(engine.config['fs']), int(engine.config['hop_samples']))
    return json_safe(result)


def live_worker(args, bundle, model_path, original, detector_path, output, controls, updates, stop):
    engine = ComparisonEngine(bundle, original, args.max_data_age)
    experiment = Experiment(engine.config, args.rest_seconds, args.action_seconds)
    journal = Journal(output)
    model_sha = file_sha256(model_path)
    session = {'schema': APP_SCHEMA, 'script_version': APP_VERSION,
        'started_at': datetime.now().isoformat(), 'model_path': str(model_path), 'model_sha256': model_sha,
        'detector_path': str(detector_path), 'detector_sha256': file_sha256(detector_path),
        'preprocess_config': engine.config, 'original_detector': original,
        'class_ids': [0, 1], 'class_names': NAMES, 'subject_id': bundle['subject_id'],
        'model_calibration_date': bundle['calibration_date'], 'max_data_age_seconds': args.max_data_age,
        'channels_one_based': args.channels, 'input_unit': args.input_unit,
        'counter_channel_one_based': args.counter_channel,
        'rest_seconds': args.rest_seconds, 'action_seconds': args.action_seconds,
        'sklearn_version': sklearn.__version__, 'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
        'no_labels_in_inference': True, 'svm_fit_performed': False,
        'button_starts_detector': False, 'posthoc_label_resets_detector': False,
        'physical_synchrony_verified': False}
    dump_json(Path(output) / 'session.json', session)
    inlet, setup, failure = None, None, None
    health = {'state': 'connecting', 'rate_hz': None, 'data_age_s': None, 'queued_samples': None, 'issues': {}}
    guard = TransportGuard(float(engine.config['fs']), args.timestamp_gap_seconds,
        int(engine.config.get('counter_modulus', 256)))
    received = deque()
    rates_start, last_data, last_publish, started = time.monotonic(), time.monotonic(), 0., time.monotonic()
    force_break, stalled = True, False
    try:
        lsl = get_lsl()
        updates.put({'kind': 'snapshot', 'value': snapshot_for(engine, experiment, health, lsl.local_clock())})
        inlet, stream = connect_inlet(lsl, args, float(engine.config['fs']))
        session['stream'] = stream
        dump_json(Path(output) / 'session_connected.json', session)
        health['state'] = 'warming'
        while not stop.is_set():
            now = float(lsl.local_clock())
            experiment.tick(now)
            while True:
                try:
                    command = controls.get_nowait()
                except queue.Empty:
                    break
                kind = command['kind']
                now = float(lsl.local_clock())
                try:
                    if kind in ['rest_calibration', 'calibration', 'test', 'rest_test']:
                        require(engine.latest is not None and engine.latest['usable'] and engine.latest['timely']
                            and health['state'] == 'receiving', '请先等待连续、及时且质控可用的数据。')
                        experiment.start(kind, now)
                    elif kind == 'label':
                        experiment.label(command['actual'], now)
                    elif kind == 'cancel':
                        experiment.cancel(now)
                    elif kind == 'freeze':
                        updates.put({'kind': 'busy', 'message': '正在用校准记录选择门限，测试数据不参与。'})
                        setup = experiment.freeze(engine, model_sha, now)
                        dump_json(Path(output) / 'tke_rms_config.json', setup)
                        # Calibration selection can take time: explicitly discard queued input.
                        if hasattr(inlet, 'flush'):
                            inlet.flush()
                        else:
                            for _ in range(20):
                                if not inlet.pull_chunk(timeout=0., max_samples=1024)[1]:
                                    break
                        engine.interrupt('threshold_freeze_new_test_epoch')
                        force_break = True
                        health['state'] = 'warming'
                        last_data = time.monotonic()
                        reports = setup['selection_reports']
                        warnings = [ARM_NAMES[a] for a in ['B', 'C'] if reports[a]['selected']['balanced_miss_rate'] > 0
                            or reports[a]['selected']['rest_false_events'] > 0]
                        if warnings:
                            experiment.message += ' 注意：%s在校准中仍有漏检或误触发，请查看报告。' % '、'.join(warnings)
                    elif kind == 'resync':
                        if experiment.active or experiment.pending:
                            experiment.cancel(now, 'lsl_resubscription_during_capture')
                        inlet.close_stream()
                        inlet, stream = connect_inlet(lsl, args, float(engine.config['fs']))
                        engine.interrupt('manual_lsl_resubscribe')
                        force_break = True
                        guard = TransportGuard(float(engine.config['fs']), args.timestamp_gap_seconds,
                            int(engine.config.get('counter_modulus', 256)))
                        last_data, stalled = time.monotonic(), False
                        health['state'] = 'warming'
                        experiment.note('lsl_resubscribe', now, stream=stream, at_row=engine.stream.position)
                        experiment.message = '已重连LSL，正在重新预热；正式测试门限保持冻结，请先放松。'
                    elif kind == 'stop':
                        stop.set()
                except (ValueError, RuntimeError) as exc:
                    experiment.message = str(exc)
                    experiment.note('command_rejected', now, command=kind, reason=str(exc))
                    updates.put({'kind': 'notice', 'message': str(exc)})
            if stop.is_set():
                break
            values, stamps = inlet.pull_chunk(timeout=.05, max_samples=max(25, int(engine.config['fs'] / 5)))
            if not stamps:
                if time.monotonic() - last_data >= args.stall_seconds and not stalled:
                    engine.interrupt('reception_stall')
                    force_break, stalled = True, True
                    health['state'] = 'no_data'
                    experiment.note('transport_interruption', now, at_row=engine.stream.position, reason='reception_stall')
                    if experiment.active:
                        experiment.cancel(now, 'reception_stall_during_capture')
                    experiment.message = '暂未收到数据，输出已暂停。恢复后重新预热；也可点击重连LSL。'
            else:
                matrix, timestamps = np.asarray(values, float), np.asarray(stamps, float)
                require(matrix.ndim == 2 and len(matrix) == len(timestamps) and matrix.shape[1] == stream['channels'], 'LSL数据形状发生变化。')
                arrival = float(lsl.local_clock())
                counters = matrix[:, args.counter_channel - 1] if args.counter_channel else None
                flags, invalid, issues = guard.inspect(timestamps, counters)
                if force_break:
                    flags[0] = True
                    force_break = False
                stale = np.isfinite(timestamps) & ((arrival - timestamps > args.max_data_age) | (timestamps - arrival > .25))
                raw = matrix[:, np.asarray(args.channels) - 1] * {'uv': 1., 'mv': 1000., 'v': 1e6}[args.input_unit]
                for issue in issues:
                    reason = issue['reason']
                    health['issues'][reason] = health['issues'].get(reason, 0) + 1
                    experiment.note('transport_issue', arrival, at_row=engine.stream.position + issue['offset'], **issue)
                journal.raw(raw, timestamps, arrival, flags, invalid, stale)
                clean = raw.copy()
                clean[invalid | stale] = np.nan
                records = engine.process(clean, timestamps, flags, lambda row: lsl.local_clock())
                experiment.observe(records)
                if experiment.frozen:
                    experiment.all_test_records.extend(records)
                journal.windows(records)
                received.append((arrival, len(raw)))
                while received and arrival - received[0][0] > 5:
                    received.popleft()
                elapsed = arrival - received[0][0] if received else 0.
                rate = sum(n for _, n in list(received)[1:]) / elapsed if elapsed >= 1. else None
                try:
                    queued = int(inlet.samples_available())
                except (AttributeError, RuntimeError):
                    queued = None
                latest = engine.latest
                fresh_latest = latest is not None and engine.stream.valid and latest['segment_id'] == engine.stream.segment_id \
                    and -.25 <= arrival - latest['decision_lsl_time_s'] <= args.max_data_age
                health.update({'state': 'receiving' if fresh_latest and not stale[-1] else ('stale' if stale[-1] else 'warming'),
                    'rate_hz': rate, 'data_age_s': arrival - float(timestamps[-1]), 'queued_samples': queued})
                last_data, stalled = time.monotonic(), False
            journal.sync(engine, experiment)
            if time.monotonic() - last_publish >= .1:
                updates.put({'kind': 'snapshot', 'value': snapshot_for(engine, experiment, health, float(lsl.local_clock()))})
                last_publish = time.monotonic()
            if args.duration and time.monotonic() - started >= args.duration:
                stop.set()
    except Exception as exc:
        failure = '%s: %s' % (type(exc).__name__, exc)
        print(traceback.format_exc(), file=sys.stderr)
        updates.put({'kind': 'error', 'message': failure})
    finally:
        try:
            journal.sync(engine, experiment)
            journal.close()
            result = write_report(output, session, setup, experiment, engine, 'error' if failure else 'completed', failure)
            print('报告：%s' % (Path(output) / 'OPEN_REPORT.html'), flush=True)
            updates.put({'kind': 'finished', 'output': str(output), 'failure': failure, 'summary': result})
        except Exception as exc:
            failure = failure or str(exc)
            updates.put({'kind': 'finished', 'output': str(output), 'failure': '保存报告失败：' + failure})
            print('保存报告失败：%s；已写入的原始CSV/JSONL仍保留。' % failure, file=sys.stderr)
        if inlet:
            try:
                inlet.close_stream()
            except Exception:
                pass
        stop.set()


def run_gui(args, bundle, model_path, original, detector_path):
    import tkinter as tk
    from tkinter import ttk, messagebox
    output = unique_directory(args.output, 'RP_TKE_COMPARE_' + str(bundle['subject_id']))
    controls, updates, stop = queue.Queue(), queue.Queue(), threading.Event()
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError('无法打开图形界面，请在Windows桌面的bci环境运行。离线检查可用--check-only或--self-test。') from exc
    root.title('OpenBCI 石头/布 · TKE/RMS对照 ' + APP_VERSION)
    root.geometry('1050x860')
    root.minsize(940, 780)
    outer = ttk.Frame(root)
    outer.pack(fill='both', expand=True)
    canvas = tk.Canvas(outer, highlightthickness=0)
    scrollbar = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
    scrollbar.pack(side='right', fill='y')
    canvas.pack(side='left', fill='both', expand=True)
    canvas.configure(yscrollcommand=scrollbar.set)
    main = ttk.Frame(canvas, padding=16)
    interior = canvas.create_window((0, 0), window=main, anchor='nw')
    main.bind('<Configure>', lambda event: canvas.configure(scrollregion=canvas.bbox('all')))
    canvas.bind('<Configure>', lambda event: canvas.itemconfigure(interior, width=event.width))
    root.bind_all('<MouseWheel>', lambda event: canvas.yview_scroll(-int(event.delta / 120), 'units'))
    root.bind_all('<Button-4>', lambda event: canvas.yview_scroll(-1, 'units'))
    root.bind_all('<Button-5>', lambda event: canvas.yview_scroll(1, 'units'))
    ttk.Label(main, text='四通道 sEMG · 石头/布自动识别与门控对照', font=('Microsoft YaHei', 20, 'bold')).pack(anchor='w')
    ttk.Label(main, text='%s ｜ 模型校准 %s ｜ %.0f Hz ｜ 固定SVM，先动作后标注' %
        (bundle['subject_id'], bundle['calibration_date'], bundle['preprocess_config']['fs'])).pack(anchor='w', pady=4)
    if bundle['calibration_date'] != datetime.now().strftime('%Y-%m-%d'):
        ttk.Label(main, text='当前使用旧日期分类模型：本次可比较门控，分类能力仍需当天独立测试。', foreground='#9a6700').pack(anchor='w')
    display_arm = tk.StringVar(value='C')
    choice = ttk.Frame(main)
    choice.pack(anchor='w', pady=4)
    ttk.Label(choice, text='显示结果：').pack(side='left')
    for arm in ['C', 'B', 'A']:
        ttk.Radiobutton(choice, text=ARM_NAMES[arm], variable=display_arm, value=arm).pack(side='left', padx=12)
    gesture = ttk.Label(main, text='—', foreground='#174bc0', font=('Microsoft YaHei', 40, 'bold'))
    gesture.pack()
    state_label = ttk.Label(main, text='正在连接LSL……', font=('Microsoft YaHei', 13))
    state_label.pack()
    health_label = ttk.Label(main, text='')
    health_label.pack(pady=4)
    progress = ttk.Progressbar(main, maximum=100)
    progress.pack(fill='x')
    activity_label = ttk.Label(main, text='活动条不代表分类置信度；三组始终处理同一数据。')
    activity_label.pack(anchor='w', pady=3)
    signal_label = ttk.Label(main, text='RMS：—')
    signal_label.pack(anchor='w')
    phase_box = ttk.LabelFrame(main, text='校准与测试', padding=10)
    phase_box.pack(fill='x', pady=8)
    phase_label = ttk.Label(phase_box, text='预热后，请保持自然放松并测静息。', wraplength=980)
    phase_label.pack(anchor='w')
    countdown = ttk.Label(phase_box, text='')
    countdown.pack(anchor='w')
    row = ttk.Frame(phase_box)
    row.pack(fill='x', pady=6)
    buttons = {}
    for key, text in [('rest_calibration', '1 测静息（%g秒）' % args.rest_seconds),
        ('calibration', '2 记录校准动作（%g秒）' % args.action_seconds), ('freeze', '3 冻结门限')]:
        b = ttk.Button(row, text=text, command=lambda k=key: controls.put({'kind': k}))
        b.pack(side='left', padx=(0, 8))
        buttons[key] = b
    row2 = ttk.Frame(phase_box)
    row2.pack(fill='x', pady=4)
    for key, text in [('test', '记录一次测试（%g秒）' % args.action_seconds), ('rest_test', '静息测试（60秒）'), ('cancel', '本次作废')]:
        b = ttk.Button(row2, text=text, command=lambda k=key: controls.put({'kind': k}))
        b.pack(side='left', padx=(0, 8))
        buttons[key] = b
    ttk.Label(phase_box, text='“记录一次测试”只记时间；识别持续自动运行，按钮不预选类别、不重置门控。', foreground='#555').pack(anchor='w')
    label_box = ttk.LabelFrame(main, text='记录结束后，确认刚才实际做的动作', padding=10)
    label_box.pack(fill='x', pady=4)
    pending_label = ttk.Label(label_box, text='没有待标注记录；没有输出的测试也要确认实际动作。', wraplength=930)
    pending_label.pack(anchor='w')
    label_row = ttk.Frame(label_box)
    label_row.pack(fill='x', pady=5)
    label_buttons = []
    for actual, name in enumerate(NAMES):
        b = ttk.Button(label_row, text='实际是' + name, command=lambda k=actual: controls.put({'kind': 'label', 'actual': k}))
        b.pack(side='left', expand=True, fill='x', padx=4)
        label_buttons.append(b)
    ttk.Label(label_box, text='校准标签用于选择活动门限；正式测试标签只统计，不训练SVM，也不解除放松锁定。', foreground='#555').pack(anchor='w')
    table = ttk.Treeview(main, columns=('name', 'activity', 'threshold', 'state', 'score'), show='headings', height=3)
    for key, text, width in [('name', '方案', 95), ('activity', '活动值', 90), ('threshold', 'OFF / ON', 140),
        ('state', '当前状态', 280), ('score', '正确/有效测试；漏检', 200)]:
        table.heading(key, text=text)
        table.column(key, width=width, anchor='w')
    for arm in ARM_NAMES:
        table.insert('', 'end', iid=arm, values=(ARM_NAMES[arm], '—', '—', '校准中', '—'))
    table.pack(fill='x', pady=8)
    statistics = ttk.Label(main, text='正式测试正确率包括漏检；校准动作不计入测试。')
    statistics.pack(anchor='w')
    bottom = ttk.Frame(main)
    bottom.pack(fill='x', pady=8)
    resync_button = ttk.Button(bottom, text='重连LSL（不重启板子）', command=lambda: controls.put({'kind': 'resync'}))
    resync_button.pack(side='left')
    stop_button = ttk.Button(bottom, text='停止并保存', command=lambda: controls.put({'kind': 'stop'}))
    stop_button.pack(side='right')
    ttk.Label(main, text='输出：' + str(output), wraplength=1000, foreground='#555').pack(anchor='w')
    worker = threading.Thread(target=live_worker,
        args=(args, bundle, model_path, original, detector_path, output, controls, updates, stop), daemon=True)
    worker.start()
    last_snapshot, finished, closing = None, False, False

    def disabled(button, flag):
        button.configure(state='disabled' if flag else 'normal')

    def render(s):
        arm = display_arm.get()
        a = s['arms'][arm]
        live = s['health']['state'] == 'receiving'
        g = a['current_class_id']
        gesture.configure(text=NAMES[g] if g in [0, 1] and live else '—')
        state_label.configure(text=STATE_TEXT.get(a['state'], a['state']) if live else
            {'connecting': '正在连接LSL', 'warming': '等待新数据预热', 'stale': '数据时间戳过时，暂停输出', 'no_data': '数据流中断，暂停输出'}.get(s['health']['state'], '等待数据'))
        age, rate, queued = s['health'].get('data_age_s'), s['health'].get('rate_hz'), s['health'].get('queued_samples')
        health_label.configure(text='接收%d点 ｜ QC %s ｜ LSL年龄 %s ms ｜ 近5秒 %s Hz ｜ 缓存 %s点' %
            (s['received_rows'], s['qc'] or '—', '—' if age is None else '%.0f' % (age * 1000),
                '—' if rate is None else '%.1f' % rate, '—' if queued is None else queued))
        progress['value'] = min(100., max(0., 100 * a['activity'] / (2 * a['on']))) if a['activity'] is not None and a['on'] else 0.
        activity_label.configure(text='%s ｜ 活动 %s ｜ 释放≤%s ｜ 启动≥%s；活动条不代表置信度' %
            (ARM_NAMES[arm], '—' if a['activity'] is None else '%.3f' % a['activity'],
                '—' if a['off'] is None else '%.3f' % a['off'], '—' if a['on'] is None else '%.3f' % a['on']))
        signal_label.configure(text='分类窗口RMS（μV）：' + ('—' if s['rms'] is None else ' ｜ '.join('CH%d %.2f' % (i + 1, x) for i, x in enumerate(s['rms']))))
        phase_label.configure(text=s['message'])
        active, pending = s['active'], s['pending']
        remaining = max(0., active['end_clock_s'] - s['clock_s']) if active else 0.
        counts = s['calibration_counts']
        countdown.configure(text=('本段剩余 %.1f 秒 ｜ ' % remaining if active else '') +
            '有效校准：石头%d次，布%d次' % (counts.get('0', counts.get(0, 0)), counts.get('1', counts.get(1, 0))))
        predictions = ' ｜ '.join('%s：%s' % (ARM_NAMES[k], NAMES[v] if v in [0, 1] else '无输出')
            for k, v in s['pending_predictions'].items())
        pending_label.configure(text=('待标注 #%d：请选择刚才实际动作。%s %s' %
            (pending['attempt_id'], predictions, '本段数据质量不足，保留标签但不计入有效测试。' if not pending['valid'] else '')) if pending else
            '没有待标注记录；没有输出的测试也要确认实际动作。')
        for b in label_buttons:
            disabled(b, pending is None or finished)
        busy = active is not None or pending is not None or not live or finished
        disabled(buttons['rest_calibration'], busy or s['frozen'] or sum(counts.values()) > 0)
        disabled(buttons['calibration'], busy or s['frozen'] or not s['rest_ready'])
        disabled(buttons['freeze'], busy or s['frozen'] or min(counts.get('0', 0), counts.get('1', 0)) < 3)
        disabled(buttons['test'], busy or not s['frozen'])
        disabled(buttons['rest_test'], busy or not s['frozen'])
        disabled(buttons['cancel'], (active is None and pending is None) or finished)
        for key in ARM_NAMES:
            state, m = s['arms'][key], s['evaluation']['arms'][key]
            table.item(key, values=(ARM_NAMES[key], '—' if state['activity'] is None else '%.3f' % state['activity'],
                '—' if state['on'] is None else '%.3f / %.3f' % (state['off'], state['on']),
                STATE_TEXT.get(state['state'], state['state']), '%d/%d；漏检%d' %
                    (m['correct'], s['evaluation']['valid_test_attempts'], m['misses'])))
        statistics.configure(text='有效正式测试%d次；信号质量无效%d次。正确率=正确/全部有效测试（含漏检）。' %
            (s['evaluation']['valid_test_attempts'], s['evaluation']['quality_invalid_test_attempts']))

    def poll():
        nonlocal last_snapshot, finished
        try:
            while True:
                item = updates.get_nowait()
                if item['kind'] == 'snapshot':
                    last_snapshot = item['value']
                elif item['kind'] in ['notice', 'error', 'busy']:
                    phase_label.configure(text=item['message'])
                    if item['kind'] == 'busy':
                        for b in list(buttons.values()) + label_buttons:
                            disabled(b, True)
                        last_snapshot = None
                elif item['kind'] == 'finished':
                    finished = True
                    for b in list(buttons.values()) + label_buttons + [resync_button, stop_button]:
                        disabled(b, True)
                    phase_label.configure(text=('运行失败：' + item['failure']) if item.get('failure') else '已停止，报告与原始记录已保存。')
                    if item.get('failure'):
                        messagebox.showerror('运行未完成', item['failure'])
                    if closing:
                        root.destroy()
                        return
        except queue.Empty:
            pass
        if last_snapshot is not None and not finished:
            render(last_snapshot)
        root.after(100, poll)

    def close():
        nonlocal closing
        if finished:
            root.destroy()
            return
        closing = True
        controls.put({'kind': 'stop'})
        phase_label.configure(text='正在停止并保存，请稍候……')

    root.protocol('WM_DELETE_WINDOW', close)
    root.after(100, poll)
    try:
        root.mainloop()
    finally:
        stop.set()
        worker.join(timeout=10.)
    return output


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def replay_logs(directory, output_parent, model_override=None):
    """Re-run saved raw signal; truth is loaded only after inference completes."""
    directory = Path(directory).expanduser().resolve()
    session = read_json(directory / 'session.json')
    require(session.get('schema') == APP_SCHEMA, '请使用本TKE对照程序的日志目录。')
    setup = read_json(directory / 'tke_rms_config.json')
    setup_content = {k: v for k, v in setup.items() if k != 'config_id'}
    require(setup.get('config_id') == hashlib.sha256(json.dumps(json_safe(setup_content), sort_keys=True).encode()).hexdigest(),
        '冻结配置的内容校验失败，请勿手工修改后声称是原始独立测试。')
    bundle, model_path = load_daily(model_override or session['model_path'])
    require(file_sha256(model_path) == session['model_sha256'] == setup['model_sha256'], '回放模型哈希不匹配。')
    validate_detector(bundle, model_path, session['original_detector'])
    check_contract(bundle['preprocess_config'], session['preprocess_config'])
    check_contract(bundle['preprocess_config'], setup['preprocess_config'])
    output = unique_directory(output_parent, 'RP_TKE_REPLAY')
    with (directory / 'windows.csv').open(encoding='utf-8-sig', newline='') as f:
        expected_windows = {int(r['decision_row']): r for r in csv.DictReader(f)}
    require(expected_windows, '日志没有有效窗口。')
    engine = ComparisonEngine(bundle, session['original_detector'], session['max_data_age_seconds'])
    engine.activate(setup)
    all_frames, raw_rows = [], []
    journal = Journal(output)

    def consume(rows):
        origin = engine.stream.position
        require([int(r['row']) for r in rows] == list(range(origin, origin + len(rows))), '原始日志行号不连续。')
        values = np.array([[float(r['ch%d_uv' % k]) for k in range(1, 5)] for r in rows])
        timestamps = np.array([float(r['lsl_timestamp_s']) for r in rows])
        flags = np.array([bool(int(r['break_before'])) for r in rows])
        invalid = np.array([bool(int(r['transport_invalid'])) or bool(int(r['stale_input'])) for r in rows])
        values[invalid] = np.nan
        def clock(row):
            require(row in expected_windows, '回放出现原日志没有的窗口，可能存在重置日程差异。')
            return float(expected_windows[row]['processed_local_clock_s'])
        records = engine.process(values, timestamps, flags, clock)
        all_frames.extend(records)
        journal.windows(records)

    try:
        with (directory / 'raw_input.csv').open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                raw_rows.append(row)
                if len(raw_rows) >= 1000:
                    consume(raw_rows)
                    raw_rows = []
        if raw_rows:
            consume(raw_rows)
        expected_events = read_jsonl(directory / 'events.jsonl')
        expected = [(e['arm'], e['event_id'], e['decision_row'], e['class_id']) for e in expected_events]
        actual = [(e['arm'], e['event_id'], e['decision_row'], e['class_id']) for e in engine.events]
        features_match = len(all_frames) == len(expected_windows)
        max_feature_error = 0.
        max_tke_error = 0.
        for record in all_frames:
            previous = expected_windows[record['decision_row']]
            old_features = np.array([float(previous[k]) for k in FEATURE_NAMES])
            old_tke = np.array([float(previous['tke_ch%d_uv2' % k]) for k in range(1, 5)])
            features_match &= bool(np.allclose(record['features'], old_features, rtol=1e-9, atol=1e-9, equal_nan=True))
            features_match &= bool(np.allclose(record['tke'], old_tke, rtol=1e-9, atol=1e-9, equal_nan=True))
            if np.isfinite(record['features']).all():
                max_feature_error = max(max_feature_error, float(np.max(np.abs(record['features'] - old_features))))
            if np.isfinite(record['tke']).all():
                max_tke_error = max(max_tke_error, float(np.max(np.abs(record['tke'] - old_tke))))
        # Load evaluation labels only after all predictions/events have been reconstructed.
        exp = Experiment(bundle['preprocess_config'], session.get('rest_seconds', 30.), session.get('action_seconds', 4.))
        exp.frozen = True
        exp.all_test_records = all_frames
        flow = read_jsonl(directory / 'evaluation_journal.jsonl')
        exp.attempts = [item['attempt'] for item in flow if item['kind'] == 'posthoc_label']
        exp.rest_tests = [item['interval'] for item in flow if item['kind'] == 'interval_end' and item['interval']['stage'] == 'rest_test']
        exp.flow = flow
        replay_session = {**session, 'model_path': str(model_path)}
        journal.sync(engine, exp)
        journal.close()
        report = write_report(output, replay_session, setup, exp, engine, 'replayed')
        audit = {'schema': APP_SCHEMA, 'source_log_directory': str(directory),
            'raw_recalculation_features_and_tke_match': features_match,
            'events_match_by_arm_id_row_class': expected == actual,
            'expected_event_count': len(expected), 'replayed_event_count': len(actual),
            'max_feature_abs_error': max_feature_error, 'max_tke_abs_error': max_tke_error,
            'physical_board_or_gui_synchrony_proven': False,
            'model_used_for_prediction_only': True}
        dump_json(output / 'replay_audit.json', audit)
        print(json.dumps(json_safe(audit), ensure_ascii=False, indent=2), flush=True)
        print('回放报告：%s' % (output / 'OPEN_REPORT.html'), flush=True)
        return audit, report, output
    finally:
        for handle in journal.handles.values():
            if not handle.closed:
                handle.close()


def self_test():
    """Synthetic engineering checks only, never a measured real-user accuracy."""
    import unittest
    import tempfile
    from scipy.signal import butter

    def config():
        c = {k: None for k in CONTRACT_KEYS}
        c.update({'schema': 'openbci_rp_binary_4ch_preprocess_v1', 'fs': 250., 'n_channels': 4,
            'channel_columns': ['EXG Channel %d' % i for i in range(4)], 'processing_unit': 'uV',
            'filter_mode': 'causal_sos', 'sos': butter(4, [20, 100], fs=250., btype='bandpass', output='sos').tolist(),
            'warmup_samples': 50, 'window_samples': 50, 'hop_samples': 25,
            'zc_threshold_uv': 2., 'ssc_threshold_uv': 2., 'wl_normalized': False,
            'feature_names': FEATURE_NAMES,
            'qc': {'flat_p2p_uv': 1e-6, 'saturation_abs_uv': 1e6,
                'warn_filtered_p2p_uv': 1e5, 'corr_warning': .999}, 'counter_modulus': 256})
        return c

    def params():
        return {'schema': SCHEMA, 'script_version': '1.1.0', 'on_threshold': 4., 'off_threshold': 2.,
            'on_windows': 2, 'release_windows': 3, 'vote_windows': 3, 'minimum_votes': 3,
            'settle_samples': 0, 'settle_seconds': 0., 'noise_reference_rms_uv': [1.] * 4,
            'source_hashes': {}, 'source_calibration_runs': [1, 2], 'target_used_for_selection': False,
            'emit_requires_current_activity_above_on': True, 'vote_history_excludes_settling_and_low_activity': True,
            'window_samples': 50, 'hop_samples': 25, 'preprocess_config': config()}

    class FixedPipeline:
        classes_ = np.array([0, 1])
        def decision_function(self, values):
            return np.ones(len(values))

    def bundle():
        return {'pipeline': FixedPipeline(), 'preprocess_config': config()}

    def setup():
        return {'schema': APP_SCHEMA, 'config_id': 'synthetic-test-only', 'activation_row': 0,
            'preprocess_config': config(), 'arms': {a: {'reference': [1.] * 4, 'metric': 'tke' if a == 'C' else 'rms',
                'params': params()} for a in ARM_NAMES}}

    class Checks(unittest.TestCase):
        def test_tke_sinusoid_and_packet_invariance(self):
            t = np.arange(800)
            omega, amplitude = .7, 3.
            signal = np.repeat((amplitude * np.cos(omega * t))[:, None], 4, axis=1)
            expected = amplitude ** 2 * np.sin(omega) ** 2
            full = TKEOperator().process(signal)
            np.testing.assert_allclose(full[2:], expected, rtol=1e-12, atol=1e-12)
            operator = TKEOperator()
            sizes, pieces, pos = [1, 1, 7, 3, 61, 2, 125, 13, 587], [], 0
            for n in sizes:
                pieces.append(operator.process(signal[pos:pos + n]))
                pos += n
            np.testing.assert_allclose(np.concatenate(pieces), full, equal_nan=True)

        def test_tke_causality_and_reset(self):
            signal = np.random.default_rng(5).normal(size=(100, 4))
            a = TKEOperator().process(signal)
            signal[-1] += 1000
            b = TKEOperator().process(signal)
            np.testing.assert_array_equal(a[:-1], b[:-1])
            op = TKEOperator()
            op.process(signal)
            op.reset()
            self.assertTrue(np.isnan(op.process(signal[:2])).all())

        def test_filter_features_and_tke_chunking_with_breaks(self):
            signal = np.random.default_rng(22).normal(size=(1800, 4)) * 5
            signal[720:727] = np.nan
            flags = np.zeros(len(signal), bool)
            flags[[0, 453, 1303]] = True
            expected = FeatureStream(config()).push(signal, flags)
            stream = FeatureStream(config())
            actual, origin = [], 0
            for n in [1, 100, 37, 400, 23, 3, 800, 436]:
                actual += stream.push(signal[origin:origin + n], flags[origin:origin + n])
                origin += n
            self.assertEqual([r['decision_row'] for r in expected], [r['decision_row'] for r in actual])
            self.assertEqual([r['segment_id'] for r in expected], [r['segment_id'] for r in actual])
            np.testing.assert_allclose([r['features'] for r in expected], [r['features'] for r in actual], rtol=0, atol=0)
            np.testing.assert_allclose([r['tke'] for r in expected], [r['tke'] for r in actual], rtol=1e-12, atol=1e-12)

        def test_classic_feature_fixture(self):
            x = np.tile(np.array([-3., 0., 4., -4., 2.])[:, None], (1, 4))
            one = [2.6, math.sqrt(9.), 3 + 4 + 8 + 6, 3., 2.]
            np.testing.assert_allclose(extract_features(x), one * 4)

        def test_one_output_until_true_release(self):
            d = GestureDetector(params())
            row = 0
            def advance(activity, count, prediction=0):
                nonlocal row
                states = []
                for _ in range(count):
                    state, _ = d.step(row, 1, True, activity, prediction, np.array([1., -1.]))
                    row += 25
                    states.append(state)
                return states
            advance(10, 12)
            self.assertEqual(len(d.events), 0)
            advance(1, 5)
            advance(10, 30)
            self.assertEqual(len(d.events), 1)
            advance(1, 5)
            advance(10, 10, 1)
            self.assertEqual([e['class_id'] for e in d.events], [0, 1])

        def test_unstable_votes_and_qc_do_not_emit(self):
            d = GestureDetector(params())
            for k in range(4):
                d.step(k * 25, 1, True, 1., 0, np.array([1., -1.]))
            for k in range(4, 35):
                d.step(k * 25, 1, True, 10., k % 2, np.array([1., -1.]))
            self.assertEqual(d.events, [])
            d.step(900, 1, False, 20., 0, np.array([1., -1.]))
            for k in range(40, 45):
                d.step(k * 25, 1, True, 20., 0, np.array([1., -1.]))
            self.assertEqual(d.events, [])

        def test_posthoc_label_does_not_touch_detector(self):
            engine = ComparisonEngine(bundle(), params())
            engine.activate(setup())
            d = engine.detectors['C']
            for k in range(4):
                d.step(k * 25, 1, True, 1., 0, np.array([1., -1.]))
            for k in range(4, 10):
                d.step(k * 25, 1, True, 10., 0, np.array([1., -1.]))
            before = copy.deepcopy(d.__dict__)
            exp = Experiment(config())
            exp.frozen = True
            exp.pending = {'attempt_id': 1, 'stage': 'test', 'start_clock_s': 0., 'end_clock_s': 4.,
                'duration_s': 4., 'valid': True}
            exp.label(1, 5.)
            self.assertEqual(d.armed, before['armed'])
            self.assertEqual(d.emitted, before['emitted'])
            self.assertEqual(d.events, before['events'])
            self.assertTrue(d.emitted)

        def test_no_output_is_labelled_as_miss(self):
            exp = Experiment(config())
            exp.frozen = True
            exp.all_test_records = []
            exp.start('test', 10.)
            exp.all_test_records = [{'usable': True, 'timely': True, 'processed_local_clock_s': 10. + k / 10} for k in range(40)]
            exp.tick(14.)
            self.assertIsNotNone(exp.pending)
            exp.label(1, 15.)
            m = evaluate(exp.attempts, [], [], [], 250, 25)
            self.assertEqual(m['valid_test_attempts'], 1)
            self.assertEqual(m['arms']['C']['misses'], 1)
            self.assertEqual(m['arms']['C']['attempt_success_rate'], 0.)

        def test_metrics_wrong_miss_duplicate_and_rest_coverage(self):
            attempts = [{'attempt_id': k, 'stage': 'test', 'start_clock_s': k * 10., 'end_clock_s': k * 10. + 4.,
                'actual_class_id': k % 2, 'valid': True} for k in range(3)]
            events = [{'arm': a, 'emitted_local_clock_s': t, 'class_id': p} for a in ARM_NAMES
                for t, p in [(1., 0), (2., 0), (11., 0)]]
            result = evaluate(attempts, [], events, [], 250, 25)
            m = result['arms']['C']
            self.assertEqual((m['correct'], m['wrong'], m['misses'], m['duplicate_outputs']), (1, 1, 1, 1))
            self.assertAlmostEqual(m['attempt_success_rate'], 1 / 3)
            rest = [{'start_clock_s': 30., 'end_clock_s': 90., 'duration_s': 60., 'valid': True}]
            self.assertIsNone(evaluate([], rest, [], [], 250, 25)['arms']['C']['rest_false_events_per_minute'])

        def test_stale_data_never_emits(self):
            t = np.arange(2000) / 250.
            amplitude = np.where(t < 3, 1., 20.)
            signal = amplitude[:, None] * np.column_stack([np.sin(2 * np.pi * f * t) for f in [35, 41, 48, 60]])
            e = ComparisonEngine(bundle(), params(), max_data_age=.2)
            e.activate(setup())
            records = e.process(signal, t, np.zeros(len(t), bool), lambda row: t[row] + 2.)
            self.assertEqual(e.events, [])
            self.assertTrue(all(not r['timely'] for r in records))

        def test_threshold_selection_ignores_prediction_correctness(self):
            records = []
            attempts = []
            for i in range(380):
                stamp = i / 10.
                active = any(start <= stamp < start + 3 for start in [8, 13, 18, 23, 28, 33])
                value = 6. if active else 1. + .03 * math.sin(i)
                records.append({'decision_row': i * 25, 'segment_id': 1, 'processed_local_clock_s': stamp,
                    'usable': True, 'timely': True, 'tke_usable': True, 'rms': np.ones(4) * value,
                    'predicted_class_id': 0, 'scores': np.array([1., -1.])})
            for i, start in enumerate([8, 13, 18, 23, 28, 33]):
                attempts.append({'start_clock_s': float(start), 'end_clock_s': float(start + 3), 'actual_class_id': i % 2})
            rest = {'start_clock_s': 0., 'end_clock_s': 7.}
            p1, _ = choose_thresholds(records, rest, attempts, 'rms', np.ones(4), params())
            flipped = copy.deepcopy(records)
            for r in flipped:
                r['predicted_class_id'] = 1
                r['scores'] = -r['scores']
            p2, _ = choose_thresholds(flipped, rest, attempts, 'rms', np.ones(4), params())
            self.assertEqual(p1['on_threshold'], p2['on_threshold'])
            self.assertEqual(p1['off_threshold'], p2['off_threshold'])

        def test_saved_raw_replay_end_to_end(self):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data = root / 'run'
                data.mkdir()
                rng = np.random.default_rng(32)
                X = np.r_[rng.normal(size=(30, 20)), rng.normal(size=(30, 20)) + 2]
                pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='linear', break_ties=True, decision_function_shape='ovr'))])
                pipe.fit(X, np.r_[np.zeros(30, int), np.ones(30, int)])
                b = {'schema': DAILY_SCHEMA, 'artifact_role': 'daily_calibrated', 'pipeline': pipe,
                    'preprocess_config': config(), 'feature_names': FEATURE_NAMES, 'class_ids': [0, 1],
                    'class_names': NAMES, 'subject_id': 'SYNTHETIC', 'calibration_date': '2026-11-18',
                    'calibration_selection_runs': [1, 2], 'independent_test_runs_not_used_for_fit': [3, 4],
                    'calibration_applied': False, 'training': {'today_runs_used_for_fit': []}, 'today_hashes': {}}
                model = root / 'daily_model.joblib'
                joblib.dump(b, model)
                p = params()
                p['model_sha256'] = file_sha256(model)
                validate_detector(b, model, p)
                s = setup()
                s['activation_row'] = 500
                s['model_sha256'] = file_sha256(model)
                s['arms']['A']['params'] = p
                s['config_id'] = hashlib.sha256(json.dumps(json_safe({k: v for k, v in s.items() if k != 'config_id'}), sort_keys=True).encode()).hexdigest()
                dump_json(data / 'tke_rms_config.json', s)
                session = {'schema': APP_SCHEMA, 'model_path': str(model), 'model_sha256': file_sha256(model),
                    'original_detector': p, 'preprocess_config': config(), 'max_data_age_seconds': .75}
                dump_json(data / 'session.json', session)
                engine = ComparisonEngine(b, p)
                engine.activate(s)
                exp = Experiment(config())
                exp.frozen = True
                exp.all_test_records = []
                stamps = 100. + np.arange(5000) / 250.
                t = stamps - 100.
                amplitude = np.where((t > 6) & (t < 10), 25., .8)
                signal = amplitude[:, None] * np.column_stack([np.sin(2 * np.pi * f * t) for f in [35, 43, 51, 62]])
                log = Journal(data)
                for start in range(0, len(t), 37):
                    values = signal[start:start + 37]
                    ts = stamps[start:start + 37]
                    flags = np.zeros(len(ts), bool)
                    flags[0] = start in [0, 2997]
                    log.raw(values, ts, ts[-1] + .05, flags, np.zeros(len(ts), bool), np.zeros(len(ts), bool))
                    records = engine.process(values, ts, flags, lambda row: stamps[row] + .06)
                    exp.all_test_records.extend(records)
                    log.windows(records)
                self.assertGreater(len(engine.events), 0)
                log.sync(engine, exp)
                log.close()
                dump_json(data / 'summary.json', {'status': 'synthetic'})
                audit, _, _ = replay_logs(data, root / 'audits')
                self.assertTrue(audit['raw_recalculation_features_and_tke_match'])
                self.assertTrue(audit['events_match_by_arm_id_row_class'])

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print('以上均为合成信号工程自检，不代表真实肌电准确率或板卡同步已验证。')
    return 0 if result.wasSuccessful() else 1


def parser():
    p = argparse.ArgumentParser(description='OpenBCI四通道石头/布：TKE/RMS门控对照，先动作后标注。',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--run', help='现有二分类pipeline_state.json所在目录')
    p.add_argument('--model', help='配套二分类daily_model.joblib；与--detector同时提供')
    p.add_argument('--detector', help='匹配该模型的replay_detector_config.json')
    p.add_argument('--check-only', action='store_true', help='只核查模型和门控，不连接LSL')
    p.add_argument('--list-streams', action='store_true', help='只列出LSL流')
    p.add_argument('--self-test', action='store_true', help='合成信号工程自检，不连接板卡')
    p.add_argument('--replay', help='离线复算本程序保存的原始日志目录，可用--model指定同哈希模型新路径')
    p.add_argument('--stream-name', default='rps_raw')
    p.add_argument('--stream-type', help='可选，限定EMG/EEG等Type，自动忽略前后空白')
    p.add_argument('--source-id', help='重名LSL流时限定source_id')
    p.add_argument('--channels', type=int, nargs=4, default=[1, 2, 3, 4])
    p.add_argument('--counter-channel', type=int, help='仅在流确实包含原始Sample Index时使用')
    p.add_argument('--input-unit', choices=['uv', 'mv', 'v'], default='uv')
    p.add_argument('--rest-seconds', type=float, default=30.)
    p.add_argument('--action-seconds', type=float, default=4.)
    p.add_argument('--max-data-age', type=float, default=.75)
    p.add_argument('--stall-seconds', type=float, default=1.)
    p.add_argument('--timestamp-gap-seconds', type=float, default=.2)
    p.add_argument('--resolve-seconds', type=float, default=3.)
    p.add_argument('--duration', type=float, default=0., help='程序总运行秒数，0表示直到停止')
    p.add_argument('--output', default='RP_BINARY_TKE_COMPARE')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.self_test:
        return self_test()
    if args.list_streams:
        lsl = get_lsl()
        streams = lsl.resolve_streams(wait_time=args.resolve_seconds)
        for stream in streams:
            print(json.dumps(describe_stream(stream), ensure_ascii=False))
        if not streams:
            print('未发现LSL流，请检查GUI Networking是否正在发送TimeSeriesRaw。')
        return 0
    require(len(set(args.channels)) == 4 and min(args.channels) >= 1, '--channels必须是四个不同的正整数。')
    require(args.counter_channel is None or args.counter_channel >= 1 and args.counter_channel not in args.channels,
        '计数器通道必须为独立的正整数通道。')
    require(10 <= args.rest_seconds <= 120 and 2 <= args.action_seconds <= 10, '静息时长需10~120秒，动作时长需2~10秒。')
    require(np.isfinite([args.max_data_age, args.stall_seconds, args.timestamp_gap_seconds, args.resolve_seconds, args.duration]).all()
        and min(args.max_data_age, args.stall_seconds, args.timestamp_gap_seconds, args.resolve_seconds) > 0 and args.duration >= 0, '时间参数必须有限且在有效范围内。')
    if args.replay:
        require(not args.run and not args.detector, '回放用--replay，模型移动时额外提供--model。')
        audit, _, _ = replay_logs(args.replay, args.output, args.model)
        return 0 if audit['raw_recalculation_features_and_tke_match'] and audit['events_match_by_arm_id_row_class'] else 2
    bundle, model_path, original, detector_path = load_inputs(args)
    if args.check_only:
        print(json.dumps(json_safe({'script_version': APP_VERSION, 'model': str(model_path),
            'detector': str(detector_path), 'model_calibration_date': bundle['calibration_date'],
            'fs': bundle['preprocess_config']['fs'], 'window_samples': bundle['preprocess_config']['window_samples'],
            'hop_samples': bundle['preprocess_config']['hop_samples'], 'classes': NAMES,
            'source_ON': original['on_threshold'], 'source_OFF': original['off_threshold'],
            'vote_windows': original['vote_windows'], 'minimum_votes': original['minimum_votes'],
            'new_RMS_TKE_thresholds': 'measured_from_current_session_then_frozen', 'loads_three_class_model': False}),
            ensure_ascii=False, indent=2))
        return 0
    output = run_gui(args, bundle, model_path, original, detector_path)
    summary = output / 'summary.json'
    return 1 if summary.is_file() and read_json(summary).get('status') == 'error' else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as exc:
        print('[未完成] %s' % exc, file=sys.stderr)
        raise SystemExit(1)
