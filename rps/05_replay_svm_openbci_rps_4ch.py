#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenBCI四通道SVM：连续信号加速回放，因果窗口、活动门控、稳定输出、事后评分。

Python>=3.9；依赖：numpy pandas scipy scikit-learn joblib。
独立.py，不导入前序脚本。
python 05_replay_svm_openbci_rps_4ch.py --model "第一份/daily_model.joblib" --target "第二份预处理目录"

源数据/基础模型路径默认从daily_model旁calibration_config.json读取，可--source-data/--base-model覆盖。
模型与StandardScaler始终冻结。门控只用源Session Run1+2设置：Run1候选基线作噪声参考，
Run1动作核心段作活动参考；Run2选择门限和3窗/5窗稳定判决，然后冻结后回放整个目标记录。
源Run3+4和所有目标数据都不参与门控参数估计或选择。

连续推理只接收前四通道、Sample Index及当前时间戳；不会把Marker/Trial/Run边界送入检测器。
先连续单向SOS滤波，遇断点/非有限数值重置并预热，再取过去200ms窗口、100ms步长。
活动持续且类别投票稳定后输出一次；观察到持续低活动才重新允许输出。质检异常锁定至重新放松。
v1.1：启动后可等待短时间再累积新票，等待长度仅在源Run2选择；低于启动门限时
清空待输出投票，释放等待期间禁止新输出，避免把下降阶段的信号当作新识别结果。
离线按批计算固定SVM的逐窗分数以加速；各窗口分数独立，不使用未来窗口平滑当前输出。
名义样本时间用于决策时序；本程序不连接设备，不宣称测得硬件端到端延迟。

优先从session_audit记录的原TXT重读四路float64并核对SHA；原TXT不在时从保存的X_raw float32回放，
报告注明精度来源。--raw-mode stored可强制从保存的原始数组回放。

候选基线不当作已确认静息，Marker0不当作静息标签。可选--verified-rest-csv提供目标人工确认静息：
start_row,stop_row,verified_rest,note；区间左闭右开、0基，verified_rest=true，不得重叠动作标记。
未提供时，真实静息误触发率为null；仅报告候选基线触发与动作标记外输出。

全部Trial保留，不按识别正确与否改QC。动作主要指标采用Marker起止区间内的第一次输出：
正确首触发/全部Trial、漏检、错分类、重复输出；晚些正确不能抵消首次错误。
Marker延迟是相对提示事件，不是肌肉真实起始延迟。门控选择分数不当作独立测试成绩。
本次目标记录此前用于离线分类评估，连续回放作为固定方案补充验证；后续调整另用新数据验证。
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
SCHEMA = "openbci_rps_4ch_svm_continuous_replay_v1"
DAILY_SCHEMA = "openbci_rps_4ch_svm_daily_v1"
CHANNELS = ["EXG Channel %d" % i for i in range(4)]

INPUT_SCHEMA = "openbci_rps_4ch_preprocess_v1"


BASE_SCHEMA = "openbci_rps_4ch_svm_v1"


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
        require("class_names" in z and z["class_names"].shape == (3,), "缺少三类名称。")
        names = z["class_names"].tolist()
        values = {}
        for key in META_KEYS:
            require(key in z and z[key].ndim == 1 and len(z[key]) == len(X), "字段缺失或长度不符：%s" % key)
            values[key] = z[key].copy()
        require("fs" in z and float(z["fs"].item()) == float(config.get("fs", -1)), "采样率不一致。")
    require(config.get("schema") == INPUT_SCHEMA and config.get("n_channels") == 4, "配置格式/通道数错误。")
    require(config.get("channel_columns") == ["EXG Channel %d" % i for i in range(4)], "必须只使用前四通道。")
    require(config.get("feature_names") == FEATURE_NAMES and config.get("filter_mode") == "causal_sos", "特征或因果滤波配置错误。")
    require(mapping.get("marker_to_class_id") == {"11": 0, "12": 1, "13": 2}, "Marker到类别的映射不匹配。")
    require([mapping.get("class_id_to_name", {}).get(str(i)) for i in range(3)] == names, "类别名称文件不一致。")
    meta = pd.DataFrame(values)
    for key in ("y", "marker_code", "run_id", "start_row", "stop_row"):
        v = pd.to_numeric(meta[key], errors="raise").to_numpy(dtype=float)
        require(np.isfinite(v).all() and np.equal(v, np.floor(v)).all(), "整数字段无效：%s" % key)
        meta[key] = v.astype(np.int64)
    for key in set(META_KEYS) - {"y", "marker_code", "run_id", "start_row", "stop_row", "usable_mask"}:
        meta[key] = meta[key].astype(str)
    require(set(meta.y.unique()) == {0, 1, 2}, "窗口未覆盖三类，或包含未知类别。")
    require(np.array_equal(meta.marker_code.to_numpy(), meta.y.to_numpy() + 11), "Marker与类别编号不一致。")
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
    require(set(trials.class_id.astype(int)).issubset({0, 1, 2}), "Trial存在未知类别。")
    require(np.array_equal(trials.marker_code.to_numpy(dtype=int), trials.class_id.to_numpy(dtype=int) + 11), "Trial Marker与类别不一致。")
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
    scores = np.asarray(pipeline.decision_function(X), dtype=float)
    require(scores.shape == (len(X), 3) and np.isfinite(scores).all(), "SVM决策输出异常。")
    return pipeline.classes_[scores.argmax(axis=1)].astype(np.int64), scores


def percent(value):
    return "—" if value is None or not np.isfinite(value) else "%.2f%%" % (100 * value)


def table(headers, rows):
    esc = lambda value: html.escape(str(value))
    return "<table><thead><tr>" + "".join("<th>" + esc(v) + "</th>" for v in headers) + "</tr></thead><tbody>" + \
        "".join("<tr>" + "".join("<td>" + esc(v) + "</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table>"


def check_pipeline(bundle):
    pipeline = bundle.get("pipeline")
    require(isinstance(pipeline, Pipeline) and isinstance(pipeline.named_steps.get("scaler"), StandardScaler)
            and isinstance(pipeline.named_steps.get("svm"), SVC), "模型必须是StandardScaler+SVC。")
    require(bundle.get("feature_names") == FEATURE_NAMES and bundle.get("class_ids") == [0, 1, 2], "模型特征或类别不匹配。")
    require(np.array_equal(pipeline.classes_, [0, 1, 2]) and pipeline["svm"].break_ties
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


def verify_chronology(source, target):
    sa, ta = audit_for(source), audit_for(target)
    raw_s, raw_t = sa.get("source_sha256"), ta.get("source_sha256")
    if raw_s and raw_t:
        require(raw_s != raw_t, "源和目标是同一份原始数据，不能当作新Session。")
    sf, tf = sa.get("first_host_timestamp"), ta.get("first_host_timestamp")
    span = sa.get("host_timestamp_span_seconds")
    if all(isinstance(v, (int, float)) and np.isfinite(v) for v in (sf, tf, span)):
        require(span >= 0 and tf > sf + span, "目标记录并非源记录结束后的独立记录，请核查时间顺序或文件重叠。")
        return {"verified": True, "method": "session_audit_host_timestamps", "source_first_timestamp": sf,
                "source_end_timestamp": sf + span, "target_first_timestamp": tf,
                "start_to_start_minutes": (tf - sf) / 60, "gap_after_source_end_minutes": (tf - sf - span) / 60,
                "raw_source_hashes_checked": bool(raw_s and raw_t)}
    def clock(data):
        session = str(data["trials"].session_id.iloc[0])
        found = re.search(r"(20\d{2}-\d{2}-\d{2})[_T](\d{2})[-:](\d{2})[-:](\d{2})", session)
        return datetime.strptime(" ".join([found[1], ":".join(found.group(i) for i in (2, 3, 4))]), "%Y-%m-%d %H:%M:%S") if found else None
    first, second = clock(source), clock(target)
    if first is not None and second is not None:
        require(second > first, "目标Session文件名时间未晚于源Session。")
        return {"verified": False, "method": "session_name_start_time_only", "source_start": first.isoformat(),
                "target_start": second.isoformat(), "start_to_start_minutes": (second - first).total_seconds() / 60,
                "note": "仅核查文件名起始时间，缺少完整时间审计，未核实记录是否重叠。"}
    return {"verified": False, "method": "unavailable",
            "note": "Session/文件哈希不同，日期合法；缺少时间审计，未核实具体先后或间隔。"}


def validate_protocol(data):
    tr = data["trials"]
    require(tr.collection_date.nunique() == 1 and tr.session_id.nunique() == 1, "请输入单日期、单Session的预处理目录。")
    require(set(tr.run_id.astype(int)) == {1, 2, 3, 4}, "需要完整4Run，不自动拼接或随机拆分窗口。")
    for run in (1, 2, 3, 4):
        group = tr[tr.run_id.eq(run)]
        require(len(group) == 18 and group.class_id.astype(int).value_counts().to_dict() == {0: 6, 1: 6, 2: 6},
                "Run%d需要18Trial、三类各6，请核查采集完整性。" % run)


def load_case(args):
    fixed, model_path = load_daily(args.model)
    adjacent = model_path.parent / "calibration_config.json"
    config = read_json(adjacent) if adjacent.is_file() else {}
    source_path = args.source_data or config.get("today_directory")
    require(bool(source_path), "找不到源Session路径；保留模型旁calibration_config.json，或显式 --source-data。")
    source, target = load_features(source_path), load_features(args.target)
    validate_protocol(source)
    validate_protocol(target)
    require(source["hashes"] == fixed.get("today_hashes"), "源Session哈希与当天模型不匹配。")
    require(target["hashes"]["rps_windows.npz"] != source["hashes"]["rps_windows.npz"], "目标数据与源数据相同。")
    subject = str(fixed["subject_id"])
    for data in (source, target):
        require(data["trials"].subject_id.iloc[0] == subject, "源模型、源数据和目标数据必须属于同一个人。")
        check_contract(fixed["preprocess_config"], data["config"])
        require(data["mapping"]["marker_to_class_id"] == fixed["label_mapping"]["marker_to_class_id"], "Marker映射不一致。")
        if fixed["label_mapping"].get("gesture_names_supplied_by_user") and data["mapping"].get("gesture_names_supplied_by_user"):
            require(data["mapping"]["class_id_to_name"] == fixed["label_mapping"]["class_id_to_name"], "手势名称映射不一致。")
    source_date, target_date = str(source["trials"].collection_date.iloc[0]), str(target["trials"].collection_date.iloc[0])
    require(source_date == fixed.get("calibration_date"), "模型校准日期与源Session不一致。")
    require(target_date >= source_date, "目标日期早于模型校准日期，不作为后续Session评估。")
    require(target_date not in fixed.get("training", {}).get("history_dates", []), "目标日期已用于历史基础模型训练。")
    require(set(source["trials"].session_id).isdisjoint(set(target["trials"].session_id)),
            "目标与源Session相同，此脚本用于固定模型跨Session复用评估。")
    require(set(source["trials"].trial_id).isdisjoint(set(target["trials"].trial_id)), "源/目标Trial重叠。")
    source_calibration = source["meta"].model_eligible.to_numpy() & source["meta"].run_id.isin([1, 2]).to_numpy()
    require(fixed["training"].get("today_n_windows_used_for_fit") ==
            (int(source_calibration.sum()) if fixed["calibration_applied"] else 0), "源校准窗口数量不匹配。")
    chronology = verify_chronology(source, target)
    models = {"fixed_source": fixed}
    paths = {"fixed_source": model_path}
    base_path = args.base_model or config.get("base_model")
    require(bool(base_path), "找不到历史基础模型；请显式 --base-model，或保留源模型旁calibration_config.json。")
    base_path = Path(base_path).expanduser().resolve()
    require(base_path.is_file() and file_sha256(base_path) == fixed.get("base_model_sha256"), "历史基础模型文件与源当天模型记录不一致。")
    base = joblib.load(base_path)
    require(base.get("schema") == BASE_SCHEMA and base.get("artifact_role") == "deployment_base", "历史基础模型格式错误。")
    check_pipeline(base)
    require(base.get("subject_id") == subject and base.get("source_hashes") == fixed.get("history_hashes"), "历史基础模型的受试者/历史数据不一致。")
    check_contract(fixed["preprocess_config"], base["preprocess_config"])
    models["base"], paths["base"] = base, base_path
    if args.target_daily_model:
        reference, ref_path = load_daily(args.target_daily_model)
        require(reference.get("today_hashes") == target["hashes"], "目标自身校准模型与目标数据哈希不匹配。")
        require(reference.get("subject_id") == subject and reference.get("calibration_date") == target_date,
                "目标校准模型的受试者/日期不一致。")
        require(reference.get("base_model_sha256") == fixed.get("base_model_sha256") and
                reference.get("history_hashes") == fixed.get("history_hashes"), "两个当天模型未使用相同历史基础模型。")
        check_contract(fixed["preprocess_config"], reference["preprocess_config"])
        require(reference.get("class_names") == fixed.get("class_names"), "两个模型的显示类别名称不一致。")
        target_cal = target["meta"].model_eligible.to_numpy() & target["meta"].run_id.isin([1, 2]).to_numpy()
        require(reference["training"].get("today_n_windows_used_for_fit") ==
                (int(target_cal.sum()) if reference["calibration_applied"] else 0), "目标校准模型窗口数量不匹配。")
        models["target_recalibrated"], paths["target_recalibrated"] = reference, ref_path
    return fixed, source, target, models, paths, chronology


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
    scores = np.full((len(frames), 3), np.nan)
    indices = np.flatnonzero(frames.usable.to_numpy() & np.isfinite(features).all(axis=1))
    for lo in range(0, len(indices), 256):
        part = indices[lo:lo + 256]
        predicted[part], scores[part] = predict(pipeline, features[part])
        if output is not None and lo % 2560 == 0:
            log("固定SVM推理：%d/%d 窗。" % (min(lo + len(part), len(indices)), len(indices)), output)
    frames = frames.copy()
    frames["predicted_class_id"] = predicted
    for k in range(3): frames["decision_%d" % k] = scores[:, k]
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
        votes = np.bincount(labels, minlength=3)
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
        decision = np.array([frame.decision_0, frame.decision_1, frame.decision_2])
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
    recalls = [int(correct.get(k, 0)) / int(counts[k]) if int(counts.get(k, 0)) else None for k in range(3)]
    cm = np.zeros((3, 4), dtype=int)
    for r in result.itertuples(index=False): cm[int(r.class_id), int(r.first_predicted_class_id) if r.first_predicted_class_id >= 0 else 3] += 1
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
        "confusion_class0_class1_class2_no_trigger": cm.tolist(),
        "median_correct_trigger_latency_from_marker_s": float(np.median(latencies)) if len(latencies) else None,
        "p90_correct_trigger_latency_from_marker_s": float(np.quantile(latencies, .9)) if len(latencies) else None}
    return result, score


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
            outside = len([e for e in validation_events if not any(int(t.start_row) <= e["decision_row"] < int(t.end_row) for t in run2.itertuples(index=False))])
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


def confirmed_rest(path, target, n_rows):
    if path is None: return []
    rest = pd.read_csv(Path(path).expanduser(), keep_default_na=False, encoding="utf-8-sig")
    require({"start_row", "stop_row", "verified_rest"}.issubset(rest.columns), "静息CSV需要start_row、stop_row、verified_rest。")
    require(rest.verified_rest.astype(str).str.lower().isin(["true", "1"]).all(), "所有提供的静息区间必须明确verified_rest=true。")
    intervals = []
    for r in rest.itertuples(index=False):
        a, b = int(r.start_row), int(r.stop_row)
        require(0 <= a < b <= n_rows, "静息区间越界。")
        require(not any(a < int(t.end_row) and b > int(t.start_row) for t in target["trials"].itertuples(index=False)), "确认静息与动作标记冲突。")
        intervals.append((a, b))
    return merge_intervals(intervals)


def merge_intervals(intervals):
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]: merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else: merged.append((a, b))
    return merged


def exposure(intervals, events, fs):
    intervals = merge_intervals(intervals)
    seconds = sum(b - a for a, b in intervals) / fs
    count = len(events_in_intervals(events, intervals))
    return {"seconds": seconds, "triggers": count, "triggers_per_minute": count * 60 / seconds if seconds > 0 else None}


def timeline_svg(frames, scores, params, events, trials, fs):
    width, height = 1080, 240
    indices = np.linspace(0, len(frames) - 1, min(1500, len(frames)), dtype=int)
    times = frames.decision_row.to_numpy(dtype=float) / fs
    maximum_time = max(float(times[-1]), 1)
    finite = scores[np.isfinite(scores)]
    scale = max(float(np.quantile(finite, .995)) if len(finite) else 1, params["on_threshold"] * 1.2)
    sy = lambda v: height - 35 - min(math.log1p(max(v, 0)) / math.log1p(scale), 1) * (height - 65)
    sx = lambda t: 45 + t / maximum_time * (width - 65)
    parts = ["<svg viewBox='0 0 %d %d' role='img' aria-label='连续活动分数和触发时间线'>" % (width, height),
             "<rect width='100%%' height='100%%' fill='#f7fafc'/>".replace("100%%", "100%")]
    for t in trials.itertuples(index=False):
        a, b = sx(int(t.start_row) / fs), sx(int(t.end_row) / fs)
        parts.append("<rect x='%.2f' y='20' width='%.2f' height='%d' fill='#dfe9f2'/>" % (a, max(1, b - a), height - 55))
    points = " ".join("%.2f,%.2f" % (sx(times[i]), sy(scores[i])) for i in indices if np.isfinite(scores[i]))
    parts.append("<polyline fill='none' stroke='#285e89' stroke-width='1' points='%s'/>" % points)
    for value, label in ((params["on_threshold"], "ON"), (params["off_threshold"], "OFF")):
        y = sy(value)
        parts.append("<line x1='45' y1='%.2f' x2='%d' y2='%.2f' stroke='#b55e27' stroke-dasharray='5 4'/><text x='2' y='%.2f' font-size='11'>%s</text>" % (y, width - 20, y, y, label))
    colors = ["#d35d5d", "#3b8d68", "#755bc2"]
    for e in events:
        parts.append("<circle cx='%.2f' cy='%.2f' r='3' fill='%s'><title>class %d at %.2fs</title></circle>" %
            (sx(e["decision_row"] / fs), sy(e["activity_score"]), colors[e["class_id"]], e["class_id"], e["decision_row"] / fs))
    for fraction in (0, .25, .5, .75, 1):
        t = maximum_time * fraction
        parts.append("<text x='%.2f' y='%d' font-size='11'>%.0fs</text>" % (sx(t), height - 10, t))
    return "".join(parts) + "</svg>"


def write_report(output, result, params, frames, scores, events, trial_scores, names, trials):
    esc = lambda v: html.escape(str(v))
    m = result["event_metrics"]
    blocks = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>SVM连续回放报告</title><style>body{font-family:system-ui,'Microsoft YaHei',sans-serif;max-width:1180px;margin:32px auto;padding:0 22px;line-height:1.7;color:#192c3d}table{border-collapse:collapse;width:100%;margin:16px 0 24px;font-size:14px}th,td{border:1px solid #d6e0e8;padding:9px;text-align:left;overflow-wrap:anywhere}th{background:#e7eff7}tr:nth-child(even){background:#f7f9fc}.box{padding:16px 20px;background:#edf4fb;border-left:4px solid #39729d}svg{width:100%;height:auto}small{color:#526172}</style>",
        "<h1>四通道 SVM · 连续信号回放</h1>",
        "<p>运行用途：<b>%s</b>。%s</p>" %
            ("调试复核" if result["evaluation_purpose"] == "debug" else "新数据验证",
             "此版本在分析既有回放错误后修订；同一目标记录的重跑用于调试，不能作为独立改进证据。" if result["evaluation_purpose"] == "debug" else
             "本次目标由操作者声明为未用于当前检测方案设计和调参的新数据。"),
        "<p>受试者：%s。目标：%s。整条记录按样本顺序处理，门控不接收Marker、Trial或Run边界。</p>" %
            (esc(result["subject_id"]), esc(result["target_session"])),
        "<div class='box'>首次正确触发：<b>%d/%d（%s）</b>；漏检 %d；首次错分类 %d；重复触发 %d。<br>正确触发相对Marker的中位延迟：%s 秒。</div>" %
            (m["correct_first_triggers"], m["n_trials_retained"], percent(m["first_trigger_accuracy_all_trials"]),
             m["missed_trials"], m["wrong_first_class_trials"], m["duplicate_trigger_count"],
             "—" if m["median_correct_trigger_latency_from_marker_s"] is None else "%.3f" % m["median_correct_trigger_latency_from_marker_s"]),
        "<h2>连续时间线</h2>", timeline_svg(frames, scores, params, events, trials, result["fs"]),
        "<p>蓝线是活动分数，虚线是冻结的启动/释放门限，彩点是实际输出，浅色块仅在事后显示动作Marker区间。曲线为对数纵轴、抽样概览，不代表按Marker切片预测。</p>",
        "<h2>冻结的识别规则</h2>", table(["项目", "值"], [["滤波", "连续单向SOS，断点后重置并预热"],
            ["窗口/步长", "%d/%d 点" % (result["window_samples"], result["hop_samples"])],
            ["源噪声RMS参考（uV）", params["noise_reference_rms_uv"]], ["启动/释放门限", [params["on_threshold"], params["off_threshold"]]],
            ["启动/释放连续窗数", [params["on_windows"], params["release_windows"]]],
            ["启动后等待（秒）", params.get("settle_seconds", 0)],
            ["稳定判决", "%d窗内至少%d票，输出后持续低活动才重新允许输出" % (params["vote_windows"], params["minimum_votes"])],
            ["参考活动重叠提示", params["activity_reference_overlap_warning"]]]),
        "<p>原SVM和StandardScaler保持不变。门控参考仅来自第一份Run1，等待时间、门限和稳定参数仅用第一份Run2选择。两者是校准用途，不能作为独立成绩。等待期间不累积类别票，当前活动必须达到启动门限才允许输出；低活动会清空待输出票。分批推理只加速独立窗口计算，稳定判决按时间顺序且只使用过去窗口。</p>",
        "<h2>动作级结果</h2>", table(["指标", "结果"], [["保留Trial", m["n_trials_retained"]],
            ["Marker区间可评分", m["n_marker_intervals_scorable"]], ["首次正确触发率（全部Trial）", percent(m["first_trigger_accuracy_all_trials"])],
            ["首次触发平衡召回率", percent(m["first_trigger_balanced_recall"])], ["未触发", m["missed_trials"]],
            ["首次错分类", m["wrong_first_class_trials"]], ["出现重复的Trial", m["trials_with_duplicate_triggers"]],
            ["额外重复次数", m["duplicate_trigger_count"]], ["正确触发P90 Marker延迟（秒）", m["p90_correct_trigger_latency_from_marker_s"]]]),
        table(["Marker对齐补充诊断", "次数"], [["标记前已有输出且持续进入动作", m["preceding_output_held_into_trial_count"]],
             ["其中标记内没有新输出", m["strict_misses_with_preceding_held_output"]]]),
        table(["真实 / 首次输出"] + names + ["未触发/未核实"], [[names[i]] + row for i, row in enumerate(m["confusion_class0_class1_class2_no_trigger"])]),
        "<p>每个动作以Marker起止区间内第一次输出评分。首次错、后面正确仍算首次错；额外输出计重复。标记前已有输出只作补充诊断，不改判为标记内正确。所有Trial保留。Marker是采集提示事件，延迟不等同于肌肉真实起始延迟；本程序的样本时序延迟也不是硬件端到端延迟。</p>",
        "<h2>静息与标记外输出</h2>", table(["项目", "值"], [["动作标记外输出（未标注，非静息真值）", result["outside_action_marker_triggers"]],
            ["候选基线区间触发", result["candidate_baseline_exposure"]], ["人工确认静息区间", result["verified_rest_exposure"]],
            ["真实静息误触发/分钟", result["verified_rest_false_triggers_per_minute"]]]),
        "<p>动作前基线只是候选，Marker0表示无事件脉冲。未提供人工确认静息时，真实静息误触发率为null；不能把上面的候选或未标注区间输出直接称为真实静息误报。输出目录含verified_rest_template.csv，可填写确认过的目标静息区间后通过--verified-rest-csv补评分。</p>",
        "<h2>各Run</h2>", table(["Run", "首次正确/Trial", "漏检", "首次错", "重复", "Marker中位延迟（秒）"],
            [[run, "%d/%d" % (p["correct_first_triggers"], p["n_trials_retained"]), p["missed_trials"],
              p["wrong_first_class_trials"], p["duplicate_trigger_count"], p["median_correct_trigger_latency_from_marker_s"]]
             for run, p in result["per_run_event_metrics"].items()]),
        "<h2>主要文件</h2>", table(["文件", "用途"], [["replay_detector_config.json", "冻结门限、稳定判决、参考与模型身份，后续在线需沿用。"],
            ["events.csv", "检测器实际输出与释放时间。"], ["trial_event_scores.csv", "所有动作的首输出、漏检、重复和相对Marker延迟。"],
            ["window_trace.csv", "连续逐窗预测、活动分数、状态与QC。"], ["source_gate_validation.csv", "仅源Run2的门控选择结果。"],
            ["evaluation.json", "完整动作级指标、静息标签状态、输入精度和耗时。"]]),
        "<p>输入精度：%s。处理耗时仅是离线计算耗时；未按实际墙钟播放，也未连接设备。对方案作新调整后，应另用新数据确认。</p>" % esc(result["input_precision"]),
        "<small>schema: %s；script: %s；原模型文件未覆盖。</small></html>" % (SCHEMA, VERSION)]
    (output / "OPEN_REPORT.html").write_text("\n".join(blocks), encoding="utf-8")


def run(args):
    started = time.perf_counter()
    require(args.chunk_samples > 0, "chunk_samples必须为正数。")
    fixed, source, target, models, model_paths, chronology = load_case(args)
    model_path = model_paths["fixed_source"]
    before_hash = file_sha256(model_path)
    subject = str(fixed["subject_id"])
    safe_subject = "".join(c if c.isalnum() or c in "_-" else "_" for c in subject)[:60]
    output = Path(args.output).expanduser().resolve() / ("RPS_SVM_REPLAY_%s_%s" % (safe_subject, datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    output.mkdir(parents=True, exist_ok=False); args._output = output
    config, fs = fixed["preprocess_config"], float(fixed["preprocess_config"]["fs"])
    log("读取源校准信号，只生成源Run1+2及之前的因果窗口。", output)
    source_signal = load_signal(source, args.raw_mode)
    source_stop = run_intervals(source)[2][1]
    source_frames, source_features = collect_frames(source_signal, config, source_stop, args.chunk_samples)
    source_frames = classify_frames(source_frames, source_features, fixed["pipeline"], output)
    params = choose_detector(source, source_frames, source_features, fs, len(source_signal["raw"]), output,
                             args.settle_candidates_ms)
    params.update({"schema": SCHEMA, "script_version": VERSION, "model_sha256": before_hash,
        "source_hashes": source["hashes"], "source_calibration_runs": [1, 2], "target_used_for_selection": False,
        "window_samples": config["window_samples"], "hop_samples": config["hop_samples"],
        "class_names": fixed["class_names"], "preprocess_config": config})
    write_json(output / "replay_detector_config.json", params)
    log("v%s 门控已冻结：ON=%.3f，OFF=%.3f，等待%.3f秒，%d窗至少%d票。" %
        (VERSION, params["on_threshold"], params["off_threshold"], params["settle_seconds"],
         params["vote_windows"], params["minimum_votes"]), output)
    log("按采样顺序回放目标整条信号，不向检测器提供Marker。", output)
    target_signal = load_signal(target, args.raw_mode)
    frames, features = collect_frames(target_signal, config, None, args.chunk_samples)
    frames = classify_frames(frames, features, fixed["pipeline"], output)
    activity = activity_scores(features, params["noise_reference_rms_uv"])
    events, trace = detect(frames, activity, params, keep_trace=True)
    trial_scores, event_metrics = score_trials(events, target["trials"], fs, len(target_signal["raw"]))
    action_intervals = [(int(t.start_row), int(t.end_row)) for t in target["trials"].itertuples(index=False)
                        if 0 <= int(t.start_row) < int(t.end_row) <= len(target_signal["raw"])]
    outside = [e for e in events if not any(a <= e["decision_row"] < b for a, b in action_intervals)]
    candidates = candidate_intervals(target)
    rests = confirmed_rest(args.verified_rest_csv, target, len(target_signal["raw"]))
    names = list(fixed["class_names"])
    event_frame = pd.DataFrame(events, columns=["event_id", "decision_row", "onset_candidate_row", "class_id",
        "vote_agreement", "activity_score", "release_row", "release_reason"])
    if len(event_frame):
        event_frame["decision_time_s"] = event_frame.decision_row / fs
        event_frame["class_name"] = [names[int(v)] for v in event_frame.class_id]
        event_frame["release_time_s"] = pd.to_numeric(event_frame.release_row, errors="coerce") / fs
    write_csv(output / "events.csv", event_frame)
    write_csv(output / "trial_event_scores.csv", trial_scores)
    window_trace = frames.copy()
    for column in trace.columns:
        if column != "decision_row": window_trace[column] = trace[column]
    window_trace["decision_time_s"] = window_trace.decision_row / fs
    write_csv(output / "window_trace.csv", window_trace)
    write_csv(output / "verified_rest_template.csv", pd.DataFrame(columns=["start_row", "stop_row", "verified_rest", "note"]))
    per_run = {}
    for run_id in (1, 2, 3, 4):
        _, pack = score_trials(events, target["trials"][target["trials"].run_id.eq(run_id)], fs, len(target_signal["raw"]))
        per_run[str(run_id)] = pack
    require(file_sha256(model_path) == before_hash, "源模型在回放过程中改变。")
    verified_exposure = exposure(rests, events, fs) if rests else None
    result = {"schema": SCHEMA, "script_version": VERSION, "subject_id": subject,
        "source_session": str(source["trials"].session_id.iloc[0]), "target_session": str(target["trials"].session_id.iloc[0]),
        "fs": fs, "window_samples": config["window_samples"], "hop_samples": config["hop_samples"],
        "event_metrics": event_metrics, "per_run_event_metrics": per_run, "trigger_count": len(events),
        "outside_action_marker_triggers": len(outside), "candidate_baseline_exposure": exposure(candidates, events, fs),
        "verified_rest_exposure": verified_exposure,
        "verified_rest_false_triggers_per_minute": verified_exposure["triggers_per_minute"] if verified_exposure else None,
        "baseline_reference_verified_rest": False, "detector_parameters_frozen_before_target": True,
        "evaluation_purpose": args.evaluation_purpose,
        "independent_improvement_validation_claimed": args.evaluation_purpose == "new-data-check",
        "marker_used_as_detector_input": False, "target_used_for_selection": False,
        "svm_or_scaler_refitted": False, "model_sha256_unchanged": True,
        "n_continuous_windows": len(frames), "window_qc_counts": frames.qc_status.value_counts().to_dict(),
        "source_signal_break_count": len(source_signal["issues"]), "target_signal_break_count": len(target_signal["issues"]),
        "input_precision": {"source": source_signal["precision"], "target": target_signal["precision"]},
        "chronology": chronology, "processing_elapsed_seconds": time.perf_counter() - started,
        "latency_reference": "nominal_sample_time_relative_to_marker_not_physiological_or_hardware_latency",
        "output_directory": str(output), "versions": {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__}}
    write_json(output / "evaluation.json", result)
    write_report(output, result, params, frames, activity, events, trial_scores, names, target["trials"])
    log("连续回放：首触发正确%d/%d（%s），漏检%d，首次错%d，重复%d，标记外输出%d。" %
        (event_metrics["correct_first_triggers"], event_metrics["n_trials_retained"], percent(event_metrics["first_trigger_accuracy_all_trials"]),
         event_metrics["missed_trials"], event_metrics["wrong_first_class_trials"], event_metrics["duplicate_trigger_count"], len(outside)), output)
    log("正确首触发相对Marker中位延迟：%s秒。" % event_metrics["median_correct_trigger_latency_from_marker_s"], output)
    log("标记前已有输出延续进入动作：%d次；其中标记内无新输出%d次（主评分不豁免）。" %
        (event_metrics["preceding_output_held_into_trial_count"], event_metrics["strict_misses_with_preceding_held_output"]), output)
    if args.evaluation_purpose == "debug":
        log("本次为调试复核；修订方案的独立提升须另用未参与设计的新数据验证。", output)
    log("未确认静息时，仅报告候选基线触发和标记外输出；真实静息误触发率保留为空。", output)
    log("完成。报告：%s" % (output / "OPEN_REPORT.html"), output)
    return result


def build_parser():
    p = argparse.ArgumentParser(description="四通道SVM连续信号加速回放，源校准设置门控，目标只评分。",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.set_defaults(target_daily_model=None)
    p.add_argument("--model", required=True, help="固定第一份daily_model.joblib")
    p.add_argument("--target", required=True, help="第二份完整预处理目录，包含sessions连续信号")
    p.add_argument("--source-data", help="第一份完整预处理目录，默认从模型旁calibration_config.json读取")
    p.add_argument("--base-model", help="历史基础模型，用于身份核查，默认从同一calibration_config.json读取")
    p.add_argument("--raw-mode", choices=["auto", "stored", "txt"], default="auto")
    p.add_argument("--chunk-samples", type=int, default=1000, help="离线读取分块，不改变因果窗口时间")
    p.add_argument("--settle-candidates-ms", nargs="+", type=float, default=[0, 200, 400, 600, 800],
                   help="启动后等待时间候选，只用源Run2选择；等待期间不积累类别票")
    p.add_argument("--evaluation-purpose", choices=["debug", "new-data-check"], default="debug",
                   help="现有目标重跑用debug；未参与当前方案设计的新数据才可声明new-data-check")
    p.add_argument("--verified-rest-csv", help="可选：人工确认的目标静息区间CSV，否则不报告真实静息误报率")
    p.add_argument("--output", default="RPS_SVM_REPLAY", help="输出父目录，每次创建独立子目录")
    return p


def main():
    args = build_parser().parse_args()
    args.target_daily_model = None
    try:
        run(args)
    except Exception as exc:
        output = getattr(args, "_output", None)
        if output is not None: (output / "FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("\n[失败] %s" % exc, file=sys.stderr, flush=True)
        if output is not None: print("详细信息：%s" % (output / "FAILED.txt"), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__": main()
