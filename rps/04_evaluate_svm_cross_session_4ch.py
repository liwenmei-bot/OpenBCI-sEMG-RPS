#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenBCI四通道RPS：固定第一份当天模型，评估后续Session，无任何训练。

Python>=3.9；安装：python -m pip install numpy pandas scipy scikit-learn joblib
本文件独立运行，无须导入01/02/03脚本。

示例：
python 04_evaluate_svm_cross_session_4ch.py --model "第一份/daily_model.joblib" --target "第二份预处理目录"

--source-data 可显式指定第一份预处理目录；默认从模型旁calibration_config.json读取。
--base-model 可提供原历史base_model.joblib；默认从同一calibration_config.json读取。
--target-daily-model 可选，提供第二份自行校准的模型，作为Run3+4的同批动作对照。
该对照模型已使用第二份Run1+2，因此不会对这些校准Run报告独立成绩。

固定模型和历史模型在第二份全部4Run上评估，并单列Run3+4共36Trial供同批比较。
检查第一份模型与源数据哈希、受试者、特征契约、Session和原始数据SHA，核查时间顺序。
不拟合StandardScaler/SVM、不选择权重、不更新模型。标签只用于评分。
WARNING保留；REVIEW/非有限特征/预处理不可用只标记，不删除Trial或窗口。
Trial采用多数票，票数相同按平均OVR分数，再相同取较小类别，与前序脚本一致。
此处是已分段动作的固定模型复用评估；不计算连续静息误报或实时触发延迟。
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import platform
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

try:
    import joblib
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
except ImportError as exc:
    raise SystemExit("缺少依赖：%s\n请运行：python -m pip install numpy pandas scipy scikit-learn joblib" % exc)

VERSION = "1.0.0"
SCHEMA = "openbci_rps_4ch_svm_fixed_session_v1"
DAILY_SCHEMA = "openbci_rps_4ch_svm_daily_v1"

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


def aggregate(meta, predicted, scores, manifest):
    grouped = meta.groupby("trial_id", sort=False).indices
    out = manifest.copy().reset_index(drop=True)
    rows = []
    for trial in out.itertuples(index=False):
        indices = np.asarray(grouped.get(trial.trial_id, []), dtype=int)
        valid = indices[predicted[indices] >= 0]
        votes = np.bincount(predicted[valid], minlength=3) if len(valid) else np.zeros(3, dtype=int)
        means = scores[valid].mean(axis=0) if len(valid) else np.full(3, np.nan)
        tied = np.flatnonzero(votes == votes.max())
        label = int(tied[np.argmax(means[tied])]) if len(valid) else -1
        rows.append({"n_windows": len(indices), "n_predicted_windows": len(valid), "predicted_class_id": label,
                     "has_prediction": label >= 0, "correct": label == int(trial.class_id),
                     **{"votes_%d" % k: int(votes[k]) for k in range(3)},
                     **{"mean_decision_%d" % k: float(means[k]) for k in range(3)}})
    return pd.concat([out, pd.DataFrame(rows)], axis=1)


def metrics(y, predicted):
    y, predicted = np.asarray(y, dtype=int), np.asarray(predicted, dtype=int)
    cm = confusion_matrix(y, predicted, labels=[0, 1, 2]) if len(y) else np.zeros((3, 3), dtype=int)
    support = cm.sum(axis=1)
    recall = np.divide(cm.diagonal(), support, out=np.full(3, np.nan), where=support > 0)
    return json_safe({"n": len(y), "correct": int(np.sum(y == predicted)),
        "accuracy": float(np.mean(y == predicted)) if len(y) else None,
        "balanced_accuracy": float(np.nanmean(recall)) if len(y) else None,
        "macro_f1": float(f1_score(y, predicted, labels=[0, 1, 2], average="macro", zero_division=0)) if len(y) else None,
        "support": support, "recall": recall, "confusion_matrix": cm})


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


def assess(meta, trial_result, predictions, runs):
    chosen = meta.run_id.isin(runs).to_numpy()
    valid = chosen & (predictions >= 0)
    trials = trial_result[trial_result.run_id.isin(runs)]
    covered = trials[trials.has_prediction]
    wm = metrics(meta.loc[valid, "y"], predictions[valid])
    tm = metrics(covered.class_id, covered.predicted_class_id)
    wm.update(n_total=int(chosen.sum()), coverage=float(valid.sum() / chosen.sum()) if chosen.any() else None)
    tm.update(n_total=len(trials), coverage=len(covered) / len(trials) if len(trials) else None,
              strict_accuracy_all_retained_trials=int(covered.correct.sum()) / len(trials) if len(trials) else None)
    return {"window": wm, "trial": tm}


def write_report(output, result, names, per_run):
    esc = lambda value: html.escape(str(value))
    labels = {"base": "历史基础模型", "fixed_source": "第一份校准模型（固定复用）", "target_recalibrated": "第二份自身重新校准模型（参考）"}
    fixed_all = result["all_target_runs"]["fixed_source"]
    blocks = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>固定SVM模型跨Session复用报告</title><style>body{font-family:system-ui,'Microsoft YaHei',sans-serif;max-width:1160px;margin:32px auto;padding:0 22px;color:#182b3c;line-height:1.7}table{border-collapse:collapse;width:100%;margin:16px 0 24px;font-size:14px}td,th{border:1px solid #d7e0e8;padding:10px;text-align:left;overflow-wrap:anywhere}th{background:#e7eff7}tr:nth-child(even){background:#f7f9fc}.box{padding:18px;background:#edf4fb;border-left:4px solid #39749c}small{color:#526273}</style>",
        "<h1>固定SVM模型 · 跨Session复用</h1>",
        "<p>受试者：%s。源Session：%s；目标Session：%s。</p>" %
            (esc(result["subject_id"]), esc(result["source_session"]), esc(result["target_session"])),
        "<div class='box'>第一份模型直接识别第二份全部动作：Trial BA <b>%s</b>，正确 <b>%d/%d</b>；窗口 BA <b>%s</b>。<br>预测覆盖 %d/%d Trial；全部保留Trial的严格准确率 %s。</div>" %
            (percent(fixed_all["trial"]["balanced_accuracy"]), fixed_all["trial"]["correct"], fixed_all["trial"]["n"],
             percent(fixed_all["window"]["balanced_accuracy"]), fixed_all["trial"]["n"], fixed_all["trial"]["n_total"],
             percent(fixed_all["trial"]["strict_accuracy_all_retained_trials"])),
        "<h2>模型保持冻结</h2><p>本次只调用已保存Pipeline的推理，不拟合标准化、SVM或校准权重，不使用目标标签更新模型。目标标签只用于计算指标。所有模型文件哈希在运行前后保持一致。</p>",
        "<p>时间顺序：%s。</p>" % esc(result["chronology"]),
        "<p>评估范围：%s。两次记录之间是否重新贴电极、改变姿势或接线，需要结合采集记录说明；脚本不能从特征文件推断。</p>" % esc(result["scope"]),
        "<h2>第二份全部Run1–4（72次动作）</h2>",
        table(["模型", "Trial覆盖", "Trial正确", "Trial准确率", "Trial BA", "Trial Macro-F1", "窗口 BA"],
              [[labels[key], "%d/%d" % (p["trial"]["n"], p["trial"]["n_total"]), p["trial"]["correct"],
                percent(p["trial"]["accuracy"]), percent(p["trial"]["balanced_accuracy"]), percent(p["trial"]["macro_f1"]),
                percent(p["window"]["balanced_accuracy"])] for key, p in result["all_target_runs"].items()]),
        "<h2>相同Run3+4（36次动作）比较</h2>",
        table(["模型", "Trial覆盖", "Trial正确", "Trial准确率", "Trial BA", "窗口 BA"],
              [[labels[key], "%d/%d" % (p["trial"]["n"], p["trial"]["n_total"]), p["trial"]["correct"],
                percent(p["trial"]["accuracy"]), percent(p["trial"]["balanced_accuracy"]),
                percent(p["window"]["balanced_accuracy"])] for key, p in result["shared_run3_4"].items()]),
        "<p>第二份自行校准的模型已使用本份Run1+2，所以只在Run3+4作为参考。不要把第一份模型在72次动作上的百分比与第二份模型在36次动作上的百分比直接当作同批比较。本报告不按测试分数挑选模型或调整任何参数。</p>",
        "<h2>各Run结果</h2>", table(["模型", "Run", "Trial覆盖", "Trial正确", "Trial BA", "窗口 BA"],
            [[labels[r.model], r.run_id, "%d/%d" % (r.trial_n_predicted, r.trial_n_total), r.trial_correct,
              percent(r.trial_ba), percent(r.window_ba)] for r in per_run.itertuples(index=False)]),
        "<h2>第一份固定模型混淆矩阵（全部目标动作）</h2><p>行是真实类别，列是预测类别。</p>",
        table(["真实 / 预测"] + names, [[names[i]] + row for i, row in enumerate(fixed_all["trial"]["confusion_matrix"])]),
        "<h2>记录保留和质量标记</h2>", table(["项目", "值"], [["目标Trial保留", result["n_trials_retained"]],
            ["目标窗口保留", result["n_windows_retained"]], ["不参与预测窗口", result["n_windows_not_eligible"]],
            ["窗口QC", result["window_qc_counts"]], ["Trial QC", result["trial_qc_counts"]]]),
        "<p>WARNING继续预测。REVIEW、非有限特征和预处理不可用只作资格标记，全部Trial/窗口仍保存在清单。标准指标针对有预测记录，覆盖率单列；严格Trial准确率将未预测动作计入分母。不根据分类对错改QC。</p>",
        "<h2>输出与下一步</h2>", table(["文件", "用途"], [["evaluation.json", "固定模型的整体结果及相同Run3+4比较。"],
            ["per_run_metrics.csv", "各模型在各Run的结果，目标自身校准模型仅列Run3+4。"],
            ["trial_predictions.csv / window_predictions.csv", "全部目标记录、质量标记和各模型预测。"],
            ["evaluation_config.json", "源/目标哈希、时间顺序、冻结模型身份和软件版本。"]]),
        "<p>这是已分段动作的复用补充评估。Trial结果是约3秒核心段多个窗口投票，不能解释为200ms识别率、连续静息误报率或实时触发延迟。后续需要固定方案做连续回放：用源校准Run确定活动门控和稳定判决，在未参与设置的后续数据上统计动作触发、静息误报和重复触发。</p>",
        "<p>本次不产生新模型。11/12/13手势名称沿用采集配置；Marker0不代表静息真值；SVM decision score不是概率。对方案作新调整后，应另用新数据验证。</p>",
        "<small>schema: %s；script: %s</small></html>" % (SCHEMA, VERSION)]
    (output / "OPEN_REPORT.html").write_text("\n".join(blocks), encoding="utf-8")


def run(args):
    log("核查固定模型、源/目标Session、数据哈希与时间顺序……")
    fixed, source, target, models, paths, chronology = load_case(args)
    before = {name: file_sha256(path) for name, path in paths.items()}
    meta, manifest, X = target["meta"], target["trials"], target["X"]
    subject = str(fixed["subject_id"])
    safe_subject = "".join(c if c.isalnum() or c in "_-" else "_" for c in subject)[:60]
    output = Path(args.output).expanduser().resolve() / ("RPS_SVM_FIXED_%s_%s" %
        (safe_subject, datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    output.mkdir(parents=True, exist_ok=False)
    args._output = output
    names = list(fixed["class_names"])
    log("目标%dTrial / %d窗；第一份模型与标准化参数保持冻结。" % (len(manifest), len(meta)), output)
    config = {"schema": SCHEMA, "script_version": VERSION, "model_files": {k: str(v) for k, v in paths.items()},
        "model_sha256": before, "source_directory": str(source["root"]), "target_directory": str(target["root"]),
        "source_hashes": source["hashes"], "target_hashes": target["hashes"], "chronology": chronology,
        "feature_names": FEATURE_NAMES, "class_names": names, "trial_aggregation_rule": TRIAL_RULE,
        "fixed_and_base_evaluation_runs": [1, 2, 3, 4], "target_self_calibrated_reference_runs": [3, 4],
        "no_fit_performed": True, "target_used_to_adapt_fixed_model": False, "test_used_for_model_selection": False,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "command_line": sys.argv}
    write_json(output / "evaluation_config.json", config)
    write_json(output / "preprocess_config.json", fixed["preprocess_config"])
    write_json(output / "label_mapping.json", fixed["label_mapping"])
    write_csv(output / "target_trial_manifest.csv", manifest)
    write_csv(output / "target_window_usage.csv", meta)
    window_output, trial_output = meta.copy(), manifest.copy().reset_index(drop=True)
    trial_output["class_name"] = [names[int(y)] for y in trial_output.class_id]
    results_all, results_shared, per_run, predicted_all = {}, {}, [], {}
    eligible = meta.model_eligible.to_numpy()
    for name, model in models.items():
        runs = [3, 4] if name == "target_recalibrated" else [1, 2, 3, 4]
        mask = eligible & meta.run_id.isin(runs).to_numpy()
        predictions, scores = np.full(len(meta), -1, dtype=np.int64), np.full((len(meta), 3), np.nan)
        if mask.any():
            predictions[mask], scores[mask] = predict(model["pipeline"], X[mask])
        trial_result = aggregate(meta, predictions, scores, manifest)
        predicted_all[name] = predictions
        if name != "target_recalibrated":
            results_all[name] = assess(meta, trial_result, predictions, [1, 2, 3, 4])
        results_shared[name] = assess(meta, trial_result, predictions, [3, 4])
        window_output[name + "_predicted_class_id"] = predictions
        window_output[name + "_predicted_name"] = [names[v] if v >= 0 else "" for v in predictions]
        window_output[name + "_has_prediction"] = predictions >= 0
        window_output[name + "_correct"] = (predictions >= 0) & (predictions == meta.y.to_numpy())
        window_output[name + "_no_prediction_reason"] = ["" if p >= 0 else
            ("excluded_source_calibration_runs" if name == "target_recalibrated" and run not in (3, 4)
             else reason or "no_eligible_features") for p, run, reason in zip(predictions, meta.run_id, meta.not_used_reason)]
        for k in range(3):
            window_output["%s_decision_%d" % (name, k)] = scores[:, k]
        for key in ("predicted_class_id", "has_prediction", "correct", "n_predicted_windows", "votes_0", "votes_1", "votes_2"):
            trial_output[name + "_" + key] = trial_result[key]
        trial_output[name + "_predicted_name"] = [names[int(v)] if v >= 0 else "" for v in trial_result.predicted_class_id]
        for run_id in runs:
            pack = assess(meta, trial_result, predictions, [run_id])
            per_run.append({"model": name, "run_id": run_id, "trial_n_predicted": pack["trial"]["n"],
                "trial_n_total": pack["trial"]["n_total"], "trial_correct": pack["trial"]["correct"],
                "trial_accuracy": pack["trial"]["accuracy"], "trial_ba": pack["trial"]["balanced_accuracy"],
                "trial_macro_f1": pack["trial"]["macro_f1"], "trial_strict_accuracy": pack["trial"]["strict_accuracy_all_retained_trials"],
                "window_ba": pack["window"]["balanced_accuracy"], "window_accuracy": pack["window"]["accuracy"]})
        pack = results_shared[name]
        log("%s：相同Run3+4 Trial BA=%s，正确%d/%d。" %
            (name, percent(pack["trial"]["balanced_accuracy"]), pack["trial"]["correct"], pack["trial"]["n"]), output)
    write_csv(output / "window_predictions.csv", window_output)
    write_csv(output / "trial_predictions.csv", trial_output)
    per_run_frame = pd.DataFrame(per_run)
    write_csv(output / "per_run_metrics.csv", per_run_frame)
    after = {name: file_sha256(path) for name, path in paths.items()}
    require(before == after, "模型文件在评估期间发生变化，本次不作为冻结模型结果。")
    source_date = str(source["trials"].collection_date.iloc[0])
    target_date = str(manifest.collection_date.iloc[0])
    result = {"schema": SCHEMA, "script_version": VERSION, "subject_id": subject,
        "source_session": str(source["trials"].session_id.iloc[0]), "target_session": str(manifest.session_id.iloc[0]),
        "source_date": source_date, "target_date": target_date,
        "scope": "同一人同日跨Session固定模型复用" if source_date == target_date else "同一人跨日跨Session固定模型复用",
        "chronology": chronology, "all_target_runs": results_all, "shared_run3_4": results_shared,
        "n_trials_retained": len(manifest), "n_windows_retained": len(meta), "n_windows_not_eligible": int((~eligible).sum()),
        "window_qc_counts": meta.qc_status.value_counts().to_dict(), "trial_qc_counts": manifest.qc_status.value_counts().to_dict(),
        "no_fit_performed": True, "target_used_to_adapt_fixed_model": False, "test_used_for_model_selection": False,
        "all_model_hashes_unchanged": before == after, "new_model_saved": False,
        "source_model_selected_today_fraction": fixed.get("selected_today_fraction"),
        "output_directory": str(output), "versions": config["versions"]}
    write_json(output / "evaluation.json", result)
    write_report(output, result, names, per_run_frame)
    pack = results_all["fixed_source"]
    log("固定第一份模型 → 第二份全部动作：Trial BA=%s，窗口 BA=%s，正确%d/%d。" %
        (percent(pack["trial"]["balanced_accuracy"]), percent(pack["window"]["balanced_accuracy"]),
         pack["trial"]["correct"], pack["trial"]["n"]), output)
    log("完成。报告：%s" % (output / "OPEN_REPORT.html"), output)
    return result


def build_parser():
    p = argparse.ArgumentParser(description="固定第一份SVM当天模型，评估后续Session；不训练、不重新校准。",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", required=True, help="第一份daily_model.joblib")
    p.add_argument("--target", required=True, help="第二份预处理子目录或rps_windows.npz")
    p.add_argument("--source-data", help="第一份预处理目录，默认从模型旁calibration_config.json读取")
    p.add_argument("--base-model", help="原历史基础模型，默认从同一calibration_config.json读取")
    p.add_argument("--target-daily-model", help="可选：第二份自行校准的daily_model，仅在Run3+4作同批参考")
    p.add_argument("--output", default="RPS_SVM_CROSS_SESSION", help="输出父目录；每次新建子目录")
    return p


def main():
    args = build_parser().parse_args()
    try:
        run(args)
    except Exception as exc:
        output = getattr(args, "_output", None)
        if output is not None:
            (output / "FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("\n[失败] %s" % exc, file=sys.stderr, flush=True)
        if output is not None:
            print("详细信息：%s" % (output / "FAILED.txt"), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
