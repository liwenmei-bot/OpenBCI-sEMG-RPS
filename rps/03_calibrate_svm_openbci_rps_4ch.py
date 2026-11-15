#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenBCI 四通道 RPS SVM：新一天校准及独立测试（独立 .py，无需导入01/02）。

Python >= 3.9。安装：python -m pip install numpy pandas scipy scikit-learn joblib

先用01预处理新一天单个TXT，建议 --split-mode none，subject与历史数据相同。
    python 01_preprocess_openbci_rps_4ch.py --input "新一天.txt" --subject USER01 --split-mode none
再运行本脚本：
    python 03_calibrate_svm_openbci_rps_4ch.py --base-model "训练输出/base_model.joblib" --today "新一天预处理子目录"
历史数据路径自动从基础模型旁的 training_config.json 读取；文件搬迁后可显式 --history 指定。

固定协议：单个人、一个新日期、一个Session、4个Run、每Run18Trial且三类各6。
Run1：校准内部拟合；Run2：选择校准权重；选择冻结后用Run1+2重训当天模型。
Run3+4：独立测试，共36Trial；分别评估基础模型和当天模型，不据此再次选权重。
不采用预处理文件中的 development/test 角色，本脚本明确按上述完整Run重分配。
历史数据必须与基础模型训练时的文件哈希一致，新增日期必须晚于基础模型的所有训练日。

SVM不做partial_fit。本脚本保持基础模型的核、C、gamma等设置不变，将历史特征与
校准特征合并，加权重训StandardScaler与SVC。默认今天训练权重总占比候选为25/50/75%。
样本权重归一到平均1，标准化与SVM使用相同权重；不使用测试Run拟合任何参数。
候选中保留原基础模型作为不校准方案；内部验证并列时优先选择较低的今天权重。

WARNING保留；全部Trial/窗口保留清单；仅按预处理usable_mask、非有限特征、REVIEW
标记拟合资格。不根据分类错误删除数据或改QC。原基础模型与历史文件不被覆盖。
majority vote + 平均OVR decision score破平局，与02脚本相同；分数不视为概率。
当前仍是三类动作模型；实时连续识别的活动检测、静息门控和重复触发需后续验证。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import platform
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

try:
    import joblib
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    from sklearn.base import clone
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
except ImportError as exc:
    raise SystemExit("缺少依赖：%s\n请运行：python -m pip install numpy pandas scipy scikit-learn joblib" % exc)

VERSION = "1.0.0"
INPUT_SCHEMA = "openbci_rps_4ch_preprocess_v1"
BASE_SCHEMA = "openbci_rps_4ch_svm_v1"
SCHEMA = "openbci_rps_4ch_svm_daily_v1"
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


def load_base_and_data(args):
    model_path = Path(args.base_model).expanduser().resolve()
    require(model_path.is_file(), "找不到基础模型：%s" % model_path)
    base = joblib.load(model_path)
    require(isinstance(base, dict) and base.get("schema") == BASE_SCHEMA, "基础模型格式不匹配。")
    require(base.get("artifact_role") == "deployment_base", "请使用02生成的 base_model.joblib 部署基础模型。")
    require(base.get("feature_names") == FEATURE_NAMES and base.get("class_ids") == [0, 1, 2], "模型特征/类别不匹配。")
    pipeline = base.get("pipeline")
    require(isinstance(pipeline, Pipeline) and isinstance(pipeline.named_steps.get("scaler"), StandardScaler)
            and isinstance(pipeline.named_steps.get("svm"), SVC), "模型必须是 StandardScaler + SVC Pipeline。")
    require(np.array_equal(pipeline.classes_, [0, 1, 2]) and pipeline["svm"].break_ties
            and pipeline["svm"].decision_function_shape == "ovr", "模型的类别/决策规则不匹配。")
    history_path = args.history
    if not history_path:
        config_path = model_path.parent / "training_config.json"
        require(config_path.is_file(), "找不到历史路径；请保留基础模型旁的training_config.json，或显式提供 --history。")
        history_path = read_json(config_path).get("source_directory")
        require(bool(history_path), "training_config没有历史路径，请用 --history 显式指定。")
    history, today = load_features(history_path), load_features(args.today)
    require(base.get("source_hashes") == history["hashes"], "历史文件哈希与基础模型训练时不一致，停止混用数据。")
    require(base.get("subject_id") == history["trials"].subject_id.iloc[0] == today["trials"].subject_id.iloc[0],
            "基础模型、历史数据与当天数据的受试者编号必须一致。")
    check_contract(base["preprocess_config"], history["config"])
    check_contract(base["preprocess_config"], today["config"])
    require(base["label_mapping"].get("marker_to_class_id") == today["mapping"].get("marker_to_class_id"), "类别编号映射不一致。")
    if base["label_mapping"].get("gesture_names_supplied_by_user") and today["mapping"].get("gesture_names_supplied_by_user"):
        require(base["label_mapping"]["class_id_to_name"] == today["mapping"]["class_id_to_name"], "当天手势名称与基础模型不一致。")
    hm = history["meta"]
    history_mask = hm.model_eligible.to_numpy() & hm.split_role.isin(["development", "test"]).to_numpy()
    used = hm.loc[history_mask]
    training = base.get("training", {})
    require(len(used) == training.get("n_windows") and used.trial_id.nunique() == training.get("n_trials"), "历史参与训练数量不匹配。")
    require(sorted(used.collection_date.unique().tolist()) == training.get("dates") and
            sorted(used.session_id.unique().tolist()) == training.get("sessions"), "历史训练日期/Session不匹配。")
    dates = today["trials"].collection_date.astype(str).unique().tolist()
    require(len(dates) == 1 and today["trials"].session_id.nunique() == 1, "当天输入必须是单个日期、单个Session的4Run数据。")
    require(dates[0] > max(training["dates"]),
            "当天日期%s已包含在基础模型训练范围内或早于历史末日%s；请采集新一天数据。" % (dates[0], max(training["dates"])))
    require(set(today["trials"].trial_id).isdisjoint(set(history["trials"].trial_id)), "历史与当天Trial重叠。")
    require(set(today["trials"].session_id).isdisjoint(set(history["trials"].session_id)), "历史与当天Session重叠。")
    require(set(today["trials"].run_id.astype(int)) == {1, 2, 3, 4}, "当天需要完整Run1/2/3/4，不自动拼接或随机拆窗口。")
    for run in range(1, 5):
        part = today["trials"][today["trials"].run_id == run]
        require(len(part) == 18 and part.class_id.astype(int).value_counts().to_dict() == {0: 6, 1: 6, 2: 6},
                "当天Run%d必须18个Trial且三类各6；请核查Marker/采集完整性。" % run)
        mask = today["meta"].model_eligible.to_numpy() & today["meta"].run_id.eq(run).to_numpy()
        require(set(today["meta"].loc[mask, "y"]) == {0, 1, 2}, "当天Run%d合格窗口未覆盖三类，请先核查QC。" % run)
    return base, history, today, history_mask, model_path


def weights_for_pool(n_history, n_today, fraction):
    require(n_history > 0 and n_today > 0 and 0 < fraction < 1, "训练池和权重占比无效。")
    total = n_history + n_today
    return np.r_[np.full(n_history, (1 - fraction) * total / n_history),
                 np.full(n_today, fraction * total / n_today)]


def fit_weighted(base_pipeline, history_X, history_y, today_X, today_y, fraction):
    pipeline = clone(base_pipeline)
    X = np.concatenate([history_X, today_X])
    y = np.concatenate([history_y, today_y])
    weights = weights_for_pool(len(history_X), len(today_X), fraction)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(X, y, scaler__sample_weight=weights, svm__sample_weight=weights)
    require(np.array_equal(pipeline.classes_, [0, 1, 2]), "校准模型类别异常。")
    return pipeline


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


def assess_test(meta, trial_result, predicted):
    test = meta.run_id.isin([3, 4]).to_numpy()
    valid = test & (predicted >= 0)
    trials = trial_result[trial_result.run_id.isin([3, 4])]
    covered = trials[trials.has_prediction]
    wm, tm = metrics(meta.loc[valid, "y"], predicted[valid]), metrics(covered.class_id, covered.predicted_class_id)
    wm.update(n_total=int(test.sum()), coverage=float(valid.sum() / test.sum()))
    tm.update(n_total=len(trials), coverage=len(covered) / len(trials), strict_accuracy_all_retained_trials=int(covered.correct.sum()) / len(trials))
    return {"window": wm, "trial": tm}


def percent(value):
    return "—" if value is None or not np.isfinite(value) else "%.2f%%" % (100 * value)


def table(headers, rows):
    esc = lambda value: html.escape(str(value))
    return "<table><thead><tr>" + "".join("<th>" + esc(v) + "</th>" for v in headers) + "</tr></thead><tbody>" + \
        "".join("<tr>" + "".join("<td>" + esc(v) + "</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table>"


def write_report(output, result, validation, names):
    base, daily = result["base_test"], result["daily_test"]
    blocks = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>四通道 SVM 当天校准报告</title><style>body{font-family:system-ui,'Microsoft YaHei',sans-serif;max-width:1120px;margin:32px auto;padding:0 22px;color:#182838;line-height:1.7}table{border-collapse:collapse;width:100%;margin:16px 0 24px;font-size:14px}th,td{border:1px solid #d8e0e8;text-align:left;padding:10px;overflow-wrap:anywhere}th{background:#e8f0f7}tr:nth-child(even){background:#f7f9fc}.box{padding:16px 20px;background:#edf4fb;border-left:4px solid #39709a}small{color:#526170}</style>",
        "<h1>四通道 SVM · 当天校准与独立测试</h1>",
        "<p>受试者：%s；新日期：%s。历史最晚训练日：%s。</p>" %
            (html.escape(result["subject_id"]), result["today_date"], result["last_history_date"]),
        "<div class='box'>同一批 Run3+4 测试动作：基础模型 Trial BA <b>%s</b>，当天模型 <b>%s</b>；变化 <b>%+.2f 个百分点</b>。</div>" %
            (percent(base["trial"]["balanced_accuracy"]), percent(daily["trial"]["balanced_accuracy"]), result["delta_trial_ba"] * 100),
        "<h2>固定协议</h2>", table(["阶段", "数据", "用途"], [
            ["内部拟合", "Run1，18次动作", "历史数据 + Run1，拟合加权标准化与SVM。"],
            ["内部验证", "Run2，18次动作", "选择今天训练权重占比；保留不校准候选。"],
            ["最终校准", "Run1+2，36次动作", "冻结选择后重训当天模型。"],
            ["独立测试", "Run3+4，36次动作", "分别测试原基础模型和当天模型，均不再据此调参。"]]),
        "<p>基础模型的核、C、gamma保持不变。今天权重占比指新校准窗口的训练权重总和占训练池的比例；每个候选的样本权重均归一到平均1，并同时用于StandardScaler与SVC。测试Run不参与标准化、权重选择或训练。</p>",
        "<h2>校准内部验证（选参用途）</h2>",
        table(["今天权重占比", "验证Trial BA", "验证Macro-F1", "验证窗口 BA", "选中"],
              [["原基础模型" if r.today_fraction == 0 else percent(r.today_fraction), percent(r.trial_ba),
                percent(r.trial_macro_f1), percent(r.window_ba), "是" if r.selected else ""]
               for r in validation.itertuples(index=False)]),
        "<p>选择规则：验证Trial BA、Trial Macro-F1、窗口 BA依次优先；并列选较低今天权重。内部验证只有一个Run，结果用于校准选择，不能当作独立准确率。</p>",
        "<p>选中今天权重：<b>%s</b>。%s</p>" %
            (percent(result["selected_today_fraction"]), "保持原基础模型，本次未应用加权重训。" if result["selected_today_fraction"] == 0 else "已用历史数据与Run1+2重新拟合当天模型。"),
        "<h2>Run3+4 独立测试</h2>",
        table(["模型", "层级", "有预测/总数", "准确率", "平衡准确率 BA", "Macro-F1"],
              [[label, level, "%d/%d" % (pack[level]["n"], pack[level]["n_total"]),
                percent(pack[level]["accuracy"]), percent(pack[level]["balanced_accuracy"]), percent(pack[level]["macro_f1"])]
               for label, pack in [("基础模型", base), ("当天模型", daily)] for level in ("window", "trial")]),
        "<p>无预测Trial仍在清单。全部保留测试Trial的严格准确率：基础模型 %s，当天模型 %s。窗口相互重叠，不能当作独立实验重复。整次动作多数票结果不能代替实时误触发率或识别延迟。</p>" %
            (percent(base["trial"]["strict_accuracy_all_retained_trials"]), percent(daily["trial"]["strict_accuracy_all_retained_trials"])),
        "<h2>Trial 混淆矩阵</h2><p>行是真实类别，列是预测类别。</p>"]
    for label, pack in [("基础模型", base), ("当天模型", daily)]:
        blocks += ["<h3>%s</h3>" % label, table(["真实 / 预测"] + names,
                    [[names[i]] + row for i, row in enumerate(pack["trial"]["confusion_matrix"])])]
    blocks += ["<h2>质量与记录保留</h2>", table(["项目", "值"], [["今天保留Trial", result["n_today_trials"]],
        ["今天保留窗口", result["n_today_windows"]], ["QC窗口分布", result["qc_window_counts"]],
        ["QC Trial分布", result["qc_trial_counts"]], ["不参与模型的窗口", result["n_today_windows_not_eligible"]]]),
        "<p>WARNING不排除。原始Trial和窗口全部保留；仅根据预处理固定QC与数值有效性标记模型资格。前两个Run是校准数据，预测清单中不会冒充独立测试结果。</p>",
        "<h2>文件与后续</h2>", table(["文件", "用途"], [["daily_model.joblib", "选定的当天模型；训练不包含Run3+4。"],
        ["evaluation.json", "基础模型与当天模型在同一测试集的结果及覆盖率。"],
        ["calibration_validation_scores.csv", "只由Run2得到的候选权重验证分数。"],
        ["trial_predictions.csv / window_predictions.csv", "全部当天记录，区分校准与测试角色。"],
        ["today_window_usage.csv / protocol_trial_manifest.csv", "按完整Run固定的用途和质量标记。"]]),
        "<p>本次没有覆盖基础模型，也没有把测试数据加入当天模型。不根据测试分数重新挑权重或替换方案；调整方案后应使用新的完整Run验证。接下来先做连续数据回放，验证活动门控、静息误报和重复触发，再接实时流。</p>",
        "<p>当前仍是三分类动作模型，没有静息类别。decision score不是概率。11/12/13与手势名称须沿用采集配置，脚本不猜测映射。</p>",
        "<small>schema: %s；script: %s；结果仅对应本次新日期的独立测试。</small></html>" % (SCHEMA, VERSION)]
    (output / "OPEN_REPORT.html").write_text("\n".join(blocks), encoding="utf-8")


def run(args):
    fractions = sorted(set(float(x) for x in args.today_fractions))
    require(len(fractions) > 0 and all(np.isfinite(x) and 0 < x < 1 for x in fractions), "校准权重占比必须在0和1之间。")
    log("读取基础模型、历史特征与新一天数据；核查日期、哈希和预处理参数……")
    base, history, today, history_mask, model_path = load_base_and_data(args)
    tm, trials, X = today["meta"], today["trials"], today["X"]
    hm = history["meta"]
    Xh, yh = history["X"][history_mask], hm.loc[history_mask, "y"].to_numpy()
    eligible = tm.model_eligible.to_numpy()
    fit = eligible & tm.run_id.eq(1).to_numpy()
    val = eligible & tm.run_id.eq(2).to_numpy()
    calibration = eligible & tm.run_id.isin([1, 2]).to_numpy()
    test = eligible & tm.run_id.isin([3, 4]).to_numpy()
    names = list(base["class_names"])
    day = str(trials.collection_date.iloc[0])
    subject = str(base["subject_id"])
    safe_subject = "".join(c if c.isalnum() or c in "_-" else "_" for c in subject)[:60]
    output = Path(args.output).expanduser().resolve() / ("RPS_SVM_DAILY_%s_%s_%s" %
        (safe_subject, day, datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    output.mkdir(parents=True, exist_ok=False)
    args._output = output
    tm["daily_role"] = np.select([tm.run_id.eq(1), tm.run_id.eq(2), tm.run_id.isin([3, 4])],
                                  ["calibration_fit", "calibration_validation", "independent_test"], default="unassigned")
    trials["daily_role"] = np.select([trials.run_id.eq(1), trials.run_id.eq(2), trials.run_id.isin([3, 4])],
                                      ["calibration_fit", "calibration_validation", "independent_test"], default="unassigned")
    trials["class_name"] = [names[int(y)] for y in trials.class_id]
    write_csv(output / "today_window_usage.csv", tm)
    write_csv(output / "protocol_trial_manifest.csv", trials)
    write_json(output / "preprocess_config.json", base["preprocess_config"])
    write_json(output / "label_mapping.json", base["label_mapping"])
    config = {"schema": SCHEMA, "script_version": VERSION, "base_model": str(model_path),
        "base_model_sha256": file_sha256(model_path), "history_directory": str(history["root"]),
        "today_directory": str(today["root"]), "history_hashes": history["hashes"], "today_hashes": today["hashes"],
        "today_fraction_candidates": [0.0] + fractions, "svm_params_frozen": base["selected_params"],
        "calibration_fit_runs": [1], "calibration_validation_runs": [2], "final_calibration_runs": [1, 2],
        "independent_test_runs": [3, 4], "test_used_for_selection": False, "trial_aggregation_rule": TRIAL_RULE,
        "sample_weight_mean": 1.0, "scaler_uses_sample_weight": True,
        "command_line": sys.argv, "versions": {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__}}
    write_json(output / "calibration_config.json", config)
    log("新日期%s；历史%d窗，校准拟合%d窗，校准验证%d窗，独立测试%d窗。" %
        (day, len(Xh), fit.sum(), val.sum(), test.sum()), output)
    selection, validation_trials, best_key, best_fraction = [], [], None, None
    for fraction in [0.0] + fractions:
        start = time.perf_counter()
        log("校准候选：今天训练权重占比 %s。" % percent(fraction), output)
        pipeline = base["pipeline"] if fraction == 0 else fit_weighted(
            base["pipeline"], Xh, yh, X[fit], tm.loc[fit, "y"].to_numpy(), fraction)
        prediction, score = predict(pipeline, X[val])
        vtrials = trials[trials.run_id.eq(2)].copy()
        aggregate_val = aggregate(tm.loc[val].reset_index(drop=True), prediction, score, vtrials)
        covered = aggregate_val[aggregate_val.has_prediction]
        wm = metrics(tm.loc[val, "y"], prediction)
        trm = metrics(covered.class_id, covered.predicted_class_id)
        row = {"today_fraction": fraction, "trial_ba": trm["balanced_accuracy"], "trial_macro_f1": trm["macro_f1"],
               "window_ba": wm["balanced_accuracy"], "trial_n_predicted": len(covered), "trial_n_total": len(vtrials),
               "elapsed_seconds": time.perf_counter() - start}
        selection.append(row)
        aggregate_val["today_fraction"] = fraction
        validation_trials.append(aggregate_val)
        rank = (row["trial_ba"], row["trial_macro_f1"], row["window_ba"], -fraction)
        if best_key is None or rank > best_key:
            best_key, best_fraction = rank, fraction
        log("  Run2验证：Trial BA=%s，窗口 BA=%s。" % (percent(row["trial_ba"]), percent(row["window_ba"])), output)
    selection_frame = pd.DataFrame(selection)
    selection_frame["selected"] = selection_frame.today_fraction.eq(best_fraction)
    write_csv(output / "calibration_validation_scores.csv", selection_frame)
    write_csv(output / "calibration_validation_trial_predictions.csv", pd.concat(validation_trials, ignore_index=True))
    write_json(output / "selected_calibration.json", {"today_fraction": best_fraction,
        "selected_without_test": True, "selection_rule": "trial_BA_then_trial_F1_then_window_BA_then_lower_today_fraction"})
    log("校准选择已冻结：%s；用历史与Run1+2拟合当天模型。" % percent(best_fraction), output)
    daily_pipeline = copy.deepcopy(base["pipeline"]) if best_fraction == 0 else fit_weighted(
        base["pipeline"], Xh, yh, X[calibration], tm.loc[calibration, "y"].to_numpy(), best_fraction)
    bundle = {"schema": SCHEMA, "script_version": VERSION, "artifact_role": "daily_calibrated",
        "pipeline": daily_pipeline, "feature_names": FEATURE_NAMES, "class_ids": [0, 1, 2], "class_names": names,
        "label_mapping": base["label_mapping"], "preprocess_config": base["preprocess_config"], "subject_id": subject,
        "selected_params": base["selected_params"], "selected_today_fraction": best_fraction,
        "calibration_applied": best_fraction > 0, "calibration_date": day,
        "training": {"history_dates": base["training"]["dates"], "history_n_windows": len(Xh),
                     "today_runs_used_for_fit": [1, 2] if best_fraction > 0 else [],
                     "today_n_windows_used_for_fit": int(calibration.sum()) if best_fraction > 0 else 0},
        "calibration_selection_runs": [1, 2], "independent_test_runs_not_used_for_fit": [3, 4],
        "base_model_sha256": config["base_model_sha256"], "history_hashes": history["hashes"],
        "today_hashes": today["hashes"], "trial_aggregation_rule": TRIAL_RULE,
        "rest_class_trained": False, "decision_scores_are_probabilities": False, "versions": config["versions"]}
    joblib.dump(bundle, output / "daily_model.joblib", compress=3)
    # 两个模型在完全相同的独立测试窗口上各前向一次；不据测试结果修改选择。
    log("开始Run3+4独立测试：基础模型与当天模型各预测一次。", output)
    baseline_prediction, baseline_score = predict(base["pipeline"], X[test])
    daily_prediction, daily_score = predict(daily_pipeline, X[test])
    all_predictions, all_scores, trial_results = {}, {}, {}
    for label, predicted, score in [("base", baseline_prediction, baseline_score), ("daily", daily_prediction, daily_score)]:
        p, s = np.full(len(tm), -1, dtype=np.int64), np.full((len(tm), 3), np.nan)
        p[test], s[test] = predicted, score
        all_predictions[label], all_scores[label] = p, s
        trial_results[label] = aggregate(tm, p, s, trials)
    window_output = tm.copy()
    for label in ("base", "daily"):
        p, s = all_predictions[label], all_scores[label]
        window_output[label + "_predicted_class_id"] = p
        window_output[label + "_predicted_name"] = [names[v] if v >= 0 else "" for v in p]
        window_output[label + "_correct"] = (p >= 0) & (p == tm.y.to_numpy())
        for k in range(3):
            window_output["%s_decision_%d" % (label, k)] = s[:, k]
    write_csv(output / "window_predictions.csv", window_output)
    trial_output = trials.copy().reset_index(drop=True)
    for label in ("base", "daily"):
        for key in ("predicted_class_id", "has_prediction", "correct", "n_predicted_windows", "votes_0", "votes_1", "votes_2"):
            trial_output[label + "_" + key] = trial_results[label][key]
        trial_output[label + "_predicted_name"] = [names[int(v)] if v >= 0 else "" for v in trial_results[label].predicted_class_id]
    write_csv(output / "trial_predictions.csv", trial_output)
    base_test = assess_test(tm, trial_results["base"], all_predictions["base"])
    daily_test = assess_test(tm, trial_results["daily"], all_predictions["daily"])
    group_rows = []
    for run_id in (3, 4):
        mask = test & tm.run_id.eq(run_id).to_numpy()
        for label in ("base", "daily"):
            tr = trial_results[label]
            subset = tr[(tr.run_id == run_id) & tr.has_prediction]
            w = metrics(tm.loc[mask, "y"], all_predictions[label][mask])
            t = metrics(subset.class_id, subset.predicted_class_id)
            group_rows.append({"model": label, "run_id": run_id, "trial_n_predicted": len(subset),
                "trial_n_total": int(trials.run_id.eq(run_id).sum()), "trial_ba": t["balanced_accuracy"],
                "trial_accuracy": t["accuracy"], "window_ba": w["balanced_accuracy"], "window_accuracy": w["accuracy"]})
    write_csv(output / "per_test_run_metrics.csv", pd.DataFrame(group_rows))
    result = {"schema": SCHEMA, "script_version": VERSION, "subject_id": subject, "today_date": day,
        "last_history_date": max(base["training"]["dates"]), "selected_today_fraction": best_fraction,
        "base_test": base_test, "daily_test": daily_test,
        "delta_trial_ba": daily_test["trial"]["balanced_accuracy"] - base_test["trial"]["balanced_accuracy"],
        "delta_window_ba": daily_test["window"]["balanced_accuracy"] - base_test["window"]["balanced_accuracy"],
        "test_used_for_selection": False, "test_prediction_passes_per_model": 1,
        "n_today_trials": len(trials), "n_today_windows": len(tm),
        "n_today_windows_not_eligible": int((~eligible).sum()),
        "qc_window_counts": tm.qc_status.value_counts().to_dict(), "qc_trial_counts": trials.qc_status.value_counts().to_dict(),
        "trial_aggregation_rule": TRIAL_RULE, "base_model_overwritten": False, "test_data_added_to_daily_model": False,
        "selected_by_calibration_validation_not_test": True, "output_directory": str(output), "versions": config["versions"]}
    write_json(output / "evaluation.json", result)
    write_report(output, result, selection_frame, names)
    log("独立测试Trial BA：基础模型%s，当天模型%s，变化%+.2f个百分点。" %
        (percent(base_test["trial"]["balanced_accuracy"]), percent(daily_test["trial"]["balanced_accuracy"]),
         result["delta_trial_ba"] * 100), output)
    log("完成。报告：%s" % (output / "OPEN_REPORT.html"), output)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description="四通道SVM新一天校准：前2Run校准，后2Run独立测试。",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-model", required=True, help="02生成的base_model.joblib，旁边保留training_config.json")
    parser.add_argument("--today", required=True, help="新一天单个4Run Session的预处理目录或rps_windows.npz")
    parser.add_argument("--history", help="历史预处理目录；省略时从基础模型旁training_config.json读取")
    parser.add_argument("--today-fractions", nargs="+", type=float, default=[0.25, 0.5, 0.75], help="当天训练权重总占比候选")
    parser.add_argument("--output", default="RPS_SVM_TODAY", help="输出父目录；不覆盖旧结果")
    return parser


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
