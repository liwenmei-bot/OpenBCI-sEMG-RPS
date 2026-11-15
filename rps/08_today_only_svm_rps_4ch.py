#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenBCI 四通道石头/剪刀/布：纯当天训练 -> 留出测试 -> 连续回放 -> 实时。

Python >=3.9。将本文件与现有01_preprocess_openbci_rps_4ch.py、
07_live_svm_rps_label_4ch.py放在同一目录。不需要02/03/04、基础模型或历史数据。
安装：python -m pip install numpy pandas scipy scikit-learn joblib pylsl

训练（自动预处理、选参数、训练、准备匹配门控、独立连续回放）：
  python 08_today_only_svm_rps_4ch.py train --input "今天的TXT完整路径" --subject USER01
已有01预处理结果：
  python 08_today_only_svm_rps_4ch.py train --today "预处理子目录" --subject USER01
运行最新一次成功训练对应的实时模型：
  python 08_today_only_svm_rps_4ch.py live --output RPS_TODAY_ONLY
只用特征训练和离线验证（不准备门控、不生成可实时启动的latest_run）：
  python 08_today_only_svm_rps_4ch.py train --today "预处理子目录" --train-only

固定方案：同一个人、同一天、同一个Session，4Run，每Run18Trial（三类各6）。
Run1拟合Scaler与SVM；Run2从12组线性/RBF参数中选型；冻结参数后以Run1+2重新拟合。
Run3+4共36Trial只验证，不参与Scaler、SVM或门控拟合/选择，不因错误改QC或删Trial。
按完整Run划分，重叠窗口不跨集合；每Trial对拟合的总权重相同，权重平均值为1。
三类映射固定11=石头、12=剪刀、13=布；只有动作分类，静息由07活动门控处理。
200ms窗/100ms步、20维MAV/RMS/WL/ZC/SSC；沿用01的20~100Hz因果滤波+50Hz陷波。
只训练当天数据。不是增量学习，也不是跨日免校准模型，不保证优于历史+当天训练。
门控在开发用Run1+2上选择；其开发分数不是独立验证结果。门控和模型冻结后才评估Run3+4。
连续回放使用真实时间顺序和07的同一个推理引擎，但不测量LSL/GUI/板卡端到端延迟。
本文件不修改01/07或旧模型。latest_run.json仅在整条训练及回放流程成功后更新。
训练中断不会把不完整结果冒充可用实时配置。每次训练自动创建新输出目录。
v1.0.1：SVC使用默认的无迭代次数上限（max_iter=-1），避免合并Run1+2后被10万次上限中止；
仍把收敛警告视为失败，并核查fit_status_，不会保存未收敛模型。可用--today复用已完成的预处理。
"""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import os
import platform
import subprocess
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
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
except ImportError as exc:
    raise SystemExit("缺少依赖：%s\n运行：python -m pip install numpy pandas scipy scikit-learn joblib" % exc)

VERSION = "1.0.1"
INPUT_SCHEMA = "openbci_rps_4ch_preprocess_v1"
SCHEMA = "openbci_rps_4ch_svm_daily_v1"
FEATURE_NAMES = ["ch%d_%s" % (ch, f) for ch in range(1, 5) for f in ("MAV", "RMS", "WL", "ZC", "SSC")]
META_KEYS = ["y", "marker_code", "run_id", "trial_id", "collection_date", "session_id", "subject_id",
             "split_role", "inner_cv_group", "qc_status", "qc_reasons", "usable_mask", "start_row", "stop_row"]
TRIAL_RULE = "majority_vote_then_mean_ovr_decision_score_then_smaller_class_id"
CLASS_NAMES = ["石头", "剪刀", "布"]


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


def percent(value):
    return "—" if value is None or not np.isfinite(value) else "%.2f%%" % (100 * value)


def validate_day(data, subject=None):
    tr, meta, cfg = data["trials"], data["meta"], data["config"]
    require(tr.collection_date.nunique() == 1 and tr.session_id.nunique() == 1,
            "一次只输入同一天的一个完整Session，不拼接多个日期。")
    if subject is not None:
        require(str(tr.subject_id.iloc[0]) == str(subject), "受试者编号不一致，请检查 --subject。")
    require(set(tr.run_id.astype(int)) == {1, 2, 3, 4}, "需要完整Run1/2/3/4。")
    for run in (1, 2, 3, 4):
        group = tr[tr.run_id.eq(run)]
        require(len(group) == 18 and group.class_id.astype(int).value_counts().to_dict() == {0: 6, 1: 6, 2: 6},
                "Run%d须有18次动作且每类6次。" % run)
    for run in (1, 2):
        use = meta.model_eligible & meta.run_id.eq(run)
        for k in range(3):
            require(meta.loc[use & meta.y.eq(k), "trial_id"].nunique() >= 2,
                    "Run%d类别%d至少需2次动作有合格窗口，先检查QC。" % (run, k))
    require(float(cfg["fs"]) == 250 and int(cfg["window_samples"]) == 50
            and int(cfg["hop_samples"]) == 25, "本路线固定250Hz、200ms窗口、100ms步长。")
    require(int(cfg["core_start_samples"]) == 125 and int(cfg["core_stop_samples"]) == 875,
            "本路线固定动作Marker后0.5~3.5秒为训练核心段，请用01默认核心段重新预处理。")
    require(float(cfg["low_hz"]) == 20 and float(cfg["high_hz"]) == 100
            and float(cfg["notch_hz"]) == 50, "本路线固定20~100Hz带通及50Hz陷波。")
    if data["mapping"].get("gesture_names_supplied_by_user"):
        names = [data["mapping"]["class_id_to_name"][str(i)] for i in range(3)]
        require(names == CLASS_NAMES, "当前项目映射应为11石头、12剪刀、13布；输入名称不一致。")
    # 同一次动作的重叠窗口必须属于同一个Run；不同动作的核心段不能重叠。
    order = tr.sort_values("start_row")
    require((order.end_row.to_numpy(dtype=int) > order.start_row.to_numpy(dtype=int)).all(),
            "Trial起止Marker不完整，不能直接训练。")
    require((order.start_row.to_numpy(dtype=int)[1:] >= order.end_row.to_numpy(dtype=int)[:-1]).all(),
            "动作区间重叠，请检查Marker。")


def classification_metrics(y, predicted):
    """-1为无输出，保留在分母中，避免QC缺失使结果虚高。"""
    y, p = np.asarray(y, dtype=int), np.asarray(predicted, dtype=int)
    require(len(y) == len(p) and np.isin(y, [0, 1, 2]).all()
            and np.isin(p, [-1, 0, 1, 2]).all(), "评分标签无效。")
    cm = confusion_matrix(y, p, labels=[0, 1, 2, -1])[:3, :] if len(y) else np.zeros((3, 4), dtype=int)
    support = cm.sum(axis=1)
    recall = np.divide(np.diag(cm[:, :3]), support, out=np.full(3, np.nan), where=support > 0)
    return json_safe({"n": len(y), "correct": int((y == p).sum()),
        "accuracy": float((y == p).mean()) if len(y) else None,
        "balanced_accuracy": float(np.nanmean(recall)) if len(y) else None,
        "macro_f1": float(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0)) if len(y) else None,
        "coverage": float((p >= 0).mean()) if len(y) else None,
        "no_prediction": int((p < 0).sum()), "support": support, "recall": recall,
        "confusion_columns": CLASS_NAMES + ["无输出"], "confusion_matrix": cm})


def trial_weights(meta):
    counts = meta.groupby("trial_id").trial_id.transform("size").to_numpy(dtype=float)
    weights = 1. / counts
    return weights / weights.mean()


def candidates():
    return ([{"kernel": "linear", "C": c, "gamma": "scale"} for c in (.1, 1., 10.)] +
            [{"kernel": "rbf", "C": c, "gamma": g} for c in (1., 10., 100.) for g in ("scale", .01, .1)])


def fit_pipeline(X, meta, params):
    require(set(meta.y.astype(int)) == {0, 1, 2}, "训练窗口缺少类别。")
    pipe = Pipeline([("scaler", StandardScaler()), ("svm", SVC(
        kernel=params["kernel"], C=params["C"], gamma=params["gamma"],
        # SVC默认不限制迭代次数。合并Run后即使参数不变，也可能需要超过10万次迭代。
        # 仅改变求解预算；核函数、C、gamma、tol及训练数据均保持原有规则。
        decision_function_shape="ovr", break_ties=True, cache_size=256, max_iter=-1))])
    weights = trial_weights(meta)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipe.fit(X, meta.y.to_numpy(dtype=int), scaler__sample_weight=weights, svm__sample_weight=weights)
    if pipe["svm"].fit_status_ != 0:
        raise ConvergenceWarning("SVM求解未成功收敛，已停止；不会保存或部署本次未收敛的模型。")
    require(np.array_equal(pipe.classes_, [0, 1, 2]), "SVM训练类别异常。")
    return pipe


def evaluate_runs(data, pipeline, runs):
    meta = data["meta"].loc[data["meta"].run_id.isin(runs)].copy().reset_index(drop=True)
    X = data["X"][data["meta"].run_id.isin(runs).to_numpy()]
    use = meta.model_eligible.to_numpy()
    p, scores = np.full(len(meta), -1, dtype=int), np.full((len(meta), 3), np.nan)
    if use.any():
        p[use], scores[use] = predict(pipeline, X[use])
    trials = aggregate(meta, p, scores, data["trials"].loc[data["trials"].run_id.isin(runs)])
    trials["predicted_name"] = [CLASS_NAMES[v] if v >= 0 else "无输出" for v in trials.predicted_class_id]
    windows = meta.copy()
    windows["predicted_class_id"] = p
    for k in range(3):
        windows["decision_%d" % k] = scores[:, k]
    return {"trial": classification_metrics(trials.class_id, trials.predicted_class_id),
            "window_eligible": classification_metrics(meta.loc[use, "y"], p[use]),
            "window_all_retained": classification_metrics(meta.y, p)}, trials, windows


def select_params(data, output):
    meta = data["meta"]
    fit = (meta.model_eligible & meta.run_id.eq(1)).to_numpy()
    rows, all_trials, best_key, best = [], [], None, None
    grid = candidates()
    for number, params in enumerate(grid, 1):
        start = time.perf_counter()
        log("候选%d/%d：%s" % (number, len(grid), params), output)
        try:
            pipe = fit_pipeline(data["X"][fit], meta.loc[fit].reset_index(drop=True), params)
        except (ConvergenceWarning, ValueError) as exc:
            rows.append({"candidate_id": number, **params, "status": "failed", "error": str(exc)})
            log("该候选失败：%s" % exc, output)
            continue
        score, trials, _ = evaluate_runs(data, pipe, [2])
        t, w = score["trial"], score["window_eligible"]
        row = {"candidate_id": number, **params, "status": "ok", "error": "",
            "run2_trial_ba": t["balanced_accuracy"], "run2_trial_macro_f1": t["macro_f1"],
            "run2_min_class_recall": min(t["recall"]), "run2_window_ba": w["balanced_accuracy"],
            "run2_trial_coverage": t["coverage"], "seconds": time.perf_counter() - start}
        rows.append(row)
        trials["candidate_id"] = number
        all_trials.append(trials)
        # 完全并列时保留枚举中较早的简单参数；线性核、较小C优先列出。
        rank = (row["run2_trial_ba"], row["run2_min_class_recall"], row["run2_trial_macro_f1"], row["run2_window_ba"])
        if best_key is None or rank > best_key:
            best_key, best = rank, dict(params)
            selected_id = number
        log("  Run2 Trial BA=%s；三类召回=%s。" %
            (percent(t["balanced_accuracy"]), "/".join(percent(v) for v in t["recall"])), output)
    require(best is not None, "全部候选训练失败，请检查日志。")
    frame = pd.DataFrame(rows)
    frame["selected"] = frame.candidate_id.eq(selected_id)
    write_csv(output / "run2_parameter_search.csv", frame)
    write_csv(output / "run2_candidate_trial_predictions.csv", pd.concat(all_trials, ignore_index=True))
    selection = {"params": best, "candidate_id": selected_id,
        "selection_rule": "Run2_all_trial_BA_then_min_class_recall_then_macroF1_then_eligible_window_BA_then_fixed_grid_order",
        "fit_runs": [1], "validation_runs": [2], "test_used_for_selection": False}
    write_json(output / "selected_parameters.json", selection)
    return best, selection, frame


def load_runtime(path):
    path = Path(path).expanduser().resolve()
    require(path.is_file(), "找不到07实时脚本：%s；请放在本文件旁边或用 --live-script 指定。" % path)
    name = "rps_runtime_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "07脚本无法导入。")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for fn in ("load_daily", "validate_detector", "prepare_detector", "load_signal", "run_intervals",
               "collect_frames", "classify_frames", "activity_scores", "detect", "score_trials",
               "candidate_intervals", "events_in_intervals"):
        require(callable(getattr(module, fn, None)), "07缺少接口%s，请使用配套07脚本。" % fn)
    return module


def replay_holdout(data, model_path, detector_path, runtime, output):
    bundle, _ = runtime.load_daily(model_path)
    params = read_json(detector_path)
    runtime.validate_detector(bundle, model_path, params)
    signal = runtime.load_signal(data, "auto")
    bounds = runtime.run_intervals(data)
    frames, features = runtime.collect_frames(signal, data["config"], bounds[4][1])
    frames = runtime.classify_frames(frames, features, bundle["pipeline"], output)
    scores = runtime.activity_scores(features, params["noise_reference_rms_uv"])
    events, trace = runtime.detect(frames, scores, params, keep_trace=True)
    tr = data["trials"].loc[data["trials"].run_id.isin([3, 4])].copy()
    # 从记录开头保留真实滤波/门控状态，不按真值Marker重置，不把Marker传入推理。
    result, metrics = runtime.score_trials(events, tr, float(data["config"]["fs"]), len(signal["raw"]))
    test_events = runtime.events_in_intervals(events, [bounds[3], bounds[4]])
    baselines = runtime.candidate_intervals(data, set(tr.trial_id))
    baseline_events = runtime.events_in_intervals(test_events, baselines)
    outside = [e for e in test_events if not any(int(r.start_row) <= e["decision_row"] < int(r.end_row)
               for r in tr.itertuples(index=False))]
    baseline_seconds = sum(b - a for a, b in baselines) / float(data["config"]["fs"])
    metrics.update({"evaluation_runs": [3, 4], "test_used_for_selection": False,
        "baseline_candidate_triggers": len(baseline_events), "outside_action_marker_triggers": len(outside),
        "candidate_baseline_duration_s": baseline_seconds, "candidate_baselines_verified_rest": False,
        "candidate_baseline_triggers_per_min": len(baseline_events) * 60 / baseline_seconds if baseline_seconds else None,
        "hardware_transport_tested": False, "latency_reference": "action_marker_not_measured_physical_movement_onset",
        "activity_mode": "legacy", "raw_precision": signal["precision"]})
    write_csv(output / "holdout_replay_trials.csv", result)
    event_columns = ["event_id", "decision_row", "class_id", "release_row"]
    write_csv(output / "holdout_replay_events.csv", pd.DataFrame(test_events) if test_events else pd.DataFrame(columns=event_columns))
    frames = pd.concat([frames.reset_index(drop=True), trace.drop(columns=["decision_row"]).reset_index(drop=True)], axis=1)
    held = runtime.frames_in_intervals(frames, [bounds[3], bounds[4]])
    write_csv(output / "holdout_replay_windows.csv", frames.loc[held])
    write_json(output / "holdout_replay_metrics.json", metrics)
    return metrics


def table(headers, rows):
    esc = lambda v: html.escape(str(v))
    return "<table><thead><tr>" + "".join("<th>%s</th>" % esc(v) for v in headers) + "</tr></thead><tbody>" + \
        "".join("<tr>" + "".join("<td>%s</td>" % esc(v) for v in row) + "</tr>" for row in rows) + "</tbody></table>"


def write_report(output, report, search):
    offline, replay = report["offline_holdout"], report.get("continuous_holdout")
    trial = offline["trial"]
    rows = [["动作核心段投票：Trial BA", percent(trial["balanced_accuracy"])],
        ["动作核心段投票：正确/全部保留动作", "%d/%d" % (trial["correct"], trial["n"])],
        ["动作核心段投票：有输出覆盖率", percent(trial["coverage"])],
        ["单个200ms合格窗口BA", percent(offline["window_eligible"]["balanced_accuracy"])]]
    if replay:
        rows.extend([["连续回放：首次输出正确/全部动作", "%d/%d（%s）" %
            (replay["correct_first_triggers"], replay["n_trials_retained"], percent(replay["first_trigger_accuracy_all_trials"]))],
            ["连续回放：首次输出类别错误 / 漏检", "%d / %d" % (replay["wrong_first_class_trials"], replay["missed_trials"])],
            ["连续回放：重复触发次数", replay["duplicate_trigger_count"]],
            ["连续回放：动作Marker外触发数", replay["outside_action_marker_triggers"]],
            ["正确首次输出相对Marker的中位延迟", "%s 秒" % replay["median_correct_trigger_latency_from_marker_s"]]])
    per_class = [[CLASS_NAMES[k], trial["support"][k], percent(trial["recall"][k]),
                  percent(replay["class_recall"][k]) if replay else "未回放"] for k in range(3)]
    mapping = [["11 / 0", "石头"], ["12 / 1", "剪刀"], ["13 / 2", "布"]]
    workflow = [["Run1", "18次，每类6次", "拟合StandardScaler和SVM"],
        ["Run2", "18次，每类6次", "选核函数/C/gamma；与Run1合并重训；开发门控"],
        ["Run3+4", "36次，每类12次", "冻结后做离线和连续回放验证"],
        ["现场实时", "先每类5次，共15次", "仅使用冻结模型；事后确认实际动作，记录漏检"]]
    warnings_text = []
    if trial["coverage"] < 1:
        warnings_text.append("存在无可用窗口的动作，仍保留在分母中并计为未正确。")
    if replay and replay["first_trigger_accuracy_all_trials"] < .8:
        warnings_text.append("连续回放首次正确率低于80%的当前工程参考值，先查看逐类混淆与漏检；可以短时联调，不据此宣称稳定。")
    warnings_text.extend(["一组只有36次独立留出动作，每错1次总体正确率约变化2.78个百分点；不是跨日泛化结论。",
        "动作核心段为Marker后0.5~3.5秒，共3秒。200ms窗、100ms步通常每动作29窗，窗口彼此相关。",
        "门控开发使用Run1+2，开发表现可能偏乐观；只有Run3+4结果视为本次留出验证。",
        "候选基线并未经独立传感器确认，不能把候选基线触发率直接当成可靠静息误触发率。",
        "回放不包含GUI、LSL与板卡延迟；现场仍需检查放松释放、同步及每类实际输出。",
        "三类召回率=该真实类别被正确判出的次数/该真实类别全部动作数；不是精确率。",
        "模型不再读取历史训练文件；重新贴电极或换日期后需重新采集训练。"])
    cm = trial["confusion_matrix"]
    replay_cm = replay["confusion_class0_class1_class2_no_trigger"] if replay else None
    parts = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>纯当天RPS训练与验证</title>",
        "<style>body{font:17px/1.7 system-ui,sans-serif;background:#f3f5f8;color:#192b40;max-width:1120px;margin:40px auto;padding:28px}h1,h2{color:#114e70}table{border-collapse:collapse;width:100%;background:white;margin:18px 0 30px}td,th{padding:12px;border:1px solid #cbd6df;text-align:left}th{background:#e3edf4}.note{background:#fff4d8;padding:18px;border-radius:8px}code{word-break:break-all}</style>",
        "<h1>四通道RPS · 纯当天训练</h1>",
        "<p>受试者：%s；采集日期：%s。历史训练样本：0。算法：StandardScaler + SVM。</p>" %
            (html.escape(report["subject_id"]), html.escape(report["collection_date"])),
        "<h2>本次结果（Run3+4，固定模型和门控）</h2>", table(["指标", "结果"], rows),
        "<h2>逐类结果</h2>", table(["真实动作", "留出次数", "核心段投票召回率", "连续回放首次正确召回率"], per_class),
        "<h2>核心段投票混淆矩阵</h2>", table(["真实\\预测"] + CLASS_NAMES + ["无输出"],
            [[CLASS_NAMES[k]] + cm[k] for k in range(3)])]
    if replay_cm:
        parts += ["<h2>连续回放首次输出混淆矩阵</h2>", table(["真实\\预测"] + CLASS_NAMES + ["未触发"],
            [[CLASS_NAMES[k]] + replay_cm[k] for k in range(3)])]
    parts += ["<h2>采集与数据用途</h2>", table(["阶段", "数量", "用途"], workflow),
        "<h2>处理参数</h2><p>前四通道、250Hz；50Hz陷波+20~100Hz因果带通；MAV、RMS、WL、ZC、SSC，每通道5维，共20维。标准化器与分类器仅在当次拟合集上拟合，Trial总权重相同。</p>",
        "<p>冻结参数：<code>%s</code>。候选只按Run2结果选择，未按测试Run选择。</p>" % html.escape(str(report["selected_params"])),
        "<h2>Marker映射</h2>", table(["Marker / 类别ID", "动作"], mapping),
        "<h2>现场操作</h2><p>电极位置保持一致，GUI输出TimeSeriesRaw、250Hz、LSL名rps_raw；运行脚本的live命令。先放松15秒，观察活动随用力增大、放松减小；每次约保持4秒，释放后再做下一次。输出后确认真实动作，漏检也记录。先每类5次，之后每类10次并增加独立静息观察。若10秒仍未释放或明显错位，先保存日志排查。</p>",
        "<h2>结果的适用范围</h2><div class='note'>" + "".join("<p>%s</p>" % html.escape(s) for s in warnings_text) + "</div>",
        "<h2>主要文件</h2><p>MODEL/daily_model.joblib：模型；LIVE_SETUP内replay_detector_config.json：匹配门控；holdout_trial_predictions.csv：核心段判决；holdout_replay_trials.csv：首次触发、漏检与重复输出；evaluation.json：汇总。旧模型不被覆盖。</p>",
        "<h2>Run2参数候选（开发分数）</h2>", search.to_html(index=False, escape=True), "</html>"]
    (output / "OPEN_REPORT.html").write_text("\n".join(parts), encoding="utf-8")


def script_path(value, default):
    return Path(value).expanduser().resolve() if value else Path(__file__).resolve().with_name(default)


def preprocess_input(args, run_dir):
    if args.today:
        return Path(args.today).expanduser().resolve()
    source = Path(args.input).expanduser().resolve()
    require(source.is_file() and source.suffix.lower() in {".txt", ".csv"}, "--input须为一组正式数据的TXT/CSV完整路径。")
    script = script_path(args.preprocess_script, "01_preprocess_openbci_rps_4ch.py")
    require(script.is_file(), "找不到01预处理脚本：%s" % script)
    parent = run_dir / "PREPROCESS"
    command = [sys.executable, str(script), "--input", str(source), "--subject", args.subject,
               "--split-mode", "none", "--class-names"] + CLASS_NAMES + ["--output", str(parent)]
    log("先预处理这一个TXT。", run_dir)
    subprocess.run(command, check=True)
    found = list(parent.glob("*/rps_windows.npz"))
    require(len(found) == 1, "01未生成唯一预处理子目录，停止以避免误用数据。")
    return found[0].parent


def run_train(args):
    root = Path(args.output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in args.subject)[:60]
    run_dir = root / ("RPS_TODAY_ONLY_%s_%s" % (safe, datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    run_dir.mkdir(exist_ok=False)
    args._output = run_dir
    live_script = script_path(args.live_script, "07_live_svm_rps_label_4ch.py")
    runtime = None if args.train_only else load_runtime(live_script)
    today = preprocess_input(args, run_dir)
    data = load_features(today)
    validate_day(data, args.subject)
    log("只用输入日期%s：Run1拟合、Run2选型；Run3+4最终验证，历史样本0。" %
        data["trials"].collection_date.iloc[0], run_dir)
    params, selection, search = select_params(data, run_dir)
    log("参数已冻结；用Run1+2重新拟合Scaler和SVM（不设迭代次数上限，等待收敛）。", run_dir)
    meta = data["meta"]
    fit = (meta.model_eligible & meta.run_id.isin([1, 2])).to_numpy()
    pipe = fit_pipeline(data["X"][fit], meta.loc[fit].reset_index(drop=True), params)
    solver = {"max_iter": int(pipe["svm"].max_iter), "fit_status": int(pipe["svm"].fit_status_),
              "n_iter": json_safe(getattr(pipe["svm"], "n_iter_", None))}
    log("最终SVM已收敛；求解迭代数：%s。" % solver["n_iter"], run_dir)
    day, subject = str(data["trials"].collection_date.iloc[0]), str(data["trials"].subject_id.iloc[0])
    model_dir = run_dir / "MODEL"
    model_dir.mkdir()
    model_path = model_dir / "daily_model.joblib"
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__}
    mapping = dict(data["mapping"])
    mapping.update(class_id_to_name={str(i): v for i, v in enumerate(CLASS_NAMES)},
                   gesture_names_supplied_by_user=True, mapping_verified_from="project_confirmed_11_rock_12_scissors_13_paper")
    bundle = {"schema": SCHEMA, "script_version": VERSION,
        # daily_calibrated为既有06/07的兼容部署角色；实际训练路线由training_route明确标识。
        "artifact_role": "daily_calibrated", "training_route": "today_only", "pipeline": pipe,
        "feature_names": FEATURE_NAMES, "class_ids": [0, 1, 2], "class_names": CLASS_NAMES,
        "label_mapping": mapping, "preprocess_config": data["config"], "subject_id": subject,
        "selected_params": params, "selected_today_fraction": 1., "calibration_applied": True,
        "calibration_date": day, "training": {"history_dates": [], "history_n_windows": 0,
            "today_runs_used_for_fit": [1, 2], "today_n_windows_used_for_fit": int(fit.sum()),
            "today_n_trials_used_for_fit": int(meta.loc[fit, "trial_id"].nunique()),
            "sample_weight_policy": "equal_total_per_trial_mean_window_weight_one", "solver": solver},
        "calibration_selection_runs": [1, 2], "independent_test_runs_not_used_for_fit": [3, 4],
        "base_model_sha256": None, "history_hashes": {}, "today_hashes": data["hashes"],
        "trial_aggregation_rule": TRIAL_RULE, "rest_class_trained": False,
        "decision_scores_are_probabilities": False, "versions": versions}
    joblib.dump(bundle, model_path, compress=3)
    model_hash = file_sha256(model_path)
    config = {"schema": SCHEMA, "training_route": "today_only", "today_directory": str(data["root"]),
        "today_hashes": data["hashes"], "calibration_fit_runs": [1], "calibration_validation_runs": [2],
        "final_calibration_runs": [1, 2], "independent_test_runs": [3, 4], "history_n_windows": 0,
        "test_used_for_selection": False, "selected_params": params, "solver": solver, "model_sha256": model_hash,
        "live_script": str(live_script) if runtime else None, "versions": versions}
    write_json(model_dir / "calibration_config.json", config)
    write_json(model_dir / "preprocess_config.json", data["config"])
    write_json(model_dir / "label_mapping.json", mapping)
    usage = meta.copy()
    usage["stage_role"] = usage.run_id.map({1: "train_then_final_fit", 2: "validation_then_final_fit",
                                          3: "independent_test", 4: "independent_test"})
    usage["used_for_final_fit"] = fit
    write_csv(run_dir / "window_usage.csv", usage)
    detector_path, detector_hash = None, None
    if runtime:
        log("仅用Run1+2开发匹配门控，不使用测试Run。", run_dir)
        setup = runtime.prepare_detector(argparse.Namespace(model=str(model_path), source_data=str(data["root"]),
            output=str(run_dir / "LIVE_SETUP"), raw_mode="auto", settle_candidates_ms=[0, 200, 400, 600, 800]))
        detector_path = Path(setup) / "replay_detector_config.json"
        detector_hash = file_sha256(detector_path)
    # 直到所有拟合和选择冻结后，才执行留出测试。
    log("模型%s，门控已%s；开始Run3+4独立验证。" %
        ("已冻结", "冻结" if runtime else "跳过（仅训练模式）"), run_dir)
    offline, trials, windows = evaluate_runs(data, pipe, [3, 4])
    write_csv(run_dir / "holdout_trial_predictions.csv", trials)
    write_csv(run_dir / "holdout_window_predictions.csv", windows)
    continuous = replay_holdout(data, model_path, detector_path, runtime, run_dir) if runtime else None
    require(file_sha256(model_path) == model_hash, "测试过程中模型文件发生变化。")
    if runtime:
        require(file_sha256(detector_path) == detector_hash, "测试过程中门控文件发生变化。")
    report = {"schema": "openbci_rps_today_only_evaluation_v1", "script_version": VERSION,
        "subject_id": subject, "collection_date": day, "training_route": "today_only",
        "history_n_windows": 0, "selection": selection, "selected_params": params, "solver": solver,
        "test_used_for_selection": False, "test_used_for_fit": False,
        "n_retained_trials": len(data["trials"]), "n_retained_windows": len(meta),
        "n_fit_windows": int(fit.sum()), "qc_trial_counts": data["trials"].qc_status.value_counts().to_dict(),
        "qc_window_counts": meta.qc_status.value_counts().to_dict(), "offline_holdout": offline,
        "continuous_holdout": continuous, "activity_mode_evaluated": "legacy" if runtime else None,
        "model_sha256": model_hash, "detector_sha256": detector_hash,
        "versions": versions, "hardware_transport_tested": False}
    write_json(run_dir / "evaluation.json", report)
    write_report(run_dir, report, search)
    manifest = {"schema": "openbci_rps_today_only_run_v1", "completed": True, "ready_for_live": bool(runtime),
        "run_directory": str(run_dir), "model": str(model_path), "model_sha256": model_hash,
        "detector": str(detector_path) if detector_path else None, "detector_sha256": detector_hash,
        "live_script": str(live_script) if runtime else None,
        "live_script_sha256": file_sha256(live_script) if runtime else None,
        "collection_date": day, "subject_id": subject, "preprocessed_directory": str(data["root"]),
        "report": str(run_dir / "OPEN_REPORT.html"), "training_route": "today_only"}
    write_json(run_dir / "run_manifest.json", manifest)
    if runtime:
        temp = root / ("latest_run_%s.tmp.json" % run_dir.name)
        write_json(temp, manifest)
        os.replace(temp, root / "latest_run.json")
    log("核心段投票：Run3+4 Trial BA=%s，正确%d/%d。" %
        (percent(offline["trial"]["balanced_accuracy"]), offline["trial"]["correct"], offline["trial"]["n"]), run_dir)
    if continuous:
        log("连续回放：首次正确%d/%d=%s；漏检%d；重复触发%d。" %
            (continuous["correct_first_triggers"], continuous["n_trials_retained"],
             percent(continuous["first_trigger_accuracy_all_trials"]), continuous["missed_trials"],
             continuous["duplicate_trigger_count"]), run_dir)
        log("连续回放逐类召回：" + "；".join("%s=%s" % (n, percent(v))
            for n, v in zip(CLASS_NAMES, continuous["class_recall"])), run_dir)
    log("报告：%s" % (run_dir / "OPEN_REPORT.html"), run_dir)
    if runtime:
        log("下一步：先检查报告，再开启GUI LSL并执行本脚本live命令。", run_dir)
        print('python "%s" live --run "%s"' % (Path(__file__).resolve(), run_dir), flush=True)
    else:
        log("本次为train-only，未生成实时门控，未更新latest_run。", run_dir)
    return manifest


def run_live(args):
    manifest_path = Path(args.run).expanduser().resolve() / "run_manifest.json" if args.run else \
        Path(args.output).expanduser().resolve() / "latest_run.json"
    require(manifest_path.is_file(), "找不到完整训练记录，请先成功运行train。")
    info = read_json(manifest_path)
    require(info.get("schema") == "openbci_rps_today_only_run_v1" and info.get("completed")
            and info.get("ready_for_live"), "这次结果尚未完成门控准备与留出回放。")
    model, detector = Path(info["model"]), Path(info["detector"])
    for path, key in ((model, "model_sha256"), (detector, "detector_sha256")):
        require(path.is_file() and file_sha256(path) == info[key], "模型/门控文件缺失或已变，请重新训练准备。")
    script = script_path(args.live_script or info["live_script"], "07_live_svm_rps_label_4ch.py")
    runtime = load_runtime(script)
    bundle, _ = runtime.load_daily(model)
    runtime.validate_detector(bundle, model, read_json(detector))
    if file_sha256(script) != info["live_script_sha256"]:
        log("07文件已更新：本次现场输出还需重新验证，原报告对应训练时的07版本。")
    if info["collection_date"] != datetime.now().strftime("%Y-%m-%d"):
        log("正在使用%s采集的模型；这仅是同日期模型复用/联调，本日性能需本日数据检验。" % info["collection_date"])
    log("加载纯当天模型%s，与其匹配门控；请保持电极位置一致并先放松15秒。" % info["collection_date"])
    command = [sys.executable, str(script), "--model", str(model), "--detector", str(detector),
               "--stream-name", args.stream_name, "--output", str(Path(info["run_directory"]) / "LIVE_LABEL")]
    parser = runtime.build_live_parser()
    if any(a.dest == "activity_mode" for a in parser._actions):
        command += ["--activity-mode", "legacy"]
    subprocess.run(command, check=True)


def build_parser():
    p = argparse.ArgumentParser(description="四通道RPS纯当天训练与实时启动，不依赖历史模型。",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--version", action="version", version=VERSION)
    sub = p.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="预处理 -> Run1/2选型训练 -> 门控 -> Run3/4独立回放")
    source = train.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="当天一组带Marker的原始TXT完整路径，自动调用01")
    source.add_argument("--today", help="已有01预处理子目录或rps_windows.npz")
    train.add_argument("--subject", default="USER01", help="与01输出受试者编号一致")
    train.add_argument("--output", default="RPS_TODAY_ONLY", help="独立结果父目录，保留全部历次结果")
    train.add_argument("--preprocess-script", help="可选，01预处理脚本完整路径")
    train.add_argument("--live-script", help="可选，07脚本完整路径；训练准备门控与回放也使用它")
    train.add_argument("--train-only", action="store_true", help="只生成模型和离线报告，不需07，不准备实时门控")
    live = sub.add_parser("live", help="启动最新成功训练（或--run指定）所配套的07实时识别")
    live.add_argument("--output", default="RPS_TODAY_ONLY", help="与train的父目录相同")
    live.add_argument("--run", help="指定一次已完成的RPS_TODAY_ONLY_...结果子目录")
    live.add_argument("--live-script", help="可选，替代07路径；更新后须现场重新验证")
    live.add_argument("--stream-name", default="rps_raw")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            run_train(args)
        else:
            run_live(args)
        return 0
    except KeyboardInterrupt:
        print("\n已中断；未完成的训练不会更新latest_run。", file=sys.stderr)
        return 130
    except Exception as exc:
        print("\n[失败] %s" % exc, file=sys.stderr)
        output = getattr(args, "_output", None)
        if output:
            (output / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print("详细错误：%s" % (output / "ERROR.txt"), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
