#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenBCI 四通道 RPS：分组验证、SVM 选参、独立测试及部署基础模型。

Python >= 3.9。依赖：numpy pandas scipy scikit-learn joblib。
安装：python -m pip install numpy pandas scipy scikit-learn joblib
运行：python 02_train_svm_openbci_rps_4ch.py --data "预处理输出子文件夹"
也可把 --data 指向该子文件夹内的 rps_windows.npz。

与 01_preprocess_openbci_rps_4ch.py 配套，无须修改或导入预处理脚本。
该目录必须同时有 preprocess_config.json、label_mapping.json、trial_manifest.csv。
只读取20维特征，日期、标签、Trial/Run 编号不作为输入特征。

流程：
1. 保持已有 development/test 划分，审核 Trial 与日期/Session/Run 分组。
2. 开发集按日期留一验证；只有一天时按 Session，再不足则按已有 Run 分组。
3. StandardScaler 每折仅拟合训练窗口；比较线性 SVC 和 RBF SVC。
4. 预先固定选参规则：各验证组 Trial 平衡准确率的均值最高。
   并列时依次比较 Trial Macro-F1、窗口平衡准确率、候选原始顺序。
5. 参数冻结后，用全部开发集拟合评估模型，只预测一次独立测试集。
6. 保存 eval_model.joblib；之后默认合并开发/测试的合格窗口另训 base_model.joblib。
   部署版未被独立评估；它的效果须用新一天数据评估。

全部 Trial/窗口均保留在输出清单，WARNING 不排除。
REVIEW、预处理 usable_mask=False、非有限特征、unassigned 仅作不参与拟合标记。
不根据分类对错修改 QC，不重新设计静息标签，也不提供实时动作门控。
Trial 预测：窗口多数票；票数并列时比较这些类别的平均 OVR decision score；
再相同则取较小类别编号。此规则在读测试结果前固定。
decision score 不是概率。Trial 汇总使用整段核心动作，不能当作实时触发延迟。

模型保存为含 sklearn Pipeline 的字典，不依赖本脚本中的自定义类：
    import joblib
    bundle = joblib.load('eval_model.joblib')
    prediction = bundle['pipeline'].predict(X_20_features)
输入必须遵守 bundle['feature_names'] 的顺序和 bundle['preprocess_config']。

只给出数据上的实际运行结果，不预设或保证准确率。
"""
from __future__ import annotations

import argparse
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
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
except ImportError as exc:
    raise SystemExit("缺少依赖：%s\n请运行：python -m pip install numpy pandas scipy scikit-learn joblib" % exc)

VERSION = "1.0.0"
SCHEMA = "openbci_rps_4ch_svm_v1"
INPUT_SCHEMA = "openbci_rps_4ch_preprocess_v1"
FEATURE_NAMES = ["ch%d_%s" % (ch, name) for ch in range(1, 5)
                 for name in ("MAV", "RMS", "WL", "ZC", "SSC")]
CLASSES = np.array([0, 1, 2], dtype=np.int64)
META_KEYS = ["y", "marker_code", "run_id", "trial_id", "collection_date", "session_id",
             "subject_id", "split_role", "inner_cv_group", "qc_status", "qc_reasons",
             "usable_mask", "start_row", "stop_row"]
TRIAL_RULE = "majority_vote_then_mean_ovr_decision_score_then_smaller_class_id"


def log(message, output=None):
    line = "[%s] %s" % (datetime.now().strftime("%H:%M:%S"), message)
    print(line, flush=True)
    if output is not None:
        with (output / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def write_csv(path, frame):
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10g")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def class_set(values):
    return set(np.asarray(values, dtype=int).tolist())


def read_inputs(data_path, class_names=None):
    path = Path(data_path).expanduser().resolve()
    npz_path = path / "rps_windows.npz" if path.is_dir() else path
    require(npz_path.is_file() and npz_path.name == "rps_windows.npz",
            "--data 请指向包含 rps_windows.npz 的预处理子文件夹，或该 NPZ 文件本身。")
    root = npz_path.parent
    for name in ("preprocess_config.json", "label_mapping.json", "trial_manifest.csv"):
        require((root / name).is_file(), "缺少配套文件：%s；请保持预处理输出文件在同一目录。" % name)
    config = json.loads((root / "preprocess_config.json").read_text(encoding="utf-8-sig"))
    mapping = json.loads((root / "label_mapping.json").read_text(encoding="utf-8-sig"))
    with np.load(npz_path, allow_pickle=False) as archive:
        require("schema" in archive and str(archive["schema"].item()) == INPUT_SCHEMA,
                "NPZ 格式不匹配，请使用配套四通道预处理脚本的输出。")
        key = "X_features" if "X_features" in archive else "X"
        require(key in archive, "NPZ 缺少20维特征。")
        X = np.asarray(archive[key], dtype=np.float64)
        require(X.ndim == 2 and X.shape[1] == 20 and X.shape[0] > 0,
                "特征必须为非空 (窗口数,20) 数组，不能把 X_signal 或 rps_segments.npz 用于此脚本。")
        require("feature_names" in archive and archive["feature_names"].tolist() == FEATURE_NAMES,
                "特征名称或顺序不匹配，要求 CH1~CH4，每路 MAV/RMS/WL/ZC/SSC。")
        require("class_names" in archive and archive["class_names"].shape == (3,), "缺少三类名称。")
        names = [str(x) for x in archive["class_names"].tolist()]
        arrays = {}
        for key in META_KEYS:
            require(key in archive, "NPZ 缺少字段：%s" % key)
            value = np.asarray(archive[key])
            require(value.ndim == 1 and len(value) == len(X), "NPZ 字段长度错误：%s" % key)
            arrays[key] = value
        require("fs" in archive, "NPZ 缺少采样率。")
        fs = float(archive["fs"].item())
    require(config.get("schema") == INPUT_SCHEMA, "预处理配置格式不匹配。")
    require(config.get("n_channels") == 4 and config.get("channel_columns") ==
            ["EXG Channel %d" % i for i in range(4)], "配置必须只使用前四个 EXG 通道。")
    require(config.get("feature_names") == FEATURE_NAMES, "配置与特征名称不一致。")
    require(config.get("filter_mode") == "causal_sos", "要求使用配套的单向因果滤波配置。")
    require(np.isfinite(fs) and fs > 0 and abs(float(config.get("fs", -1)) - fs) < 1e-9,
            "NPZ 与配置的采样率不一致。")
    require(mapping.get("marker_to_class_id") == {"11": 0, "12": 1, "13": 2},
            "标签映射不匹配，禁止猜测或交换手势编号。")
    require([mapping.get("class_id_to_name", {}).get(str(i)) for i in range(3)] == names,
            "NPZ 类名与 label_mapping.json 不一致。")
    if class_names:
        names = list(class_names)
        mapping = dict(mapping, class_id_to_name={str(i): name for i, name in enumerate(names)},
                       gesture_names_supplied_by_user=True, names_need_collection_config_verification=False)
    require(len(set(names)) == 3 and all(name.strip() for name in names), "三个手势名称必须非空且互不相同。")

    meta = pd.DataFrame(arrays)
    for key in ("y", "marker_code", "run_id", "start_row", "stop_row"):
        numeric = pd.to_numeric(meta[key], errors="raise").to_numpy(dtype=float)
        require(np.isfinite(numeric).all() and np.equal(numeric, np.floor(numeric)).all(),
                "整数字段含无效值：%s" % key)
        meta[key] = numeric.astype(np.int64)
    require(class_set(meta["y"]) == {0, 1, 2}, "数据必须包含且仅包含类别0/1/2。")
    require(np.array_equal(meta["marker_code"].to_numpy(), meta["y"].to_numpy() + 11), "Marker 与类别编号不一致。")
    require(set(meta["usable_mask"].unique()).issubset({True, False, 0, 1}), "usable_mask 必须是布尔掩码。")
    meta["usable_mask"] = meta["usable_mask"].astype(bool)
    for key in set(META_KEYS) - {"y", "marker_code", "run_id", "start_row", "stop_row", "usable_mask"}:
        meta[key] = meta[key].astype(str)
    require(meta["subject_id"].nunique() == 1, "本脚本按单个人建模，不支持混合多个受试者。")
    require(set(meta["split_role"]).issubset({"development", "test", "unassigned"}), "存在未知数据集角色。")
    require(set(meta["qc_status"]).issubset({"PASS", "WARNING", "REVIEW"}), "存在未知 QC 状态。")
    require((meta["stop_row"] > meta["start_row"]).all(), "存在无效窗口行范围。")
    require(not meta.duplicated(["trial_id", "start_row", "stop_row"]).any(), "存在重复的 Trial 窗口，停止重复计数。")
    for key in ("trial_id", "collection_date", "session_id", "subject_id", "inner_cv_group"):
        require(meta[key].str.strip().ne("").all(), "分组字段不能为空：%s" % key)
    require(meta["collection_date"].str.fullmatch(r"20\d{2}-\d{2}-\d{2}").all(), "采集日期格式错误。")

    trials = pd.read_csv(root / "trial_manifest.csv", encoding="utf-8-sig", keep_default_na=False,
                         dtype={"trial_id": str, "subject_id": str, "session_id": str,
                                "collection_date": str, "split_role": str, "inner_cv_group": str})
    needed = ["trial_id", "class_id", "marker_code", "run_id", "subject_id", "session_id",
              "collection_date", "split_role", "inner_cv_group", "qc_status"]
    require(set(needed).issubset(trials.columns), "trial_manifest.csv 缺少必要字段。")
    require(not trials["trial_id"].duplicated().any(), "Trial 清单中存在重复编号。")
    require(len(trials) > 0 and set(meta["trial_id"]).issubset(set(trials["trial_id"])), "窗口与 Trial 清单对应失败。")
    require(trials["subject_id"].nunique() == 1 and
            trials["subject_id"].iloc[0] == meta["subject_id"].iloc[0], "Trial 清单受试者不一致。")
    require(set(trials["split_role"]).issubset({"development", "test", "unassigned"}), "Trial 清单存在未知角色。")
    require(set(trials["qc_status"]).issubset({"PASS", "WARNING", "REVIEW"}), "Trial 清单存在未知 QC 状态。")
    by_id = trials.set_index("trial_id")
    matched = by_id.loc[meta["trial_id"].tolist()].reset_index(drop=True)
    for key in ("marker_code", "run_id", "subject_id", "session_id", "collection_date", "split_role", "inner_cv_group"):
        require(np.array_equal(meta[key].astype(str).to_numpy(), matched[key].astype(str).to_numpy()),
                "窗口与 Trial 清单字段不一致：%s" % key)
    require(np.array_equal(meta["y"].to_numpy(), matched["class_id"].to_numpy(dtype=int)), "Trial 类别与窗口类别不一致。")
    require(class_set(trials["class_id"]).issubset({0, 1, 2}), "Trial 清单类别异常。")
    require(np.array_equal(trials["marker_code"].to_numpy(dtype=int), trials["class_id"].to_numpy(dtype=int) + 11),
            "Trial 清单 Marker 与类别编号不一致。")
    trials["class_name"] = [names[int(y)] for y in trials["class_id"]]

    # 先核查所有清单，再按预处理时已固定的规则标记拟合资格。
    assigned = meta["split_role"].isin(["development", "test"]).to_numpy()
    finite = np.isfinite(X).all(axis=1)
    eligible = assigned & finite & meta["usable_mask"].to_numpy() & meta["qc_status"].ne("REVIEW").to_numpy()
    meta["finite_features"] = finite
    meta["eligible_for_model"] = eligible
    reasons = []
    for i in range(len(meta)):
        bits = []
        if not assigned[i]: bits.append("unassigned_role")
        if not finite[i]: bits.append("nonfinite_features")
        if not bool(meta["usable_mask"].iloc[i]): bits.append("preprocess_usable_false")
        if meta["qc_status"].iloc[i] == "REVIEW": bits.append("qc_review")
        reasons.append(";".join(bits))
    meta["not_used_reason"] = reasons

    split_mode = config.get("split", {}).get("mode")
    require(split_mode in {"by-date", "by-session", "within-session"},
            "预处理必须提供开发/测试划分；当前 split-mode=%s，请重新预处理。" % split_mode)
    dev_trials = trials[trials["split_role"] == "development"]
    test_trials = trials[trials["split_role"] == "test"]
    require(len(dev_trials) > 0 and len(test_trials) > 0, "必须同时有开发集和独立测试集。")
    require(set(dev_trials["inner_cv_group"]).isdisjoint(set(test_trials["inner_cv_group"])),
            "开发/测试 inner_cv_group 重叠，存在分组泄漏。")
    if split_mode in {"by-date", "by-session"}:
        key = "collection_date" if split_mode == "by-date" else "session_id"
        require(set(dev_trials[key]).isdisjoint(set(test_trials[key])), "开发/测试 %s 重叠，停止训练。" % key)
    sources = config.get("split", {})  # 配置的 Trial 数与清单也应保持一致。
    expected_counts = sources.get("trial_counts", {})
    actual_counts = trials["split_role"].value_counts().to_dict()
    require(all(int(actual_counts.get(k, 0)) == int(v) for k, v in expected_counts.items()),
            "配置中的 Trial 划分计数与清单不一致。")
    for role in ("development", "test"):
        mask = eligible & meta["split_role"].eq(role).to_numpy()
        require(class_set(meta.loc[mask, "y"]) == {0, 1, 2},
                "%s 合格窗口不足三类，请先核查 QC/Marker；不自动改划分或降低 QC 要求。" % role)
    hashes = {name: file_sha256(root / name) for name in
              ("rps_windows.npz", "preprocess_config.json", "label_mapping.json", "trial_manifest.csv")}
    return {"X": X, "meta": meta, "trials": trials, "config": config, "mapping": mapping,
            "class_names": names, "subject_id": str(meta["subject_id"].iloc[0]),
            "root": root, "source_hashes": hashes}


def make_folds(data):
    meta = data["meta"]
    eligible = meta["eligible_for_model"].to_numpy()
    dev = eligible & meta["split_role"].eq("development").to_numpy()
    active = meta.loc[dev]
    if active["collection_date"].nunique() >= 2:
        level = "collection_date"
    elif active["session_id"].nunique() >= 2:
        level = "session_id"
    else:
        level = "inner_cv_group"
    groups = meta[level].astype(str).to_numpy()
    values = sorted(set(groups[dev]))
    require(len(values) >= 2, "开发集不足两个独立日期/Session/Run，无法分组验证；不使用随机窗口拆分。")
    folds, covered = [], np.zeros(len(meta), dtype=int)
    for number, group in enumerate(values, 1):
        fit = np.flatnonzero(dev & (groups != group))
        val = np.flatnonzero(dev & (groups == group))
        require(len(fit) and len(val), "分组验证产生空折。")
        require(class_set(meta.iloc[fit]["y"]) == {0, 1, 2} and
                class_set(meta.iloc[val]["y"]) == {0, 1, 2},
                "验证组 %s 的训练/验证未各自覆盖三类，请补充数据或核查 QC。" % group)
        require(set(meta.iloc[fit]["trial_id"]).isdisjoint(set(meta.iloc[val]["trial_id"])), "同 Trial 跨训练/验证。")
        covered[val] += 1
        folds.append({"fold": number, "held_out_group": group, "train_indices": fit,
                      "validation_indices": val, "group_level": level})
    require(np.all(covered[dev] == 1) and np.all(covered[~dev] == 0), "验证覆盖异常或包含测试数据。")
    return folds


def build_candidates(args):
    candidates = []
    c_values = list(dict.fromkeys(float(x) for x in args.c_values))
    require(all(np.isfinite(c) and c > 0 for c in c_values), "C 必须为有限正数。")
    gamma_values = []
    for value in args.gamma_values:
        if value in ("scale", "auto"):
            gamma = value
        else:
            gamma = float(value)
            require(np.isfinite(gamma) and gamma > 0, "gamma 必须为 scale/auto 或有限正数。")
        if gamma not in gamma_values:
            gamma_values.append(gamma)
    for kernel in ("linear", "rbf"):
        for c in c_values:
            for gamma in (["scale"] if kernel == "linear" else gamma_values):
                candidates.append({"candidate_id": "SVM_%02d" % (len(candidates) + 1),
                                   "kernel": kernel, "C": c, "gamma": gamma})
    return candidates


def new_pipeline(params, args):
    # 不启用 probability=True：不引入窗口随机拆分的内部概率校准。
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel=params["kernel"], C=params["C"], gamma=params["gamma"],
                    class_weight="balanced", probability=False, decision_function_shape="ovr",
                    break_ties=True, cache_size=args.cache_mb, max_iter=-1, random_state=args.seed))
    ])


def fit_pipeline(pipeline, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(X, y)
    require(np.array_equal(pipeline.classes_, CLASSES), "模型类别顺序异常。")
    return pipeline


def predict_pipeline(pipeline, X):
    # break_ties=True 时，OVR decision score 的 argmax 与 SVC.predict 一致。
    # 只求一次 decision_function，避免重复计算测试集的核矩阵。
    score = np.asarray(pipeline.decision_function(X), dtype=float)
    require(score.shape == (len(X), 3) and np.isfinite(score).all(), "SVM 输出分数形状或数值异常。")
    predicted = pipeline.classes_[np.argmax(score, axis=1)].astype(np.int64)
    return predicted, score


def aggregate_trials(meta, predictions, scores, manifest):
    out = manifest.copy().reset_index(drop=True)
    grouped = meta.groupby("trial_id", sort=False).indices
    totals, used, predicted_ids, sources, vote_lists, mean_lists = [], [], [], [], [], []
    for trial in out.itertuples(index=False):
        indices = np.asarray(grouped.get(trial.trial_id, []), dtype=int)
        valid = indices[predictions[indices] >= 0]
        totals.append(len(indices))
        used.append(len(valid))
        if not len(valid):
            predicted_ids.append(-1)
            sources.append("not_evaluated")
            vote_lists.append([0, 0, 0])
            mean_lists.append([np.nan] * 3)
            continue
        votes = np.bincount(predictions[valid], minlength=3)
        means = scores[valid].mean(axis=0)
        tied = np.flatnonzero(votes == votes.max())
        winner = int(tied[np.argmax(means[tied])])
        predicted_ids.append(winner)
        sources.append("development_oof" if trial.split_role == "development" else "heldout_test")
        vote_lists.append(votes.tolist())
        mean_lists.append(means.tolist())
    out["n_windows_in_npz"] = totals
    out["n_predicted_windows"] = used
    out["prediction_coverage"] = np.divide(used, totals, out=np.zeros(len(out)), where=np.asarray(totals) > 0)
    out["predicted_class_id"] = predicted_ids
    out["prediction_source"] = sources
    out["has_prediction"] = out["predicted_class_id"].ge(0)
    out["correct"] = out["has_prediction"] & out["predicted_class_id"].eq(out["class_id"])
    out["no_prediction_reason"] = ["" if n else ("no_windows_generated" if not total else "no_eligible_windows")
                                    for n, total in zip(used, totals)]
    for k in range(3):
        out["votes_class_%d" % k] = [v[k] for v in vote_lists]
        out["mean_decision_class_%d" % k] = [v[k] for v in mean_lists]
    return out


def metric_pack(y, predicted):
    y, predicted = np.asarray(y, dtype=int), np.asarray(predicted, dtype=int)
    if len(y) == 0:
        return {"n": 0, "correct": 0, "accuracy": None, "balanced_accuracy": None,
                "macro_f1": None, "confusion_matrix": [[0] * 3 for _ in range(3)],
                "support": [0, 0, 0], "recall": [None] * 3, "precision": [None] * 3}
    cm = confusion_matrix(y, predicted, labels=CLASSES)
    support, predicted_count = cm.sum(axis=1), cm.sum(axis=0)
    recall = np.divide(cm.diagonal(), support, out=np.full(3, np.nan), where=support > 0)
    precision = np.divide(cm.diagonal(), predicted_count, out=np.full(3, np.nan), where=predicted_count > 0)
    return json_safe({"n": len(y), "correct": int(np.sum(y == predicted)),
                      "accuracy": float(accuracy_score(y, predicted)),
                      "balanced_accuracy": float(np.nanmean(recall)),
                      "macro_f1": float(f1_score(y, predicted, labels=CLASSES, average="macro", zero_division=0)),
                      "confusion_matrix": cm, "support": support, "recall": recall, "precision": precision})


def evaluate_role(meta, trial_predictions, predictions, role):
    role_windows = meta["split_role"].eq(role).to_numpy()
    evaluated = role_windows & (predictions >= 0)
    role_trials = trial_predictions[trial_predictions["split_role"] == role]
    covered = role_trials[role_trials["has_prediction"]]
    window = metric_pack(meta.loc[evaluated, "y"], predictions[evaluated])
    trial = metric_pack(covered["class_id"], covered["predicted_class_id"])
    window.update({"n_total": int(role_windows.sum()),
                   "coverage": float(evaluated.sum() / role_windows.sum()) if role_windows.any() else None})
    trial.update({"n_total": len(role_trials), "coverage": len(covered) / len(role_trials) if len(role_trials) else None,
                  "strict_accuracy_all_retained_trials": int(covered["correct"].sum()) / len(role_trials)
                  if len(role_trials) else None})
    return {"window": window, "trial": trial}


def tune_on_development(data, folds, candidates, args, output):
    X, meta, manifest = data["X"], data["meta"], data["trials"]
    all_rows, summary_rows = [], []
    best_key, best, best_pred, best_scores = None, None, None, None
    fold_manifest = []
    for fold in folds:
        for role, indices in (("train", fold["train_indices"]), ("validation", fold["validation_indices"])):
            for trial_id in meta.iloc[indices]["trial_id"].unique():
                fold_manifest.append({"fold": fold["fold"], "group_level": fold["group_level"],
                                      "held_out_group": fold["held_out_group"], "fold_role": role, "trial_id": trial_id})
    write_csv(output / "cv_trial_folds.csv", pd.DataFrame(fold_manifest))
    for number, candidate in enumerate(candidates, 1):
        pred = np.full(len(meta), -1, dtype=np.int64)
        scores = np.full((len(meta), 3), np.nan)
        metrics, started = [], time.perf_counter()
        log("候选 %d/%d：%s，C=%s，gamma=%s" %
            (number, len(candidates), candidate["kernel"], candidate["C"], candidate["gamma"]), output)
        for fold in folds:
            fit, val = fold["train_indices"], fold["validation_indices"]
            tick = time.perf_counter()
            pipeline = fit_pipeline(new_pipeline(candidate, args), X[fit], meta.iloc[fit]["y"].to_numpy())
            fold_pred, fold_score = predict_pipeline(pipeline, X[val])
            pred[val], scores[val] = fold_pred, fold_score
            # 仅汇总本折验证 Trial；不读取测试集特征或预测。
            val_trials = manifest[manifest["trial_id"].isin(meta.iloc[val]["trial_id"])].copy()
            trial_result = aggregate_trials(meta.iloc[val].reset_index(drop=True), fold_pred, fold_score, val_trials)
            wm = metric_pack(meta.iloc[val]["y"], fold_pred)
            tm = metric_pack(trial_result["class_id"], trial_result["predicted_class_id"])
            row = dict(candidate, fold=fold["fold"], held_out_group=fold["held_out_group"],
                       train_windows=len(fit), validation_windows=len(val), validation_trials=len(val_trials),
                       window_accuracy=wm["accuracy"], window_balanced_accuracy=wm["balanced_accuracy"],
                       window_macro_f1=wm["macro_f1"], trial_accuracy=tm["accuracy"],
                       trial_balanced_accuracy=tm["balanced_accuracy"], trial_macro_f1=tm["macro_f1"],
                       elapsed_seconds=time.perf_counter() - tick)
            all_rows.append(row)
            metrics.append(row)
            log("  折%d/%d 留出 %s：Trial BA=%.2f%%，窗口 BA=%.2f%%，%.1fs" %
                (fold["fold"], len(folds), fold["held_out_group"], 100 * tm["balanced_accuracy"],
                 100 * wm["balanced_accuracy"], row["elapsed_seconds"]), output)
        summary = dict(candidate, n_folds=len(folds), elapsed_seconds=time.perf_counter() - started)
        for key in ("trial_accuracy", "trial_balanced_accuracy", "trial_macro_f1",
                    "window_accuracy", "window_balanced_accuracy", "window_macro_f1"):
            summary["mean_" + key] = float(np.mean([r[key] for r in metrics]))
            summary["std_" + key] = float(np.std([r[key] for r in metrics], ddof=0))
        summary_rows.append(summary)
        rank = (summary["mean_trial_balanced_accuracy"], summary["mean_trial_macro_f1"],
                summary["mean_window_balanced_accuracy"], -number)
        if best_key is None or rank > best_key:
            best_key, best, best_pred, best_scores = rank, dict(candidate), pred.copy(), scores.copy()
        write_csv(output / "cv_scores.csv", pd.DataFrame(all_rows))
        write_csv(output / "cv_candidates.csv", pd.DataFrame(summary_rows))
    best_summary = next(row for row in summary_rows if row["candidate_id"] == best["candidate_id"])
    return best, best_summary, best_pred, best_scores


def save_bundle(path, pipeline, data, training_mask, role, params, args):
    subset = data["meta"].loc[training_mask]
    bundle = {
        "schema": SCHEMA, "script_version": VERSION, "artifact_role": role,
        "pipeline": pipeline, "feature_names": list(FEATURE_NAMES), "class_ids": CLASSES.tolist(),
        "class_names": data["class_names"], "label_mapping": data["mapping"],
        "preprocess_config": data["config"], "subject_id": data["subject_id"],
        "selected_params": params, "trial_aggregation_rule": TRIAL_RULE,
        "decision_scores_are_probabilities": False, "rest_class_trained": False,
        "training": {"n_windows": len(subset), "n_trials": subset["trial_id"].nunique(),
                     "dates": sorted(subset["collection_date"].unique().tolist()),
                     "sessions": sorted(subset["session_id"].unique().tolist()),
                     "roles": sorted(subset["split_role"].unique().tolist())},
        "source_hashes": data["source_hashes"], "versions": software_versions(),
        "independently_evaluated": role == "evaluation",
        "evaluation_reference_file": "evaluation.json",
        "seed": args.seed,
    }
    joblib.dump(bundle, path, compress=3)


def software_versions():
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__}


def group_results(meta, trial_predictions, predictions):
    rows = []
    for role in ("development", "test"):
        for keys, level in [(["collection_date"], "date"), (["session_id"], "session"),
                            (["session_id", "run_id"], "run")]:
            subset = trial_predictions[trial_predictions["split_role"] == role]
            for group_key, group in subset.groupby(keys, sort=True):
                values = group_key if isinstance(group_key, tuple) else (group_key,)
                mask = meta["split_role"].eq(role).to_numpy()
                for key, value in zip(keys, values):
                    mask &= meta[key].eq(value).to_numpy()
                valid = mask & (predictions >= 0)
                covered = group[group["has_prediction"]]
                wm = metric_pack(meta.loc[valid, "y"], predictions[valid])
                tm = metric_pack(covered["class_id"], covered["predicted_class_id"])
                rows.append({"role": role, "prediction_source": "development_oof" if role == "development" else "heldout_test",
                             "level": level, "group": " :: ".join(str(x) for x in values),
                             "n_trials_total": len(group), "n_trials_predicted": len(covered),
                             "trial_coverage": len(covered) / len(group), "trial_accuracy": tm["accuracy"],
                             "trial_balanced_accuracy": tm["balanced_accuracy"], "trial_macro_f1": tm["macro_f1"],
                             "trial_strict_accuracy": int(covered["correct"].sum()) / len(group),
                             "n_windows_total": int(mask.sum()), "n_windows_predicted": int(valid.sum()),
                             "window_accuracy": wm["accuracy"], "window_balanced_accuracy": wm["balanced_accuracy"],
                             "window_macro_f1": wm["macro_f1"]})
    return pd.DataFrame(rows)


def percent(value):
    return "—" if value is None or not np.isfinite(value) else "%.2f%%" % (100 * value)


def table(headers, rows):
    esc = lambda value: html.escape(str(value))
    return "<table><thead><tr>" + "".join("<th>" + esc(x) + "</th>" for x in headers) + \
        "</tr></thead><tbody>" + "".join("<tr>" + "".join("<td>" + esc(x) + "</td>" for x in row) +
                                         "</tr>" for row in rows) + "</tbody></table>"


def write_report(output, result, data, grouped, candidate_frame):
    esc = lambda value: html.escape(str(value))
    test, dev = result["heldout_test"], result["development_selected_oof"]
    blocks = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'><title>四通道 RPS SVM 报告</title>",
        "<style>body{font-family:system-ui,'Microsoft YaHei',sans-serif;max-width:1180px;margin:32px auto;padding:0 22px;line-height:1.7;color:#1a2634}h1,h2{color:#14283e}table{border-collapse:collapse;width:100%;margin:16px 0 24px;font-size:14px}th,td{border:1px solid #d6dfe9;text-align:left;padding:9px 11px;overflow-wrap:anywhere}th{background:#e8f0f8}tr:nth-child(even){background:#f7f9fc}.box{padding:16px 20px;background:#edf4fb;border-left:4px solid #376a99}code{background:#edf1f5;padding:2px 5px}.scroll{overflow-x:auto}small{color:#526172}</style>",
        "<h1>四通道 RPS · SVM 基础模型</h1>",
        "<p>受试者：<b>%s</b>。算法：<b>%s SVC</b>；每窗20维特征；标准化与分类器封装在同一个 Pipeline。</p>" %
        (esc(data["subject_id"]), esc(result["selected_params"]["kernel"].upper())),
        "<div class='box'>独立测试范围：<b>%s</b>。Trial BA：<b>%s</b>；窗口 BA：<b>%s</b>。<br>测试 Trial 预测覆盖：%d/%d；正确：%d；全部保留 Trial 的严格准确率：%s。</div>" %
        (esc(result["evaluation_scope"]), percent(test["trial"]["balanced_accuracy"]),
         percent(test["window"]["balanced_accuracy"]), test["trial"]["n"], test["trial"]["n_total"],
         test["trial"]["correct"], percent(test["trial"]["strict_accuracy_all_retained_trials"])),
        "<h2>数据划分与质量标记</h2>",
        table(["项目", "值"], [["开发日期", ", ".join(result["development_dates"])],
             ["测试日期", ", ".join(result["test_dates"])], ["预处理划分", result["preprocess_split_mode"]],
             ["内部验证分组", result["cv_group_level"]], ["保留 Trial", result["audit"]["n_trials_retained"]],
             ["保留窗口", result["audit"]["n_windows_retained"]],
             ["窗口 QC", result["audit"]["window_qc_counts"]], ["Trial QC", result["audit"]["trial_qc_counts"]],
             ["不参与拟合/评分的窗口", result["audit"]["n_windows_not_used"]]]),
        "<p>WARNING 保留。REVIEW、预处理不可用、非有限特征和未分配角色仅标记为不参与拟合；全部记录仍在预测清单。没有根据分类对错修改 QC。报告中的 BA/F1 只针对有预测的记录，覆盖率单独列出；严格 Trial 准确率把未预测 Trial 计入分母。</p>",
        "<h2>开发集选参</h2><p>按组留一验证，每折 StandardScaler 只拟合该折训练数据。主选参指标为各验证组 Trial BA 的均值；并列依次比较 Trial Macro-F1、窗口 BA、原始候选顺序。以下是用于选参的验证分数，未经嵌套验证，不能代替独立测试结果。</p>",
        table(["候选", "核", "C", "gamma", "平均 Trial BA", "Trial BA 标准差", "平均窗口 BA", "选中"],
              [[row.candidate_id, row.kernel, row.C, row.gamma, percent(row.mean_trial_balanced_accuracy),
                percent(row.std_trial_balanced_accuracy), percent(row.mean_window_balanced_accuracy),
                "是" if row.candidate_id == result["selected_params"]["candidate_id"] else ""]
               for row in candidate_frame.itertuples(index=False)]),
        "<h2>冻结评估模型的结果</h2>",
        table(["数据", "层级", "已预测/总数", "准确率", "平衡准确率 BA", "Macro-F1"],
              [[label, level, "%d/%d" % (pack[level]["n"], pack[level]["n_total"]),
                percent(pack[level]["accuracy"]), percent(pack[level]["balanced_accuracy"]),
                percent(pack[level]["macro_f1"])]
               for label, pack in [("开发集选中参数的 OOF（选参用途）", dev), ("独立测试", test)]
               for level in ("window", "trial")]),
        "<p>窗口来自同一动作的重叠片段，不是相互独立的实验重复。Trial 使用完整核心动作的多数票，票数相同按平均 OVR 分数打破平局，再相同取较小类别编号。此规则预先固定；Trial 准确率不能解释为实时检测延迟或静息误触发率。</p>",
        "<h2>独立测试混淆矩阵</h2><p>行是真实类别，列是预测类别。</p>"]
    for level in ("trial", "window"):
        blocks.append("<h3>%s</h3>" % level)
        blocks.append(table(["真实 / 预测"] + data["class_names"],
                            [[data["class_names"][i]] + row for i, row in enumerate(test[level]["confusion_matrix"])]))
    blocks += ["<h2>各日期、Session、Run</h2><div class='scroll'>",
        table(["数据", "层级", "组", "Trial覆盖", "Trial BA", "Trial准确率", "窗口 BA"],
              [[r.role, r.level, r.group, "%d/%d" % (r.n_trials_predicted, r.n_trials_total),
                percent(r.trial_balanced_accuracy), percent(r.trial_accuracy), percent(r.window_balanced_accuracy)]
               for r in grouped.itertuples(index=False)]), "</div>",
        "<h2>类别对应</h2>", table(["类别编号", "Marker", "名称"], [[i, i + 11, name] for i, name in enumerate(data["class_names"])]),
        "<p>" + ("手势名称由采集配置/命令行提供。" if data["mapping"].get("gesture_names_supplied_by_user") else
                 "尚未确认 11/12/13 对应石头/剪刀/布，当前保留 class_11/class_12/class_13；不要猜测映射。") + "</p>",
        "<h2>保存的模型与下一步</h2>",
        table(["文件", "用途"], [["eval_model.joblib", "只用开发集拟合，对应此报告的独立测试结果。"],
             ["base_model.joblib", "本次未生成（--no-refit-all）。" if not result["deployment_model_saved"] else
              "评估完成后，合并开发与测试合格窗口重训的部署基础模型；未进行独立评估，需在新一天验证。"],
             ["window_predictions.csv / trial_predictions.csv", "全部记录、预测、QC、参与情况、无预测原因。"],
             ["cv_scores.csv / cv_candidates.csv / cv_trial_folds.csv", "候选逐折分数、汇总和 Trial 分组。"],
             ["evaluation.json / training_config.json", "测试结果、覆盖率、固定规则、数据哈希和软件版本。"],
             ["preprocess_config.json / label_mapping.json", "实时推理须沿用的通道、因果滤波、特征顺序与类别映射。"]]),
        "<p>" + ("部署基础模型已保存；它使用了原测试数据，因此不能再用这些数据报告该部署版准确率。" if result["deployment_model_saved"] else
                 "本次使用 --no-refit-all，未生成 base_model.joblib。") + "</p>",
        "<p>下一步：新一天先采校准数据，另留完整 Run 做测试；再做连续回放与实时流。当前是三类动作分类器，没有静息类别或活动门控，不能直接把连续静息时的三分类输出当作有效手势。decision score 是 SVM 分数，不是置信概率。</p>",
        "<small>schema: %s；script: %s；所有指标来自本次实际运行。</small></html>" % (SCHEMA, VERSION)]
    (output / "OPEN_REPORT.html").write_text("\n".join(blocks), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description="四通道 OpenBCI RPS：SVM 分组验证、独立测试和基础模型。",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", "--input", required=True, help="预处理输出子文件夹，或其 rps_windows.npz 文件")
    parser.add_argument("--output", default="RPS_SVM_STEP02", help="输出父目录；每次新建独立子目录")
    parser.add_argument("--c-values", nargs="+", type=float, default=[0.1, 1.0, 10.0], help="两种核共同搜索的 C")
    parser.add_argument("--gamma-values", nargs="+", default=["scale", "0.01", "0.1"], help="RBF gamma 候选")
    parser.add_argument("--class-names", nargs=3, help="确认映射后提供 Marker 11、12、13 的名称，不改变标签编号")
    parser.add_argument("--cache-mb", type=float, default=512.0, help="每次 SVC 拟合的核缓存 MB")
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--no-refit-all", action="store_true", help="仅保存开发集评估模型，不合并数据另训部署版")
    return parser


def run(args):
    require(np.isfinite(args.cache_mb) and args.cache_mb > 0, "--cache-mb 必须为有限正数。")
    require(0 <= args.seed <= 2 ** 32 - 1, "seed 必须在0到2**32-1之间。")
    candidates = build_candidates(args)
    log("读取预处理特征与分组，核查日期、Trial、类别和配置……")
    data = read_inputs(args.data, args.class_names)
    folds = make_folds(data)
    meta, trials, X = data["meta"], data["trials"], data["X"]
    parent = Path(args.output).expanduser().resolve()
    safe_subject = "".join(c if c.isalnum() or c in "_-" else "_" for c in data["subject_id"])
    output = parent / ("RPS_SVM_%s_%s" % (safe_subject[:60], datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    output.mkdir(parents=True, exist_ok=False)
    args._output = output
    eligible = meta["eligible_for_model"].to_numpy()
    dev = eligible & meta["split_role"].eq("development").to_numpy()
    test = eligible & meta["split_role"].eq("test").to_numpy()
    log("保留 %d Trial / %d 窗口；合格开发窗口 %d，合格测试窗口 %d。" %
        (len(trials), len(meta), dev.sum(), test.sum()), output)
    log("验证分组=%s，%d折；%d个候选，预计%d次验证拟合。" %
        (folds[0]["group_level"], len(folds), len(candidates), len(folds) * len(candidates)), output)
    training_config = {"schema": SCHEMA, "script_version": VERSION, "source_directory": str(data["root"]),
        "source_hashes": data["source_hashes"], "feature_names": FEATURE_NAMES,
        "cv_group_level": folds[0]["group_level"], "cv_groups": [f["held_out_group"] for f in folds],
        "candidates": candidates, "class_weight": "balanced", "scaler": "StandardScaler_fitted_in_each_training_fold",
        "selection_rule": "mean_fold_trial_BA_then_trial_macro_F1_then_window_BA_then_candidate_order",
        "trial_aggregation_rule": TRIAL_RULE, "test_used_for_selection": False,
        "probability_calibration": False, "qc_policy": "preserve_all_records_keep_WARNING_use_preprocess_usable_and_finite_non_REVIEW_assigned",
        "deployment_refit_after_evaluation": not args.no_refit_all, "cache_mb": args.cache_mb,
        "seed": args.seed, "versions": software_versions(), "command_line": sys.argv}
    write_json(output / "training_config.json", training_config)
    write_json(output / "preprocess_config.json", data["config"])
    write_json(output / "label_mapping.json", data["mapping"])
    write_csv(output / "window_usage_manifest.csv", meta)
    write_csv(output / "trial_manifest.csv", trials)
    best, best_summary, predictions, scores = tune_on_development(data, folds, candidates, args, output)
    write_json(output / "selected_params.json", {"params": best, "development_cv": best_summary,
                                                 "selected_without_test": True, "trial_aggregation_rule": TRIAL_RULE})
    log("参数已冻结：%s；用全部开发集合格窗口拟合评估模型。" % best, output)
    evaluation_model = fit_pipeline(new_pipeline(best, args), X[dev], meta.loc[dev, "y"].to_numpy())
    save_bundle(output / "eval_model.joblib", evaluation_model, data, dev, "evaluation", best, args)
    log("只对冻结模型执行一次独立测试推理。", output)
    test_pred, test_score = predict_pipeline(evaluation_model, X[test])
    predictions[test], scores[test] = test_pred, test_score
    trial_predictions = aggregate_trials(meta, predictions, scores, trials)
    trial_predictions["predicted_name"] = [data["class_names"][p] if p >= 0 else ""
                                           for p in trial_predictions["predicted_class_id"]]
    window_predictions = meta.copy()
    window_predictions["class_name"] = [data["class_names"][p] for p in meta["y"]]
    window_predictions["predicted_class_id"] = predictions
    window_predictions["predicted_name"] = [data["class_names"][p] if p >= 0 else "" for p in predictions]
    window_predictions["prediction_source"] = np.where(predictions < 0, "not_evaluated",
        np.where(meta["split_role"].eq("development"), "development_oof", "heldout_test"))
    window_predictions["has_prediction"] = predictions >= 0
    window_predictions["correct"] = (predictions >= 0) & (predictions == meta["y"].to_numpy())
    for k in range(3):
        window_predictions["decision_class_%d" % k] = scores[:, k]
    write_csv(output / "window_predictions.csv", window_predictions)
    write_csv(output / "trial_predictions.csv", trial_predictions)
    grouped = group_results(meta, trial_predictions, predictions)
    write_csv(output / "per_group_metrics.csv", grouped)
    dev_dates = sorted(trials.loc[trials["split_role"] == "development", "collection_date"].unique().tolist())
    test_dates = sorted(trials.loc[trials["split_role"] == "test", "collection_date"].unique().tolist())
    dev_sessions = set(trials.loc[trials["split_role"] == "development", "session_id"])
    test_sessions = set(trials.loc[trials["split_role"] == "test", "session_id"])
    scope = ("同一人跨日留出测试" if set(dev_dates).isdisjoint(test_dates) else
             ("同一人留 Session 测试（含相同日期，不代表独立跨日）" if dev_sessions.isdisjoint(test_sessions)
              else "同一人同 Session 留 Run 测试，不代表跨日"))
    result = {"schema": SCHEMA, "script_version": VERSION, "subject_id": data["subject_id"],
        "evaluation_scope": scope, "preprocess_split_mode": data["config"]["split"]["mode"],
        "development_dates": dev_dates, "test_dates": test_dates, "cv_group_level": folds[0]["group_level"],
        "cv_folds": len(folds), "selected_params": best, "selected_cv_summary": best_summary,
        "development_selected_oof": evaluate_role(meta, trial_predictions, predictions, "development"),
        "heldout_test": evaluate_role(meta, trial_predictions, predictions, "test"),
        "audit": {"n_trials_retained": len(trials), "n_windows_retained": len(meta),
                  "n_windows_eligible": int(eligible.sum()), "n_windows_not_used": int((~eligible).sum()),
                  "window_qc_counts": meta["qc_status"].value_counts().to_dict(),
                  "trial_qc_counts": trials["qc_status"].value_counts().to_dict(),
                  "not_used_reasons": meta.loc[~eligible, "not_used_reason"].value_counts().to_dict()},
        "class_names": data["class_names"], "trial_aggregation_rule": TRIAL_RULE,
        "development_oof_was_used_for_selection_not_independent": True,
        "test_used_for_selection": False, "test_prediction_passes": 1,
        "decision_scores_are_probabilities": False, "rest_class_trained": False,
        "deployment_model_saved": False, "deployment_model_independently_evaluated": False,
        "source_hashes": data["source_hashes"], "output_directory": str(output), "versions": software_versions()}
    # 先持久化独立评估。下方部署版训练不会替换、更新或重新计算该结果。
    write_json(output / "evaluation.json", result)
    held = result["heldout_test"]
    log("独立测试：窗口 BA=%s，Trial BA=%s，Trial 正确%d/%d（覆盖%d/%d）。" %
        (percent(held["window"]["balanced_accuracy"]), percent(held["trial"]["balanced_accuracy"]),
         held["trial"]["correct"], held["trial"]["n"], held["trial"]["n"], held["trial"]["n_total"]), output)
    if not args.no_refit_all:
        log("评估已保存；合并开发与测试合格窗口，另训部署基础模型（需新一天独立验证）。", output)
        base_model = fit_pipeline(new_pipeline(best, args), X[eligible], meta.loc[eligible, "y"].to_numpy())
        save_bundle(output / "base_model.joblib", base_model, data, eligible, "deployment_base", best, args)
        result["deployment_model_saved"] = True
        write_json(output / "evaluation.json", result)
    write_report(output, result, data, grouped, pd.read_csv(output / "cv_candidates.csv"))
    log("完成。报告：%s" % (output / "OPEN_REPORT.html"), output)
    log("eval_model 对应独立测试。" + ("base_model 用于后续新日校准。" if result["deployment_model_saved"]
                                      else "本次未重训部署版。"), output)
    return result


def main():
    args = build_parser().parse_args()
    try:
        run(args)
    except Exception as exc:
        detail = traceback.format_exc()
        output = getattr(args, "_output", None)
        if output is not None:
            (output / "FAILED.txt").write_text(detail, encoding="utf-8")
        print("\n[失败] %s" % exc, file=sys.stderr, flush=True)
        if output is not None:
            print("详细信息：%s" % (output / "FAILED.txt"), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
