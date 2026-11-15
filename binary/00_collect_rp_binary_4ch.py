#!/usr/bin/env python3
"""OpenBCI GUI + PsychoPy 石头/布 sEMG 采集程序。

本程序不直接连接 Cyton。OpenBCI GUI 负责连接硬件和保存原始数据；
本程序负责全屏呈现实验提示，并通过 UDP 向 GUI Marker Widget 发送标签。

Marker UDP 格式使用 OpenBCI GUI 实际解析的网络字节序 32 位浮点数：
    struct.pack("!f", float(marker))

Run 边界 Marker 直接编码 Run 编号：
    Run 1 开始/结束/中止 = 101/111/121
    Run 2 开始/结束/中止 = 102/112/122
    ...

因此 OpenBCI 原始 TXT 的 Marker Channel 与本程序 events.csv 都能恢复真实
Run 边界，不再需要分析阶段按“每12个目标 Trial”猜测 Run。

主要命令：
    python 00_collect_rp_binary_4ch.py --test-markers
    python 00_collect_rp_binary_4ch.py --mode pilot --subject S01 --session 01
    python 00_collect_rp_binary_4ch.py --mode formal --subject S01 --session 01
    python 00_collect_rp_binary_4ch.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import socket
import struct
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config_rps_emg.json"


class ExperimentAbort(RuntimeError):
    """用户按 Esc 或程序需要安全结束。"""


@dataclass(frozen=True)
class Gesture:
    key: str
    name_cn: str
    name_en: str
    instruction: str
    marker: float
    color: str


class MarkerSender:
    """向 OpenBCI GUI Marker Widget 发送单个网络字节序 32 位浮点数。"""

    def __init__(self, ip: str, port: int, enabled: bool = True) -> None:
        self.ip = ip
        self.port = int(port)
        self.enabled = bool(enabled)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, marker: float) -> float:
        sent_at = time.perf_counter()
        if self.enabled:
            payload = struct.pack("!f", float(marker))
            self.sock.sendto(payload, (self.ip, self.port))
        return sent_at

    def close(self) -> None:
        self.sock.close()


class EventLogger:
    FIELDS = [
        "local_iso",
        "monotonic_s",
        "subject",
        "session",
        "mode",
        "run",
        "planned_trial",
        "attempt",
        "gesture_key",
        "gesture_cn",
        "marker",
        "event",
        "valid",
        "detail",
    ]

    def __init__(self, path: Path, subject: str, session: str, mode: str) -> None:
        self.path = path
        self.subject = subject
        self.session = session
        self.mode = mode
        self._file = path.open("w", newline="", encoding="utf-8-sig")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._file.flush()

    def log(
        self,
        *,
        event: str,
        run: int = 0,
        planned_trial: int = 0,
        attempt: int = 0,
        gesture: Optional[Gesture] = None,
        marker: Optional[float] = None,
        valid: Optional[bool] = None,
        detail: str = "",
        monotonic_s: Optional[float] = None,
    ) -> None:
        row = {
            "local_iso": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "monotonic_s": f"{(monotonic_s if monotonic_s is not None else time.perf_counter()):.6f}",
            "subject": self.subject,
            "session": self.session,
            "mode": self.mode,
            "run": run,
            "planned_trial": planned_trial,
            "attempt": attempt,
            "gesture_key": gesture.key if gesture else "",
            "gesture_cn": gesture.name_cn if gesture else "",
            "marker": "" if marker is None else f"{float(marker):g}",
            "event": event,
            "valid": "" if valid is None else int(valid),
            "detail": detail,
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def require_key(config: Dict[str, Any], dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"配置缺少字段：{dotted}")
        value = value[part]
    return value


def run_marker_value(config: Dict[str, Any], kind: str, run_index: int) -> float:
    """根据 Run 编号生成唯一边界 Marker。"""
    if kind not in {"start", "end", "abort"}:
        raise ValueError(f"未知 Run Marker 类型：{kind}")
    if int(run_index) < 1:
        raise ValueError("run_index 必须从1开始。")
    base = float(config["run_markers"][f"{kind}_base"])
    return base + int(run_index)


def build_run_marker_map(config: Dict[str, Any], runs: int) -> List[Dict[str, Any]]:
    return [
        {
            "run": run_index,
            "run_start_marker": run_marker_value(config, "start", run_index),
            "run_end_marker": run_marker_value(config, "end", run_index),
            "run_abort_marker": run_marker_value(config, "abort", run_index),
        }
        for run_index in range(1, runs + 1)
    ]


def validate_config(config: Dict[str, Any]) -> None:
    if any(float(config["markers"][key]) != value for key,value in {"rock":11,"paper":13,"end":20,"reject":90}.items()):
        raise ValueError("二分类固定Marker：石头11、布13、结束20、拒绝90。")
    required = [
        "network.ip",
        "network.port",
        "markers.rock",
        "markers.paper",
        "markers.end",
        "markers.reject",
        "run_markers.start_base",
        "run_markers.end_base",
        "run_markers.abort_base",
        "timing.baseline_min_s",
        "timing.baseline_max_s",
        "timing.cue_s",
        "timing.action_s",
        "timing.return_s",
        "timing.training_start_after_marker_s",
        "timing.training_end_after_marker_s",
        "timing.run_boundary_display_s",
        "protocol.practice_per_class",
        "protocol.pilot.runs",
        "protocol.pilot.trials_per_run",
        "protocol.formal.runs",
        "protocol.formal.trials_per_run",
    ]
    for key in required:
        require_key(config, key)

    action_marker_values = [
        float(config["markers"][name])
        for name in ("rock", "paper", "end", "reject")
    ]
    if len(action_marker_values) != len(set(action_marker_values)):
        raise ValueError("rock/paper/end/reject 的 Marker 必须互不相同。")

    baseline_min = float(config["timing"]["baseline_min_s"])
    baseline_max = float(config["timing"]["baseline_max_s"])
    action_s = float(config["timing"]["action_s"])
    train_start = float(config["timing"]["training_start_after_marker_s"])
    train_end = float(config["timing"]["training_end_after_marker_s"])
    if not 0 < baseline_min <= baseline_max:
        raise ValueError("baseline 时间必须满足 0 < min <= max。")
    if not 0 <= train_start < train_end <= action_s:
        raise ValueError("稳定训练段必须位于 action_s 内部。")
    if float(config["timing"]["run_boundary_display_s"]) <= 0:
        raise ValueError("run_boundary_display_s 必须大于0。")

    max_runs = 0
    for mode in ("pilot", "formal"):
        trials = int(config["protocol"][mode]["trials_per_run"])
        runs = int(config["protocol"][mode]["runs"])
        if trials <= 0 or runs <= 0:
            raise ValueError(f"{mode} 的 runs 和 trials_per_run 必须大于0。")
        if trials % 2 != 0:
            raise ValueError(f"{mode}.trials_per_run 必须能被2整除，才能保证两类平衡。")
        max_runs = max(max_runs, runs)

    generated_run_markers = [
        float(item[key])
        for item in build_run_marker_map(config, max_runs)
        for key in ("run_start_marker", "run_end_marker", "run_abort_marker")
    ]
    all_marker_values = action_marker_values + generated_run_markers
    if len(all_marker_values) != len(set(all_marker_values)):
        raise ValueError(
            "动作Marker与生成的Run开始/结束/中止Marker存在重复，请调整run_markers基数。"
        )

    port = int(config["network"]["port"])
    if not 1 <= port <= 65535:
        raise ValueError("UDP port 必须在1–65535之间。")


def build_gestures(config):
    m = config["markers"]
    return {
        "rock": Gesture("rock", "石头", "ROCK", "四指握拳，拇指在外；自然用力，手腕保持中立", float(m["rock"]), "#E57373"),
        "paper": Gesture("paper", "布", "PAPER", "五指伸直并自然分开；自然用力，手腕保持中立", float(m["paper"]), "#66C2A5"),
    }



def no_three_identical(sequence: Sequence[str]) -> bool:
    return all(
        not (sequence[i] == sequence[i - 1] == sequence[i - 2])
        for i in range(2, len(sequence))
    )


def balanced_sequence(trials_per_run: int, rng: random.Random) -> List[str]:
    if trials_per_run % 2 != 0:
        raise ValueError("trials_per_run 必须能被2整除。")
    each = trials_per_run // 2
    base = [key for key in ("rock", "paper") for _ in range(each)]
    for _ in range(10000):
        candidate = base.copy()
        rng.shuffle(candidate)
        if no_three_identical(candidate):
            return candidate
    raise RuntimeError("无法生成满足约束的伪随机序列。")


def build_schedule(
    runs: int, trials_per_run: int, seed: int
) -> List[List[str]]:
    rng = random.Random(seed)
    return [balanced_sequence(trials_per_run, rng) for _ in range(runs)]


def write_schedule_csv(path: Path, schedule: Sequence[Sequence[str]], gestures: Dict[str, Gesture]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run", "planned_trial", "gesture_key", "gesture_cn", "start_marker"],
        )
        writer.writeheader()
        for run_index, run_sequence in enumerate(schedule, start=1):
            for trial_index, key in enumerate(run_sequence, start=1):
                gesture = gestures[key]
                writer.writerow(
                    {
                        "run": run_index,
                        "planned_trial": trial_index,
                        "gesture_key": key,
                        "gesture_cn": gesture.name_cn,
                        "start_marker": f"{gesture.marker:g}",
                    }
                )


def write_run_markers_csv(path: Path, config: Dict[str, Any], runs: int) -> None:
    """保存本次实验每个 Run 的真实边界 Marker 映射。"""
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run", "run_start_marker", "run_end_marker", "run_abort_marker"],
        )
        writer.writeheader()
        for row in build_run_marker_map(config, runs):
            writer.writerow(
                {
                    key: f"{float(value):g}" if key != "run" else int(value)
                    for key, value in row.items()
                }
            )


def send_and_log_marker(
    sender: MarkerSender,
    logger: EventLogger,
    marker: float,
    event_name: str,
    run_index: int,
    trial_index: int,
    attempt: int,
    gesture: Gesture,
    valid: Optional[bool] = None,
    detail: str = "",
) -> None:
    sent_at = sender.send(marker)
    logger.log(
        event=event_name,
        run=run_index,
        planned_trial=trial_index,
        attempt=attempt,
        gesture=gesture,
        marker=marker,
        valid=valid,
        detail=detail,
        monotonic_s=sent_at,
    )


def send_and_log_run_marker(
    sender: MarkerSender,
    logger: EventLogger,
    marker: float,
    event_name: str,
    run_index: int,
    detail: str,
    valid: Optional[bool] = None,
) -> None:
    """发送Run边界Marker，并用同一个单调时钟时间写入events.csv。"""
    sent_at = sender.send(marker)
    logger.log(
        event=event_name,
        run=run_index,
        marker=marker,
        valid=valid,
        detail=detail,
        monotonic_s=sent_at,
    )


def run_self_test(config: Dict[str, Any], verbose: bool = True) -> None:
    validate_config(config)
    gestures = build_gestures(config)
    if set(gestures) != {"rock", "paper"}:
        raise AssertionError("手势定义不完整。")

    for mode in ("pilot", "formal"):
        runs = int(config["protocol"][mode]["runs"])
        trials = int(config["protocol"][mode]["trials_per_run"])
        schedule = build_schedule(runs, trials, seed=20260823)
        for run_sequence in schedule:
            counts = {key: run_sequence.count(key) for key in gestures}
            expected = trials // 2
            if any(count != expected for count in counts.values()):
                raise AssertionError(f"{mode} 类别不平衡：{counts}")
            if not no_three_identical(run_sequence):
                raise AssertionError(f"{mode} 出现连续三个相同手势。")

    max_runs = max(int(config["protocol"][mode]["runs"]) for mode in ("pilot", "formal"))
    marker_values = [float(value) for value in config["markers"].values()]
    marker_values.extend(
        float(item[key])
        for item in build_run_marker_map(config, max_runs)
        for key in ("run_start_marker", "run_end_marker", "run_abort_marker")
    )
    for value in marker_values:
        payload = struct.pack("!f", float(value))
        if len(payload) != 4:
            raise AssertionError("Marker UDP 负载不是4字节。")
        recovered = struct.unpack("!f", payload)[0]
        if not math.isclose(recovered, float(value)):
            raise AssertionError("Marker 打包/解包失败。")

    if verbose:
        print("SELF-TEST PASS")
        print("- 配置字段与时间范围：通过")
        print("- Pilot/Formal 两类平衡与随机约束：通过")
        print("- OpenBCI GUI UDP 32位浮点格式（4字节）：通过")
        print("- Run开始/结束/中止Marker唯一性：通过")
        for row in build_run_marker_map(config, max_runs):
            print(
                f"  Run {row['run']}: start={row['run_start_marker']:g}, "
                f"end={row['run_end_marker']:g}, abort={row['run_abort_marker']:g}"
            )


def test_markers(config: Dict[str, Any], interval_s: float) -> None:
    validate_config(config)
    sender = MarkerSender(config["network"]["ip"], int(config["network"]["port"]))
    markers = config["markers"]
    ordered = [
        ("石头", float(markers["rock"])),
        ("布", float(markers["paper"])),
        ("结束", float(markers["end"])),
        ("作废", float(markers["reject"])),
        ("Run 1开始", run_marker_value(config, "start", 1)),
        ("Run 1结束", run_marker_value(config, "end", 1)),
        ("Run 1中止", run_marker_value(config, "abort", 1)),
    ]
    print(f"向 OpenBCI GUI Marker Widget 发送：{sender.ip}:{sender.port}")
    try:
        for name, value in ordered:
            sender.send(value)
            print(f"已发送 {name}: {value:g}")
            time.sleep(interval_s)
    finally:
        sender.close()
    expected = [f"{value:g}" for _, value in ordered]
    print(f"测试结束。请在 GUI Marker Widget 中确认依次出现：{', '.join(expected)}。")
    print("若纵轴仍为0–8，请将 Marker Widget 的 Vert Scale 设为 Auto，并检查记录文件的 Marker Channel。")


def import_psychopy() -> Tuple[Any, Any, Any]:
    try:
        from psychopy import core, event, visual
    except ImportError as exc:
        raise RuntimeError(
            "未安装 PsychoPy。请先运行：python -m pip install -r requirements.txt"
        ) from exc
    return core, event, visual


def draw_message(
    win: Any,
    visual: Any,
    *,
    title: str,
    subtitle: str = "",
    title_color: str = "white",
    font: str = "Microsoft YaHei",
    progress: str = "",
) -> None:
    visual.TextStim(
        win,
        text=title,
        color=title_color,
        height=0.095,
        pos=(0, 0.10),
        wrapWidth=1.55,
        alignText="center",
        font=font,
    ).draw()
    if subtitle:
        visual.TextStim(
            win,
            text=subtitle,
            color="#D9E2EC",
            height=0.038,
            pos=(0, -0.08),
            wrapWidth=1.55,
            alignText="center",
            font=font,
        ).draw()
    if progress:
        visual.TextStim(
            win,
            text=progress,
            color="#9FB3C8",
            height=0.027,
            pos=(0, -0.42),
            wrapWidth=1.6,
            alignText="center",
            font=font,
        ).draw()


def wait_for_space(win: Any, visual: Any, event: Any, title: str, subtitle: str, font: str) -> None:
    while True:
        draw_message(
            win,
            visual,
            title=title,
            subtitle=f"{subtitle}\n\n按空格键继续｜按 Esc 结束",
            title_color="#66C2A5",
            font=font,
        )
        win.flip()
        keys = event.waitKeys(keyList=["space", "escape"])
        if "escape" in keys:
            raise ExperimentAbort("用户按 Esc 结束。")
        if "space" in keys:
            event.clearEvents()
            return


def poll_control_keys(event: Any) -> Optional[str]:
    keys = event.getKeys(keyList=["b", "escape"])
    if "escape" in keys:
        return "escape"
    if "b" in keys:
        return "reject"
    return None


def show_for_duration(
    win: Any,
    core: Any,
    event: Any,
    draw_fn: Any,
    duration_s: float,
) -> Optional[str]:
    draw_fn()
    win.flip()
    clock = core.Clock()
    while clock.getTime() < duration_s:
        control = poll_control_keys(event)
        if control is not None:
            return control
        core.wait(0.004)
    return None


def show_rejected(win: Any, visual: Any, core: Any, font: str, message: str) -> None:
    draw_message(
        win,
        visual,
        title="本次 Trial 已作废",
        subtitle=message + "\n稍后自动补采",
        title_color="#FF6B6B",
        font=font,
    )
    win.flip()
    core.wait(0.8)


def run_practice(
    win: Any,
    visual: Any,
    core: Any,
    event: Any,
    config: Dict[str, Any],
    gestures: Dict[str, Gesture],
    rng: random.Random,
    font: str,
) -> None:
    per_class = int(config["protocol"]["practice_per_class"])
    if per_class <= 0:
        return
    sequence = [key for key in gestures for _ in range(per_class)]
    rng.shuffle(sequence)
    wait_for_space(
        win,
        visual,
        event,
        "练习阶段",
        f"共 {len(sequence)} 次练习，不发送 Marker，也不进入正式数据。\n看到手势后，等“现在做”再动作。",
        font,
    )
    timing = config["timing"]
    for index, key in enumerate(sequence, start=1):
        gesture = gestures[key]
        event.clearEvents()
        rest_duration = rng.uniform(
            float(timing["baseline_min_s"]), float(timing["baseline_max_s"])
        )
        control = show_for_duration(
            win,
            core,
            event,
            lambda: draw_message(
                win,
                visual,
                title="·",
                subtitle="手指自然微弯，完全放松",
                title_color="#D9E2EC",
                font=font,
                progress=f"练习 {index}/{len(sequence)}",
            ),
            rest_duration,
        )
        if control == "escape":
            raise ExperimentAbort("练习阶段按 Esc 结束。")

        control = show_for_duration(
            win,
            core,
            event,
            lambda g=gesture: draw_message(
                win,
                visual,
                title=f"准备：{g.name_cn}\n{g.name_en}",
                subtitle="现在仍保持放松，不要提前动作",
                title_color="#FFD166",
                font=font,
                progress=f"练习 {index}/{len(sequence)}",
            ),
            float(timing["cue_s"]),
        )
        if control == "escape":
            raise ExperimentAbort("练习阶段按 Esc 结束。")

        control = show_for_duration(
            win,
            core,
            event,
            lambda g=gesture: draw_message(
                win,
                visual,
                title=f"现在做：{g.name_cn}\n{g.name_en}",
                subtitle=f"{g.instruction}\n保持到屏幕提示放松",
                title_color=g.color,
                font=font,
                progress=f"练习 {index}/{len(sequence)}",
            ),
            float(timing["action_s"]),
        )
        if control == "escape":
            raise ExperimentAbort("练习阶段按 Esc 结束。")

        show_for_duration(
            win,
            core,
            event,
            lambda: draw_message(
                win,
                visual,
                title="放松",
                subtitle="回到手指自然微弯状态",
                title_color="#66C2A5",
                font=font,
            ),
            float(timing["return_s"]),
        )


def run_trial(
    *,
    win: Any,
    visual: Any,
    core: Any,
    event: Any,
    config: Dict[str, Any],
    sender: MarkerSender,
    logger: EventLogger,
    gesture: Gesture,
    run_index: int,
    trial_index: int,
    attempt: int,
    valid_done: int,
    valid_target: int,
    rng: random.Random,
    font: str,
) -> str:
    """返回 valid、reject 或 abort。"""
    timing = config["timing"]
    end_marker = float(config["markers"]["end"])
    reject_marker = float(config["markers"]["reject"])
    progress = f"Run {run_index}｜有效 Trial {valid_done + 1}/{valid_target}｜B 作废｜Esc 结束"
    event.clearEvents()

    baseline_s = rng.uniform(
        float(timing["baseline_min_s"]), float(timing["baseline_max_s"])
    )
    logger.log(
        event="baseline_start",
        run=run_index,
        planned_trial=trial_index,
        attempt=attempt,
        gesture=gesture,
        detail=f"duration_s={baseline_s:.3f}",
    )
    control = show_for_duration(
        win,
        core,
        event,
        lambda: draw_message(
            win,
            visual,
            title="·",
            subtitle="手指自然微弯，前臂和手腕保持不动",
            title_color="#D9E2EC",
            font=font,
            progress=progress,
        ),
        baseline_s,
    )
    if control == "escape":
        return "abort"
    if control == "reject":
        logger.log(
            event="rejected_before_marker",
            run=run_index,
            planned_trial=trial_index,
            attempt=attempt,
            gesture=gesture,
            valid=False,
            detail="B pressed during baseline; no UDP marker sent",
        )
        show_rejected(win, visual, core, font, "在基线阶段按下 B；未发送动作 Marker")
        return "reject"

    logger.log(
        event="cue_start",
        run=run_index,
        planned_trial=trial_index,
        attempt=attempt,
        gesture=gesture,
        detail=f"duration_s={float(timing['cue_s']):.3f}",
    )
    control = show_for_duration(
        win,
        core,
        event,
        lambda: draw_message(
            win,
            visual,
            title=f"准备：{gesture.name_cn}\n{gesture.name_en}",
            subtitle="保持放松，不要提前动作",
            title_color="#FFD166",
            font=font,
            progress=progress,
        ),
        float(timing["cue_s"]),
    )
    if control == "escape":
        return "abort"
    if control == "reject":
        logger.log(
            event="rejected_before_marker",
            run=run_index,
            planned_trial=trial_index,
            attempt=attempt,
            gesture=gesture,
            valid=False,
            detail="B pressed during cue; no UDP marker sent",
        )
        show_rejected(win, visual, core, font, "在提示阶段按下 B；未发送动作 Marker")
        return "reject"

    action_title = f"现在做：{gesture.name_cn}\n{gesture.name_en}"
    draw_message(
        win,
        visual,
        title=action_title,
        subtitle=f"{gesture.instruction}\n保持到屏幕提示放松",
        title_color=gesture.color,
        font=font,
        progress=progress,
    )
    win.callOnFlip(
        send_and_log_marker,
        sender,
        logger,
        gesture.marker,
        "action_start_marker",
        run_index,
        trial_index,
        attempt,
        gesture,
        None,
        "Marker scheduled on action-screen flip",
    )
    action_flip_time = win.flip()
    logger.log(
        event="action_screen_flip",
        run=run_index,
        planned_trial=trial_index,
        attempt=attempt,
        gesture=gesture,
        detail=f"psychopy_flip_time={action_flip_time}",
    )

    action_clock = core.Clock()
    while action_clock.getTime() < float(timing["action_s"]):
        control = poll_control_keys(event)
        if control in ("reject", "escape"):
            reason = "manual_B" if control == "reject" else "escape_during_action"
            send_and_log_marker(
                sender,
                logger,
                reject_marker,
                "reject_marker",
                run_index,
                trial_index,
                attempt,
                gesture,
                False,
                reason,
            )
            core.wait(0.04)
            send_and_log_marker(
                sender,
                logger,
                end_marker,
                "end_marker_after_reject",
                run_index,
                trial_index,
                attempt,
                gesture,
                False,
                reason,
            )
            if control == "escape":
                return "abort"
            show_rejected(win, visual, core, font, "动作阶段检测到 B，已发送 90 和 20")
            return "reject"
        core.wait(0.004)

    draw_message(
        win,
        visual,
        title="放松",
        subtitle="回到手指自然微弯状态",
        title_color="#66C2A5",
        font=font,
        progress=progress,
    )
    win.callOnFlip(
        send_and_log_marker,
        sender,
        logger,
        end_marker,
        "end_marker",
        run_index,
        trial_index,
        attempt,
        gesture,
        True,
        "Marker scheduled on return-screen flip",
    )
    win.flip()
    return_clock = core.Clock()
    while return_clock.getTime() < float(timing["return_s"]):
        control = poll_control_keys(event)
        if control == "escape":
            return "abort"
        core.wait(0.004)

    logger.log(
        event="trial_valid",
        run=run_index,
        planned_trial=trial_index,
        attempt=attempt,
        gesture=gesture,
        valid=True,
        detail=(
            f"recommended_training_window_s="
            f"{float(timing['training_start_after_marker_s']):.3f}-"
            f"{float(timing['training_end_after_marker_s']):.3f}"
        ),
    )
    return "valid"


def run_experiment(
    config: Dict[str, Any],
    config_path: Path,
    args: argparse.Namespace,
) -> Path:
    core, event, visual = import_psychopy()
    validate_config(config)
    gestures = build_gestures(config)
    protocol = config["protocol"][args.mode]
    runs = int(protocol["runs"])
    trials_per_run = int(protocol["trials_per_run"])
    seed = int(args.seed if args.seed is not None else config["protocol"]["random_seed"])
    schedule = build_schedule(runs, trials_per_run, seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root).expanduser().resolve()
    output_dir = output_root / f"RPS_{args.subject}_session-{args.session}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    save_json(output_dir / "config_used.json", config)
    write_schedule_csv(output_dir / "planned_schedule.csv", schedule, gestures)
    write_run_markers_csv(output_dir / "run_markers.csv", config, runs)

    logger = EventLogger(output_dir / "events.csv", args.subject, args.session, args.mode)
    sender = MarkerSender(
        str(config["network"]["ip"]),
        int(config["network"]["port"]),
        enabled=not args.no_udp,
    )

    display = config.get("display", {})
    full_screen = bool(display.get("full_screen", True)) and not args.windowed
    size = tuple(display.get("window_size", [1280, 720]))
    font = str(display.get("font", "Microsoft YaHei"))
    win = visual.Window(
        size=size,
        fullscr=full_screen,
        screen=int(display.get("screen", 0)),
        color=str(display.get("background_color", "#101820")),
        units="height",
        allowGUI=not full_screen,
    )
    win.mouseVisible = False
    rng = random.Random(seed + 101)
    status = "completed"
    valid_count = 0
    rejected_count = 0
    completed_runs = 0
    active_run_index = 0
    aborted_run_index = 0

    logger.log(
        event="session_start",
        detail=(
            f"config={config_path.name}; seed={seed}; udp_enabled={not args.no_udp}; "
            f"udp_target={sender.ip}:{sender.port}"
        ),
    )

    try:
        wait_for_space(
            win,
            visual,
            event,
            "石头・布 sEMG 采集",
            (
                f"受试者：{args.subject}　Session：{args.session}　模式：{args.mode}\n"
                "保持肘部和手腕不动，只改变手指姿势。"
            ),
            font,
        )
        run_practice(win, visual, core, event, config, gestures, rng, font)
        wait_for_space(
            win,
            visual,
            event,
            "准备正式采集",
            (
                "请确认 OpenBCI GUI 已开始数据流和原始数据记录，\n"
                f"Marker Widget 正在监听 {sender.ip}:{sender.port}。"
            ),
            font,
        )

        max_retries = int(config["protocol"].get("max_retries_per_trial", 3))
        for run_index, run_sequence in enumerate(schedule, start=1):
            active_run_index = run_index
            run_start_marker = run_marker_value(config, "start", run_index)
            draw_message(
                win,
                visual,
                title=f"Run {run_index} 开始",
                subtitle=f"即将采集 {trials_per_run} 个有效Trial，请保持姿势稳定",
                title_color="#66C2A5",
                font=font,
            )
            win.callOnFlip(
                send_and_log_run_marker,
                sender,
                logger,
                run_start_marker,
                "run_start_marker",
                run_index,
                f"target_valid={trials_per_run}; marker_on_screen_flip",
                None,
            )
            win.flip()
            core.wait(float(config["timing"]["run_boundary_display_s"]))

            pending: List[Tuple[int, str, int]] = [
                (planned_trial, key, 1)
                for planned_trial, key in enumerate(run_sequence, start=1)
            ]
            valid_in_run = 0
            while pending:
                trial_index, key, attempt = pending.pop(0)
                gesture = gestures[key]
                result = run_trial(
                    win=win,
                    visual=visual,
                    core=core,
                    event=event,
                    config=config,
                    sender=sender,
                    logger=logger,
                    gesture=gesture,
                    run_index=run_index,
                    trial_index=trial_index,
                    attempt=attempt,
                    valid_done=valid_in_run,
                    valid_target=trials_per_run,
                    rng=rng,
                    font=font,
                )
                if result == "abort":
                    raise ExperimentAbort("实验过程中按 Esc 结束。")
                if result == "valid":
                    valid_in_run += 1
                    valid_count += 1
                else:
                    rejected_count += 1
                    if attempt >= max_retries:
                        raise ExperimentAbort(
                            f"Run {run_index} Trial {trial_index} 已达到最大补采次数 {max_retries}。"
                        )
                    pending.append((trial_index, key, attempt + 1))

            run_end_marker = run_marker_value(config, "end", run_index)
            draw_message(
                win,
                visual,
                title=f"Run {run_index} 完成",
                subtitle=f"本Run已完成 {valid_in_run} 个有效Trial",
                title_color="#66C2A5",
                font=font,
            )
            win.callOnFlip(
                send_and_log_run_marker,
                sender,
                logger,
                run_end_marker,
                "run_end_marker",
                run_index,
                f"valid={valid_in_run}; rejected_total={rejected_count}; marker_on_screen_flip",
                True,
            )
            win.flip()
            core.wait(float(config["timing"]["run_boundary_display_s"]))
            completed_runs += 1
            active_run_index = 0
            if run_index < runs:
                wait_for_space(
                    win,
                    visual,
                    event,
                    f"Run {run_index} 完成",
                    (
                        f"本Run已完成 {valid_in_run} 个有效Trial。\n"
                        "建议休息2–3分钟；准备好后继续。"
                    ),
                    font,
                )

        draw_message(
            win,
            visual,
            title="采集完成",
            subtitle=(
                f"有效Trial：{valid_count}　作废尝试：{rejected_count}\n"
                "请停止 OpenBCI GUI 记录并检查文件。"
            ),
            title_color="#66C2A5",
            font=font,
        )
        win.flip()
        core.wait(2.5)
    except ExperimentAbort as exc:
        status = "aborted"
        logger.log(event="session_aborted", valid=False, detail=str(exc))
        draw_message(
            win,
            visual,
            title="实验已安全结束",
            subtitle="事件日志已保存。请停止 OpenBCI GUI 的原始数据记录。",
            title_color="#FFB347",
            font=font,
        )
        if active_run_index:
            aborted_run_index = active_run_index
            win.callOnFlip(
                send_and_log_run_marker,
                sender,
                logger,
                run_marker_value(config, "abort", active_run_index),
                "run_abort_marker",
                active_run_index,
                f"reason={exc}; marker_on_abort_screen_flip",
                False,
            )
        win.flip()
        core.wait(1.2)
    except Exception as exc:
        status = "error"
        logger.log(event="session_error", valid=False, detail=repr(exc))
        if active_run_index:
            aborted_run_index = active_run_index
            try:
                send_and_log_run_marker(
                    sender,
                    logger,
                    run_marker_value(config, "abort", active_run_index),
                    "run_abort_marker",
                    active_run_index,
                    f"program_error={exc!r}; sent_without_screen_flip",
                    False,
                )
            except Exception as marker_exc:
                logger.log(
                    event="run_abort_marker_failed",
                    run=active_run_index,
                    valid=False,
                    detail=repr(marker_exc),
                )
        raise
    finally:
        logger.log(
            event="session_end",
            valid=(status == "completed"),
            detail=f"status={status}; valid={valid_count}; rejected={rejected_count}",
        )
        logger.close()
        sender.close()
        win.close()
        save_json(
            output_dir / "session_summary.json",
            {
                "status": status,
                "subject": args.subject,
                "session": args.session,
                "mode": args.mode,
                "seed": seed,
                "valid_trials": valid_count,
                "rejected_attempts": rejected_count,
                "completed_runs": completed_runs,
                "aborted_run": aborted_run_index or None,
                "expected_valid_trials": runs * trials_per_run,
                "run_marker_map": build_run_marker_map(config, runs),
                "udp_enabled": not args.no_udp,
                "udp_ip": config["network"]["ip"],
                "udp_port": config["network"]["port"],
                "output_directory": str(output_dir),
                "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
        )
    return output_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PsychoPy 石头/布 sEMG 采集，并向 OpenBCI GUI 发送 UDP Marker。"
    )
    parser.add_argument("--config", type=Path, default=None, help="JSON配置文件。")
    parser.add_argument("--mode", choices=["pilot", "formal"], default="formal")
    parser.add_argument("--subject", default="S01", help="受试者编号，例如 S01。")
    parser.add_argument("--session", default="01", help="Session编号，例如 01。")
    parser.add_argument("--seed", type=int, default=None, help="覆盖配置中的随机种子。")
    parser.add_argument(
        "--output-root",
        default=str(SCRIPT_DIR / "collection_output"),
        help="本程序事件日志输出目录；OpenBCI原始数据仍由GUI保存。",
    )
    parser.add_argument("--windowed", action="store_true", help="窗口模式测试，不使用全屏。")
    parser.add_argument("--no-udp", action="store_true", help="仅测试视觉流程，不发送Marker。")
    parser.add_argument("--test-markers", action="store_true", help="发送11/13/20/90后退出。")
    parser.add_argument("--test-interval", type=float, default=1.0, help="Marker测试间隔秒数。")
    parser.add_argument("--self-test", action="store_true", help="不打开PsychoPy，执行内部自测。")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve() if args.config else Path(__file__).resolve()
    if args.config and not config_path.exists():
        print(f"错误：找不到配置文件：{config_path}", file=sys.stderr)
        return 2
    try:
        config = load_json(config_path) if args.config else {'experiment_name': 'OpenBCI_Rock_Paper_sEMG_2Class', 'network': {'ip': '127.0.0.1', 'port': 12350}, 'markers': {'rock': 11, 'paper': 13, 'end': 20, 'reject': 90}, 'run_markers': {'start_base': 100, 'end_base': 110, 'abort_base': 120}, 'timing': {'baseline_min_s': 2.5, 'baseline_max_s': 3.5, 'cue_s': 1.0, 'action_s': 4.0, 'return_s': 0.5, 'training_start_after_marker_s': 0.5, 'training_end_after_marker_s': 3.5, 'run_boundary_display_s': 0.5}, 'protocol': {'random_seed': 20260823, 'practice_per_class': 2, 'max_retries_per_trial': 3, 'pilot': {'runs': 2, 'trials_per_run': 12}, 'formal': {'runs': 4, 'trials_per_run': 12}}, 'display': {'full_screen': True, 'screen': 0, 'window_size': [1280, 720], 'background_color': '#101820', 'font': 'Microsoft YaHei'}}
        validate_config(config)
        if args.self_test:
            run_self_test(config)
            return 0
        if args.test_markers:
            if args.test_interval <= 0:
                raise ValueError("--test-interval 必须大于0。")
            test_markers(config, args.test_interval)
            return 0
        output_dir = run_experiment(config, config_path, args)
        print(f"实验日志已保存：{output_dir}")
        return 0
    except KeyboardInterrupt:
        print("\n程序被键盘中断。请停止 OpenBCI GUI 记录。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
