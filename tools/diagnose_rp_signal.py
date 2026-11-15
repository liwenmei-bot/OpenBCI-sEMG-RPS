#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核对石头/布实时日志：相同原始样本 -> 相同滤波 -> RMS/活动分数。

先在识别界面停止并保存，再运行：
  python diagnose_rp_signal.py --logs "D:\\肌电\\news\\new2"
也可把--logs指向某次运行的子目录。父目录下有多次运行时默认检查最新一份。
仅依赖numpy、pandas、scipy；不导入实时代码、不加载SVM、不连接板卡、
不改模型/门控/输入日志。只有输出报告会写入新建的诊断目录。
截图中的GUI滤波/时间窗、实际动作时刻不在日志里，不能据此自动证明
GUI与Python物理同步，也不能把LSL时间戳年龄当作肌肉起始延迟。
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import sys

try:
    import numpy as np
    import pandas as pd
    from scipy.signal import sosfilt, sosfilt_zi, welch
except ImportError as exc:
    raise SystemExit('缺少依赖：%s；请运行 python -m pip install numpy pandas scipy' % exc)

VERSION = '1.0.0-rp-signal-audit'
RAW_COLS = ['ch%d_uv' % channel for channel in range(1, 5)]
RMS_COLS = ['ch%d_RMS' % channel for channel in range(1, 5)]
FILES = ['live_configuration.json', 'raw_samples.csv', 'window_trace.csv']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def safe(value):
    if isinstance(value, dict): return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [safe(item) for item in value]
    if isinstance(value, np.ndarray): return safe(value.tolist())
    if isinstance(value, np.generic): return safe(value.item())
    if isinstance(value, Path): return str(value)
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): result.update(block)
    return result.hexdigest()


def file_state(paths):
    return {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}


def find_session(root):
    root = Path(root).expanduser().resolve()
    require(root.is_dir(), '目录不存在：' + str(root))
    if all((root/name).is_file() for name in FILES): return root
    found = [path.parent for path in root.rglob('live_configuration.json')
        if all((path.parent/name).is_file() for name in FILES)]
    require(found, '未找到完整实时日志；目录内需要live_configuration.json、raw_samples.csv、window_trace.csv。')
    return max(found, key=lambda path: (path/'raw_samples.csv').stat().st_mtime_ns)


def bool_column(frame, key):
    if key not in frame: return np.zeros(len(frame), bool)
    values = frame[key]
    if values.dtype == bool: return values.to_numpy()
    text = values.astype(str).str.strip().str.casefold()
    require(text.isin(['true', 'false', '1', '0']).all(), '布尔日志字段无效：' + key)
    return text.isin(['true', '1']).to_numpy()


def numeric(frame, key):
    require(key in frame, '日志缺少字段：' + key)
    return pd.to_numeric(frame[key], errors='coerce').to_numpy(dtype=float)


def quantiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values): return {'count': 0, 'p10': None, 'median': None, 'p95': None, 'maximum': None}
    return dict(count=len(values), p10=float(np.quantile(values, .1)), median=float(np.median(values)),
        p95=float(np.quantile(values, .95)), maximum=float(values.max()))


def settings(info, session):
    params = dict(info.get('runtime_detector_settings') or {})
    if not params.get('preprocess_config'):
        value = info.get('detector_path')
        if value:
            path = Path(value).expanduser()
            if not path.is_absolute(): path = session/path
            if path.is_file():
                fallback = read_json(path)
                # Recorded runtime settings take precedence over original offline settings.
                params = {**fallback, **params}
    config = params.get('preprocess_config') or info.get('preprocess_config')
    require(isinstance(config, dict), '日志未包含预处理参数，原门控文件也无法定位；请保留配套detector JSON。')
    required = ['sos', 'fs', 'n_channels', 'window_samples', 'hop_samples', 'warmup_samples']
    require(all(key in config for key in required), '预处理参数不完整。')
    require(config['n_channels'] == 4, '只支持四通道日志。')
    require(config.get('filter_mode', 'causal_sos') == 'causal_sos', '当前独立复算只支持训练时的causal_sos滤波。')
    return params, config


def independent_filter(raw, flags, invalid, config, extra_breaks=()):
    """Filter each full contiguous segment independently of the live chunk code."""
    raw = np.asarray(raw, dtype=float)
    valid = np.isfinite(raw).all(axis=1) & ~np.asarray(invalid, bool)
    starts = np.asarray(flags, bool).copy()
    starts[0] = True
    starts[1:] |= ~valid[:-1]
    for row in extra_breaks:
        if 0 <= row < len(starts): starts[row] = True
    boundaries = np.unique(np.r_[0, len(raw), np.flatnonzero(starts),
        np.flatnonzero(valid[1:] != valid[:-1])+1])
    filtered = np.full_like(raw, np.nan)
    segment = np.full(len(raw), -1, dtype=int)
    ready = np.zeros(len(raw), bool)
    sos = np.asarray(config['sos'], dtype=float)
    require(sos.ndim == 2 and sos.shape[1] == 6 and np.isfinite(sos).all(), 'SOS参数无效。')
    warmup = int(config['warmup_samples'])
    for index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if stop <= start or not valid[start]: continue
        zi = sosfilt_zi(sos)[:, :, None] * raw[start][None, None, :]
        filtered[start:stop], _ = sosfilt(sos, raw[start:stop], axis=0, zi=zi)
        segment[start:stop] = index
        ready[min(stop, start+warmup):stop] = True
    return filtered, segment, ready


def spectrum_table(raw, filtered, segment, fs):
    indices = np.flatnonzero(segment >= 0)
    if not len(indices): return pd.DataFrame()
    last = indices[-1]
    selected = np.flatnonzero(segment == segment[last])[-int(10*fs):]
    if len(selected) < max(50, int(fs)): return pd.DataFrame()
    bands = [(1, 20), (20, 45), (45, 55), (55, min(100., fs/2)), (100., fs/2)]
    rows = []
    integrate = getattr(np, 'trapezoid', None) or np.trapz
    for kind, data in [('input_before_model_filter', raw), ('after_model_filter', filtered)]:
        freq, psd = welch(data[selected], fs=fs, axis=0, nperseg=min(1024, len(selected)), detrend='constant')
        for channel in range(4):
            for lo, hi in bands:
                if hi <= lo: continue
                mask = (freq >= lo) & (freq <= hi)
                power = float(integrate(psd[mask, channel], freq[mask])) if mask.sum() >= 2 else None
                rows.append({'signal': kind, 'channel': channel+1, 'low_hz': lo, 'high_hz': hi,
                    'power_uv_squared': power, 'sample_count': len(selected),
                    'scope': 'last_contiguous_segment_at_most_10_seconds_not_verified_rest'})
    return pd.DataFrame(rows)


def audit(session):
    session = Path(session).resolve()
    paths = [session/name for name in FILES]
    initial = file_state(paths)
    info = read_json(paths[0]); params, config = settings(info, session)
    samples = pd.read_csv(paths[1], encoding='utf-8-sig')
    trace = pd.read_csv(paths[2], encoding='utf-8-sig')
    require(len(samples) > 0 and len(trace) > 0, '日志尚无完整分析窗口；请停止并保存后再检查。')
    raw = np.column_stack([numeric(samples, name) for name in RAW_COLS])
    recorded_rows = numeric(samples, 'received_row')
    require(np.array_equal(recorded_rows, np.arange(len(samples))), '原始日志行号不连续，不能按现有窗口行号准确复算。')
    stamps = numeric(samples, 'lsl_time_s')
    received = numeric(samples, 'received_local_clock_s')
    flags = bool_column(samples, 'break_before')
    invalid = bool_column(samples, 'timestamp_invalid') | bool_column(samples, 'discarded_stale')
    extra_breaks = []
    transport_path = session/'transport_issues.csv'
    if transport_path.is_file():
        transport = pd.read_csv(transport_path, encoding='utf-8-sig')
        if {'reason', 'received_row'} <= set(transport):
            manual = transport.reason.astype(str).str.startswith('user_requested_rest_calibration_revision_')
            extra_breaks = transport.loc[manual, 'received_row'].astype(int).tolist()
    filtered, segments, ready = independent_filter(raw, flags, invalid, config, extra_breaks)
    fs, width = float(config['fs']), int(config['window_samples'])
    expected = np.full((len(trace), 4), np.nan)
    raw_std, raw_mean = np.full_like(expected, np.nan), np.full_like(expected, np.nan)
    valid_window = np.zeros(len(trace), bool)
    starts = numeric(trace, 'start_row'); stops = numeric(trace, 'stop_row')
    decision_rows = numeric(trace, 'decision_row')
    reason = [''] * len(trace)
    for index, (start, stop, decision) in enumerate(zip(starts, stops, decision_rows)):
        if not np.isfinite([start, stop, decision]).all() or any(value != int(value) for value in [start, stop, decision]):
            reason[index] = 'invalid_window_indices'; continue
        start, stop, decision = int(start), int(stop), int(decision)
        if start < 0 or stop > len(raw) or stop-start != width or decision != stop-1:
            reason[index] = 'window_out_of_bounds_or_wrong_width'; continue
        if segments[start] < 0 or segments[start] != segments[stop-1] or not ready[start:stop].all():
            reason[index] = 'window_crosses_reset_invalid_data_or_warmup'; continue
        # The training/live contract casts the filtered window to float32 before
        # computing features with float64 accumulation. Keep this exact order.
        window = filtered[start:stop].astype(np.float32).astype(np.float64)
        expected[index] = np.sqrt(np.mean(window*window, axis=0))
        raw_std[index] = raw[start:stop].std(axis=0)
        raw_mean[index] = raw[start:stop].mean(axis=0)
        valid_window[index] = True
    observed = np.column_stack([numeric(trace, name) for name in RMS_COLS])
    finite_pair = valid_window[:, None] & np.isfinite(expected) & np.isfinite(observed)
    difference = np.abs(expected-observed)
    close = np.isclose(expected, observed, rtol=1e-6, atol=1e-6)
    bad_rms = finite_pair & ~close
    compare = trace[[key for key in ['start_row', 'stop_row', 'decision_row', 'detector_state', 'qc_status',
        'activity_score', 'trigger_event_id', 'data_age_s'] if key in trace]].copy()
    compare['recomputed'] = valid_window; compare['recompute_note'] = reason
    for channel in range(4):
        for label, values in [('input_std', raw_std), ('input_mean', raw_mean),
                ('recomputed_rms', expected), ('logged_rms', observed), ('absolute_error', difference)]:
            compare['ch%d_%s_uv' % (channel+1, label)] = values[:, channel]
    mode = (info.get('activity_gate_options') or {}).get('mode', info.get('activity_mode', 'legacy'))
    noise = np.asarray(params.get('noise_reference_rms_uv', []), dtype=float)
    activity_error = None
    if noise.shape == (4,) and np.isfinite(noise).all() and (noise > 0).all():
        ratios = expected/noise
        activities = np.sort(ratios, axis=1)[:, -2:].mean(axis=1)
        compare['legacy_activity_recomputed'] = activities
        for channel in range(4): compare['ch%d_rms_to_reference_ratio' % (channel+1)] = ratios[:, channel]
        if mode == 'legacy' and 'activity_score' in trace:
            logged = numeric(trace, 'activity_score')
            matchable = valid_window & np.isfinite(logged) & np.isfinite(activities)
            activity_error = {'compared_windows': int(matchable.sum()),
                'mismatched_windows': int((matchable & ~np.isclose(activities, logged, rtol=1e-6, atol=1e-6)).sum()),
                'absolute_error': quantiles(np.abs(activities[matchable]-logged[matchable]))}
    window_ages = numeric(trace, 'data_age_s') if 'data_age_s' in trace else np.full(len(trace), np.nan)
    processed = numeric(trace, 'processed_local_clock_s') if 'processed_local_clock_s' in trace else None
    decision_time = numeric(trace, 'decision_lsl_time_s') if 'decision_lsl_time_s' in trace else None
    eligible_rows = np.isfinite(decision_rows) & (decision_rows >= 0) & (decision_rows < len(raw))
    timestamp_errors = 0
    if decision_time is not None and eligible_rows.any():
        timestamp_errors = int((~np.isclose(stamps[decision_rows[eligible_rows].astype(int)],
            decision_time[eligible_rows], atol=1e-6, rtol=0)).sum())
    on, off = float(params.get('on_threshold', np.nan)), float(params.get('off_threshold', np.nan))
    activity = numeric(trace, 'activity_score') if 'activity_score' in trace else np.full(len(trace), np.nan)
    maximum_age = float(info.get('max_data_age_seconds', .75))
    findings = []
    def add(level, code, message): findings.append({'level': level, 'code': code, 'message': message})
    count = int(finite_pair.sum())
    if not count: add('ERROR', 'no_comparable_windows', '没有可复算的共同窗口，暂时不能判断RMS计算是否一致。')
    elif bad_rms.any(): add('ERROR', 'rms_replay_mismatch', '保存的原始输入独立复算后，与窗口日志RMS不一致；需检查日志配对、滤波重置和窗口索引。')
    else: add('PASS', 'rms_replay_matches', '保存的原始输入经同样滤波后，RMS与窗口日志一致；这不证明GUI和板卡物理同步。')
    if activity_error and activity_error['mismatched_windows']:
        add('ERROR', 'activity_replay_mismatch', 'legacy活动分数与四通道RMS/旧参考的复算结果不一致。')
    elif activity_error: add('PASS', 'activity_replay_matches', 'legacy活动值与配置内四通道旧参考的复算结果一致。')
    if (~valid_window).any(): add('WARNING', 'unrecomputed_windows', '部分窗口跨越无效数据/重置或不满足窗口契约，详见window_rms_check.csv。')
    if timestamp_errors: add('ERROR', 'window_timestamp_mismatch', '窗口记录的LSL时刻与原始行号对应的时间戳不一致。')
    if np.isfinite(window_ages).any() and np.nanquantile(window_ages, .95) > maximum_age:
        add('WARNING', 'old_analysis_windows', '至少5%的分析窗口时间戳年龄超过运行门限；检查缓存、处理速度和中断记录。')
    if bool_column(samples, 'discarded_stale').any(): add('WARNING', 'stale_input', '曾接收到过时/未来时间戳数据，实时程序已记录抑制标记。')
    if info.get('same_calendar_date_as_calibration') is False:
        add('CHECK', 'cross_day_reference', '本次运行日期与模型校准日期不同；旧静息参考可能不适合现在，但没有真实放松标注，不能直接确认。')
    add('CHECK', 'gui_processing_unknown', '需要核对GUI Filters、RMS统计窗口和Networking Data Type；流名rps_raw本身不能证明发送的是TimeSeriesRaw。')
    add('LIMIT', 'upstream_latency_unknown', 'LSL年龄和Python缓存不能排除GUI/板卡上游等待；当前日志不含独立的物理动作起始真值。')
    add('LIMIT', 'rest_truth_unknown', '没有实际放松/动作的时间标注，不计算识别准确率、真实静息误触发率或肌肉起始延迟。')
    elapsed = float(received[-1]-received[0]) if len(received) > 1 else 0.
    observed_rate = float((len(raw)-1)/elapsed) if elapsed > 0 else None
    if elapsed >= 10 and observed_rate is not None and observed_rate < fs*.9:
        add('WARNING', 'low_receive_rate', '按本机接收时钟估计的整体采样接收率明显低于标称；无法仅据此区分丢样、暂停与上游积压。')
    report = {'schema': 'openbci_rp_signal_audit_v1', 'script_version': VERSION,
        'session_directory': str(session), 'source_script_version': info.get('script_version'),
        'source_stream': info.get('stream'), 'channels_one_based': info.get('channels_one_based'),
        'input_unit': info.get('input_unit'), 'model_calibration_date': info.get('calibration_date'),
        'activity_mode': mode, 'processing': {key: config.get(key) for key in ['fs', 'filter_mode',
            'low_hz', 'high_hz', 'notch_hz', 'window_samples', 'hop_samples', 'warmup_samples']},
        'rows': len(raw), 'analysis_windows': len(trace), 'recomputed_windows': int(valid_window.sum()),
        'rms_consistency': {'compared_channel_windows': count, 'mismatched_channel_windows': int(bad_rms.sum()),
            'absolute_error_uv': quantiles(difference[finite_pair]),
            'tolerance': 'atol=1e-6uV, rtol=1e-6'},
        'legacy_activity_consistency': activity_error,
        'source_window_timestamp_mismatches': timestamp_errors,
        'sample_timestamp_age_at_reception_seconds': quantiles(received-stamps),
        'window_timestamp_age_at_processing_seconds': quantiles(window_ages),
        'reported_window_age_consistency_error_seconds': quantiles((processed-decision_time)-window_ages)
            if processed is not None and decision_time is not None else None,
        'receive_clock_duration_seconds': elapsed, 'overall_receive_rate_estimate_hz': observed_rate,
        'receive_rate_is_not_board_packet_loss_or_physical_delay': True,
        'thresholds': {'on': on, 'off': off}, 'noise_reference_rms_uv': noise.tolist(),
        'activity_quantiles_not_accuracy': quantiles(activity),
        'gate_counts': {'below_or_equal_off': int((activity <= off).sum()),
            'between_off_and_on': int(((activity > off) & (activity < on)).sum()),
            'above_or_equal_on': int((activity >= on).sum())},
        'detector_state_counts': trace.detector_state.value_counts().to_dict() if 'detector_state' in trace else {},
        'channel_statistics': [{'channel': index+1, 'input_std_uv_per_analysis_window': quantiles(raw_std[:, index]),
            'model_filtered_rms_uv_per_analysis_window': quantiles(expected[:, index]),
            'legacy_reference_rms_uv': noise[index] if noise.shape == (4,) else None} for index in range(4)],
        'findings': findings, 'model_loaded_or_fitted': False, 'input_files_modified': False}
    bands = spectrum_table(raw, filtered, segments, fs)
    require(file_state(paths) == initial, '日志检查期间仍在变化；请先在识别界面停止并保存，再重新运行诊断。')
    report['input_sha256'] = {path.name: digest(path) for path in paths}
    require(file_state(paths) == initial, '日志仍在写入；请停止并保存后重新诊断。')
    return safe(report), compare, bands


def report_html(report):
    esc = lambda value: html.escape(str(value))
    rows = ''.join('<tr><td>%s</td><td>%s</td></tr>' % (esc(item['level']), esc(item['message']))
        for item in report['findings'])
    channels = ''.join('<tr><td>CH%d</td><td>%s</td><td>%s</td><td>%s</td></tr>' %
        (item['channel'], esc(item['input_std_uv_per_analysis_window']['median']),
        esc(item['model_filtered_rms_uv_per_analysis_window']['median']), esc(item['legacy_reference_rms_uv']))
        for item in report['channel_statistics'])
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>肌电实时信号核对</title>
<style>body{font:16px/1.7 system-ui,sans-serif;max-width:1100px;margin:30px auto;padding:0 20px;color:#17273a}
table{border-collapse:collapse;width:100%%;margin:16px 0}td,th{border:1px solid #ccd5df;padding:9px;text-align:left}
th{background:#edf3f8}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:16px}</style>
<h1>肌电实时信号核对</h1><p>输入记录：%s</p><p>本报告只核对保存的输入与实时算法，不修改模型和阈值。</p>
<h2>检查结果</h2><table><tr><th>结果</th><th>说明</th></tr>%s</table>
<h2>四通道对照</h2><p>下表是整条记录各分析窗口的中位数，未区分真实动作和静息。输入STD消除了直流偏置；
模型RMS使用模型规定的滤波。两者不能直接当作GUI显示值。</p>
<table><tr><th>通道</th><th>输入STD中位数 μV</th><th>模型滤波RMS中位数 μV</th><th>旧基线参考 μV</th></tr>%s</table>
<h2>怎样继续排查</h2><ol><li>若复算不一致，先核对原始记录、窗口和滤波重置，不调整分类阈值。</li>
<li>若复算一致，说明大RMS确实来自Python保存的输入及其滤波；继续核对GUI Networking的数据类型、通道、单位及Filters。</li>
<li>在同一段放松和动作期间对照；截图和不同时间窗的数值不能直接证明同步。</li>
<li>只有确认信号和时间对齐后，才检查当前静息参考是否需要重新测量。当前工具不会自动改它。</li></ol>
<h2>完整结果</h2><pre>%s</pre></html>''' % (esc(report['session_directory']), rows, channels,
        esc(json.dumps(report, ensure_ascii=False, indent=2)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--logs', required=True, help='具体会话目录或包含多个实时输出目录的父目录；默认选最新完整会话')
    parser.add_argument('--output', help='诊断输出父目录；默认放在所选会话旁的RP_SIGNAL_DIAG目录')
    args = parser.parse_args(argv)
    try:
        session = find_session(args.logs)
        print('检查实时记录：' + str(session), flush=True)
        report, windows, bands = audit(session)
        parent = Path(args.output).expanduser().resolve() if args.output else session.parent/'RP_SIGNAL_DIAG'
        output = parent/('RP_SIGNAL_DIAG_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        output.mkdir(parents=True, exist_ok=False)
        (output/'diagnose_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        windows.to_csv(output/'window_rms_check.csv', index=False, encoding='utf-8-sig')
        if len(bands): bands.to_csv(output/'frequency_band_power.csv', index=False, encoding='utf-8-sig')
        (output/'OPEN_REPORT.html').write_text(report_html(report), encoding='utf-8')
        print('程序版本：%s；检查窗口%d；校准日期%s。' % (report['source_script_version'], report['analysis_windows'], report['model_calibration_date']))
        for finding in report['findings']: print('[%s] %s' % (finding['level'], finding['message']))
        print('报告：' + str(output/'OPEN_REPORT.html'))
        print('请发回这个文件继续分析：' + str(output/'diagnose_report.json'))
        return 0
    except Exception as exc:
        print('[诊断未完成] %s: %s' % (type(exc).__name__, exc), file=sys.stderr)
        return 1


if __name__ == '__main__': raise SystemExit(main())
