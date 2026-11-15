"""Portable synthetic checks; no participant recordings, fitted models or board needed."""
import importlib.util
import sys
import copy
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("rps_live", ROOT/"07_live_svm_rps_label_4ch.py")
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)

def measured_gate():
    gate = m.RestActivityGate(250, 50, seconds=2.)
    for i in range(20):
        gate.observe(np.array([2., 3., 4., 5.])*(1.+.02*(i%3-1)), 49+i*25, True)
    assert gate.state == 'ready', gate.report()
    return gate

class RestGateTests(unittest.TestCase):
    def test_single_arbitrarily_large_channel_cannot_authorize_action(self):
        gate = measured_gate()
        rms = np.array(gate.profile['median_rms_uv'])
        rms[3] = 1000000.
        score, z = gate.observe(rms, 1000, True)
        self.assertEqual(score, 0.)
        self.assertEqual(int(np.sum(z >= gate.on_z)), 1)
        rms[1] = 100.
        score, z = gate.observe(rms, 1025, True)
        self.assertGreater(score, gate.on_z)

    def test_isolated_calibration_spike_does_not_set_the_reference(self):
        gate = m.RestActivityGate(250, 50, seconds=2.)
        for i in range(20):
            rms = np.array([2., 3., 4., 5.])
            if i == 3: rms[3] = 10000.
            gate.observe(rms, 49+i*25, True)
        self.assertEqual(gate.state, 'ready')
        np.testing.assert_allclose(gate.profile['median_rms_uv'], [2, 3, 4, 5])
        self.assertTrue(np.isfinite(gate.observe([2, 3, 4, 5], 800, True)[0]))

    def test_invalid_windows_restart_the_contiguous_measurement(self):
        gate = m.RestActivityGate(250, 50, seconds=2.)
        for i in range(19): gate.observe([1., 2., 3., 4.], 49+i*25, True)
        self.assertEqual(gate.state, 'collecting')
        gate.observe([1., 2., 3., 4.], 524, False)
        gate.observe([1., 2., 3., 4.], 549, True)
        self.assertEqual(gate.state, 'collecting')
        self.assertAlmostEqual(gate.elapsed(), .2)
        self.assertEqual(gate.partial_resets, 1)

    def test_drifting_or_flat_baseline_blocks_output_until_explicit_retry(self):
        for pattern in ['drift', 'flat']:
            gate = m.RestActivityGate(250, 50, seconds=2.)
            for i in range(20):
                rms = np.ones(4)*(1. if i < 10 else 4.) if pattern == 'drift' else np.array([1., 1., 1., 0.])
                gate.observe(rms, 49+i*25, True)
            self.assertEqual(gate.state, 'failed', pattern)
            self.assertIsNone(gate.profile)
            for i in range(50): self.assertEqual(gate.observe([1., 1., 1., 1.], 1000+i*25, True), (None, None))
            gate.begin('manual_button', 3000)
            for i in range(20): gate.observe([1., 1., 1., 1.], 3049+i*25, True)
            self.assertEqual(gate.state, 'ready')
            self.assertEqual([x['status'] for x in gate.history], ['rejected', 'accepted'])

    def test_accepted_reference_does_not_adapt_during_activity(self):
        gate = measured_gate()
        before = copy.deepcopy(gate.profile)
        for i in range(500): gate.observe([100., 100., 200., 300.], 1000+25*i, True)
        self.assertEqual(gate.profile, before)
        self.assertEqual(gate.revision, 1)

    def test_display_uses_actual_state_instead_of_suggesting_another_vote(self):
        params = {'on_threshold':6., 'off_threshold':3., 'release_windows':3}
        view = m.activity_indicator(8., params, 'active_latched')
        self.assertIn('本次已输出', view['text'])
        self.assertNotIn('仍需等待与投票', view['text'])
        self.assertIn('类别投票尚未稳定', m.activity_indicator(8., params, 'unstable')['text'])

class UIRecalibrationTests(unittest.TestCase):
    def test_real_ui_callback_requests_remeasurement_and_displays_current_thresholds(self):
        widgets, variables, callbacks = [], [], []
        class Var:
            def __init__(self, value=''): self.value=value; variables.append(self)
            def set(self, value): self.value=value
        class Widget:
            def __init__(self, parent=None, **kw): self.kw=kw; widgets.append(self)
            def pack(self, **kw): pass
            def grid(self, **kw): pass
            def columnconfigure(self, *a, **kw): pass
            def configure(self, **kw): self.kw.update(kw)
            def __setitem__(self, k, v): self.kw[k]=v
        test = self
        class Root(Widget):
            def title(self, *a): pass
            def geometry(self, *a): pass
            def minsize(self, *a): pass
            def protocol(self, *a): pass
            def after(self, ms, fn): callbacks.append(fn)
            def destroy(self): pass
            def mainloop(self):
                callbacks.pop(0)()
                test.assertTrue(any('静息测量 2.0 / 8.0 秒' in str(v.value) for v in variables))
                button = next(w for w in widgets if w.kw.get('text') == '重新测静息（请先放松）')
                button.kw['command']()
                test.assertEqual(shared['live_controls'].get_nowait(), 'recalibrate_rest')
                snap = dict(snapshot)
                snap.update(state='active_latched', activity=8., gesture='石头',
                    activity_gate=dict(snapshot['activity_gate'], state='ready', on_rms_uv=[2.]*4, off_rms_uv=[1.5]*4))
                updates.put(snap); callbacks.pop(0)()
                test.assertTrue(any('本次已输出' in str(v.value) for v in variables))
                test.assertTrue(any('CH4 1.50/2.00' in str(v.value) for v in variables))
                stop.set(); shared['finished']=True; callbacks.pop(0)()
        class Worker:
            def start(self): updates.put(snapshot)
            def join(self, **kw): pass
        snapshot = {'state':'rest_calibrating', 'activity':None, 'samples':1000, 'event_count':0,
            'last_sample_lsl_time_s':1000., 'activity_gate':{'state':'collecting', 'elapsed_seconds':2.,
                'required_seconds':8., 'minimum_active_channels':2}, 'rms_uv':[1.]*4}
        args = m.build_live_parser().parse_args(['--activity-mode', 'robust'])
        stop=threading.Event(); updates=m.queue.Queue()
        shared={'posthoc_labels':m.PosthocLabels(args.class_names, clock=lambda:1000.05)}
        ttk=types.SimpleNamespace(**{k:Widget for k in ['Frame', 'Label', 'LabelFrame', 'Button', 'Progressbar']})
        tk=types.SimpleNamespace(Tk=Root, StringVar=Var, ttk=ttk)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'tkinter':tk}):
            m.show_window(args, {'subject_id':'synthetic', 'calibration_date':'2026-11-03', 'preprocess_config':{'fs':250}},
                {'on_threshold':21.951, 'off_threshold':14.268, 'release_windows':3},
                Path(tmp), stop, updates, shared, Worker())

if __name__ == "__main__": unittest.main(verbosity=2)
