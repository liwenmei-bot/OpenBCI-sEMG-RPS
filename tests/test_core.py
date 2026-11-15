"""Software behavior only: no participant records, board, GUI or measured accuracy."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


r = load('repo_binary_runner_tests', ROOT/'binary/run_rp_binary.py')
m = r.module('live')
entry = load('repo_root_entry_tests', ROOT/'run.py')


def params():
    return {'on_threshold': 4.3245, 'off_threshold': 2.5947, 'on_windows': 2,
        'release_windows': 3, 'vote_windows': 3, 'minimum_votes': 3,
        'settle_seconds': 0., 'settle_samples': 0, 'preprocess_config': {'fs': 250.}}


def feed(detector, first, count, activity, label):
    for index in range(first, first+count):
        detector.step(index*25, 1, True, activity, label, [-3., 3.] if label == 1 else [3., -3.])


class CoreTests(unittest.TestCase):
    def test_default_runtime_keeps_source_parameters(self):
        args = m.build_live_parser().parse_args([])
        self.assertEqual(m.runtime_detector_params(params(), args), params())

    def test_screenshot_profile_is_opt_in_and_never_changes_class_from_activity_height(self):
        args = m.build_live_parser().parse_args(['--gate-profile', 'trial'])
        p = m.runtime_detector_params(params(), args)
        self.assertEqual((p['on_threshold'], p['off_threshold']), (7.2, 7.0))
        self.assertFalse(p['runtime_gate_settings_independently_validated'])
        for activity, label in [(7.66, 0), (15.68, 1)]:
            d = m.GestureDetector(p); feed(d, 0, 10, 6.75, 0); feed(d, 10, 20, activity, label)
            self.assertEqual([event['class_id'] for event in d.events], [label])

    def test_release_then_action_emits_once_and_requires_real_release(self):
        p = m.runtime_detector_params(params(), m.build_live_parser().parse_args(['--gate-profile', 'trial']))
        d = m.GestureDetector(p)
        feed(d, 0, 20, 15.68, 0); self.assertFalse(d.events)
        feed(d, 20, 10, 6.75, 0); feed(d, 30, 100, 7.66, 1)
        self.assertEqual(len(d.events), 1)
        feed(d, 130, 4, 6.75, 1); self.assertIsNotNone(d.active_event)
        feed(d, 134, 1, 6.75, 1); self.assertIsNone(d.active_event)
        feed(d, 135, 20, 15.68, 0); self.assertEqual(len(d.events), 2)

    def test_invalid_runtime_thresholds_and_votes_are_rejected(self):
        for values in [['--gate-off','8'], ['--gate-on','nan'],
                ['--gate-vote-windows','5','--gate-min-votes','2'],
                ['--gate-profile','trial','--activity-mode','robust']]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                m.runtime_detector_params(params(), m.build_live_parser().parse_args(values))

    def test_binary_signed_margin_matches_svc_including_zero(self):
        pipeline = m.Pipeline([('scaler',m.StandardScaler()),('svm',m.SVC(kernel='linear',break_ties=True))])
        pipeline.fit([[-1.], [1.]], [0, 1])
        x = np.array([[-2.], [0.], [2.]])
        predicted, scores = m.predict(pipeline, x)
        np.testing.assert_array_equal(predicted, pipeline.predict(x))
        np.testing.assert_array_equal(scores[:,0], -scores[:,1])

    def test_posthoc_requires_existing_event_and_paper_maps_to_marker13(self):
        labels = m.PosthocLabels(['石头','布'])
        self.assertIsNone(labels.report()['summary']['record_based_accuracy'])
        with self.assertRaises(ValueError): labels.annotate(1, 1)
        labels.observe([{'event_id':1,'class_id':1}])
        self.assertEqual(labels.annotate(1,1)['actual_marker_code'], 13)
        labels.record_miss(0)
        self.assertEqual(labels.report()['summary']['confusion_matrix'], [[0,0,1],[0,1,0]])
        self.assertFalse(labels.report()['summary']['labels_used_for_inference'])

    def test_timestamp_jitter_is_warning_but_large_gap_resets(self):
        guard = m.TransportGuard(250.)
        flags, invalid, issues = guard.inspect([1000.,1000.004,1000.003,1000.4])
        self.assertFalse(flags[2]); self.assertTrue(flags[3]); self.assertFalse(invalid.any())
        self.assertTrue(any(item['reason']=='lsl_batch_timestamp_jitter_warning_only' for item in issues))

    def test_root_dispatch_preserves_separate_arguments(self):
        parts = entry.command(['day','--today-input','folder with spaces/example.txt'])
        self.assertEqual(parts[-1], 'folder with spaces/example.txt')
        self.assertEqual(parts[2], 'day')
        self.assertIn('run_rp_binary.py', parts[1])

    def _command(self, flags):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'pipeline_state.json').write_text(json.dumps({'schema':'openbci_rp_binary_pipeline_v1',
                'daily_model':'model path with spaces.joblib','detector':'gate path with spaces.json'}))
            with patch.object(r.subprocess,'run') as process, contextlib.redirect_stderr(io.StringIO()):
                process.return_value.returncode = 0
                code = r.main(['live','--run',str(root),*flags])
                return code, process.call_args[0][0] if process.called else None

    def test_runner_default_classic_does_not_receive_extended_flags(self):
        code, command = self._command([])
        self.assertEqual(code, 0)
        self.assertIn('07_live_rp_binary_label_4ch.py', command[1])
        self.assertEqual(command[command.index('--activity-mode')+1], 'legacy')
        self.assertEqual(command[command.index('--gate-profile')+1], 'source')
        self.assertNotIn('--trigger-mode', command); self.assertNotIn('--live-profile', command)

    def test_runner_extended_manual_receives_only_supported_switches(self):
        code, command = self._command(['--engine','extended','--trigger-mode','manual','--manual-duration-seconds','4'])
        self.assertEqual(code, 0)
        self.assertIn('extended', command[1])
        self.assertEqual(command[command.index('--trigger-mode')+1], 'manual')
        self.assertNotIn('--gate-profile', command)

    def test_runner_rejects_wrong_engine_options_before_subprocess(self):
        for flags in [['--trigger-mode','manual'],['--engine','extended','--gate-profile','trial']]:
            with self.subTest(flags=flags):
                code, command = self._command(flags)
                self.assertEqual(code, 1); self.assertIsNone(command)


if __name__ == '__main__': unittest.main(verbosity=2)
