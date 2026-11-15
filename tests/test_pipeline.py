"""End-to-end synthetic fixtures. None of these scores are participant results."""
import contextlib
from datetime import datetime, timezone
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


runner = load('repo_pipeline_runner_tests', ROOT/'binary/run_rp_binary.py')
live = runner.module('live')
audit = load('repo_pipeline_audit_tests', ROOT/'tools/diagnose_rp_signal.py')


def raw_fixture(path, date, classes):
    cursor = 600; markers = []; actions = []; fs = 250.
    for run in range(1,5):
        markers.append((cursor,100+run)); cursor += 600
        for marker in list(classes)*6:
            actions.append((cursor,marker)); markers += [(cursor,marker),(cursor+1000,20)]
            cursor += 1750
        markers.append((cursor,110+run)); cursor += 600
    n = cursor+300; rng = np.random.default_rng(918)
    values = rng.normal(0,.2,(n,4)); time = np.arange(1000)/fs
    patterns = {11:[25,16,5,4],12:[6,20,19,4],13:[4,5,16,25]}
    for row, marker in actions:
        values[row:row+1000] += np.column_stack([amplitude*np.sin(2*np.pi*frequency*time)
            for amplitude,frequency in zip(patterns[marker],[29,39,69,89])])
    data = {'Sample Index':np.arange(n)%256}
    for channel in range(8): data['EXG Channel %d'%channel] = values[:,channel] if channel<4 else np.zeros(n)
    data['Timestamp'] = datetime.fromisoformat(date).replace(tzinfo=timezone.utc).timestamp()+np.arange(n)/fs
    data['Marker Channel'] = np.zeros(n)
    for row, marker in markers: data['Marker Channel'][row] = marker
    with path.open('w',encoding='utf-8') as stream:
        stream.write('%OpenBCI Raw EXG Data\n%Number of channels = 8\n%Sample Rate = 250 Hz\n%Board = Synthetic test only\n')
        pd.DataFrame(data).to_csv(stream,index=False)


class Info:
    def name(self): return 'rps_raw'
    def type(self): return 'EMG'
    def nominal_srate(self): return 250.
    def channel_count(self): return 8
    def source_id(self): return 'synthetic_test_only'
    def uid(self): return 'synthetic_test_only_uid'


class FakeLSL:
    proc_clocksync = 1
    __version__ = 'synthetic-test-only'
    def __init__(self, values, stop): self.values,self.stop,self.offset,self.clock = values,stop,0,1000.
    def resolve_streams(self,wait_time): return [Info()]
    def local_clock(self): return self.clock
    def StreamInlet(self,selected,**kwargs): return self
    def open_stream(self,timeout): pass
    def close_stream(self): pass
    def pull_chunk(self,timeout,max_samples):
        if self.offset >= len(self.values): self.stop.set(); return [],[]
        stop = min(len(self.values), self.offset+min(max_samples,50))
        stamps = 1000.+np.arange(self.offset,stop)/250.
        samples = np.column_stack([self.values[self.offset:stop],np.zeros((stop-self.offset,4))])
        self.offset = stop; self.clock = float(stamps[-1]+.01)
        return samples.tolist(),stamps.tolist()


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name)
        cls.history = cls.root/'OpenBCI-RAW-2026-01-01_12-00-00.txt'
        cls.today = cls.root/'OpenBCI-RAW-2026-01-02_12-00-00.txt'
        raw_fixture(cls.history,'2026-01-01',(11,12,13)); raw_fixture(cls.today,'2026-01-02',(11,13))
        legacy = cls.root/'legacy_preprocess'
        result = subprocess.run([sys.executable,str(ROOT/'rps/01_preprocess_openbci_rps_4ch.py'),
            '--input',str(cls.history),'--subject','SYNTHETIC','--output',str(legacy)],capture_output=True,text=True)
        if result.returncode: raise AssertionError(result.stdout[-3000:]+result.stderr[-3000:])
        cls.legacy = next(path for path in legacy.iterdir() if (path/'rps_windows.npz').is_file())
        result = subprocess.run([sys.executable,str(ROOT/'run.py'),'init','--history-data',str(cls.legacy),
            '--today-input',str(cls.today),'--subject','SYNTHETIC','--output',str(cls.root/'init')],
            capture_output=True,text=True,timeout=180)
        if result.returncode: raise AssertionError(result.stdout[-5000:]+result.stderr[-5000:])
        cls.run_dir = next(path.parent for path in (cls.root/'init').rglob('pipeline_state.json'))
        cls.state = runner.read_json(cls.run_dir/'pipeline_state.json')
        cls.bundle,cls.model = live.load_daily(cls.state['daily_model'])
        cls.params = runner.read_json(cls.state['detector'])
        cls.data = live.load_features(cls.state['today_data'])
        cls.signal = live.load_signal(cls.data)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_legacy_three_class_data_becomes_two_classes_without_scissors_as_rest(self):
        history = runner.module('train').read_inputs(self.run_dir/'HISTORY_BINARY')
        self.assertEqual(len(history['trials']),48)
        self.assertEqual(set(history['meta'].marker_code),{11,13})
        np.testing.assert_array_equal(history['meta'].marker_code,np.take([11,13],history['meta'].y))
        self.assertEqual(len(pd.read_csv(self.run_dir/'HISTORY_BINARY/excluded_actions.csv')),24)

    def test_today_split_and_gate_protocol_are_frozen(self):
        self.assertEqual(len(self.data['trials']),48)
        self.assertEqual(self.bundle['calibration_selection_runs'],[1,2])
        self.assertEqual(self.bundle['independent_test_runs_not_used_for_fit'],[3,4])
        self.assertEqual(self.params['source_calibration_runs'],[1,2])
        self.assertIs(self.params['target_used_for_selection'],False)
        live.validate_detector(self.bundle,self.model,self.params)

    def test_replay_scores_only_run3_and4_and_binary_confusion_shape(self):
        report = runner.read_json(self.run_dir/'HOLDOUT_REPLAY/evaluation.json')
        self.assertEqual(report['evaluation_runs'],[3,4])
        self.assertEqual(report['event_metrics']['n_trials_retained'],24)
        self.assertEqual(np.asarray(report['event_metrics']['confusion_stone_paper_no_trigger']).shape,(2,3))
        self.assertFalse(report['detector_receives_truth_markers'])

    def test_actual_svc_prediction_semantics(self):
        x = self.data['X'][:100]
        predicted,scores = live.predict(self.bundle['pipeline'],x)
        np.testing.assert_array_equal(predicted,self.bundle['pipeline'].predict(x))
        self.assertEqual(scores.shape,(100,2))

    def test_causal_windows_are_chunk_invariant_and_stale_input_never_emits(self):
        traces,events = [],[]; values = self.signal['raw'][:12000]
        for size in [17,50,1000]:
            engine = live.LiveEngine(self.bundle,self.params)
            rows = []
            for start in range(0,len(values),size):
                stop = min(start+size,len(values))
                rows.extend(engine.process(values[start:stop],1000.+np.arange(start,stop)/250.))
            traces.append(pd.DataFrame(rows)); events.append([(event['decision_row'],event['class_id']) for event in engine.detector.events])
        for trace,output in zip(traces[1:],events[1:]):
            np.testing.assert_array_equal(trace[live.FEATURE_NAMES],traces[0][live.FEATURE_NAMES])
            self.assertEqual(trace.detector_state.tolist(),traces[0].detector_state.tolist())
            self.assertEqual(output,events[0])
        stale = live.LiveEngine(self.bundle,self.params,max_data_age=.75)
        stale.process(values,1000.+np.arange(len(values))/250.,emission_clock=lambda:100000.)
        self.assertFalse(stale.detector.events)

    def test_live_receiver_posthoc_isolation_and_independent_rms_audit(self):
        before = [live.file_sha256(self.model),live.file_sha256(Path(self.state['detector']))]
        outputs = []; traces = []
        for opposite in [False,True]:
            stop = threading.Event(); fake = FakeLSL(self.signal['raw'][:12000],stop)
            args = live.build_live_parser().parse_args(['--model',str(self.model),'--detector',self.state['detector'],'--no-ui'])
            labels = live.PosthocLabels(args.class_names,clock=fake.local_clock)
            shared = {'posthoc_labels':labels}
            def annotate(snapshot):
                for event in labels.pending():
                    labels.annotate(event['event_id'],1-event['class_id'] if opposite else event['class_id'])
            output = self.root/('live_opposite' if opposite else 'live_same'); output.mkdir()
            with patch.object(live.Pipeline,'fit',side_effect=AssertionError('Unexpected fitting')),contextlib.redirect_stdout(io.StringIO()):
                live.run_receiver(args,self.bundle,self.model,self.params,output,stop,annotate,shared,lsl=fake)
            summary = live.read_json(output/'session_summary.json')
            self.assertEqual(summary['status'],'completed',summary)
            self.assertGreater(summary['event_count'],0)
            self.assertTrue(summary['model_and_detector_files_unchanged'])
            outputs.append(pd.read_csv(output/'events.csv')[['decision_row','class_id','marker_code_for_display']])
            traces.append(pd.read_csv(output/'window_trace.csv')[['detector_state','predicted_class_id','activity_score']])
            report,windows,bands = audit.audit(output)
            self.assertEqual(report['rms_consistency']['mismatched_channel_windows'],0)
            self.assertEqual(report['legacy_activity_consistency']['mismatched_windows'],0)
        pd.testing.assert_frame_equal(outputs[0],outputs[1]); pd.testing.assert_frame_equal(traces[0],traces[1])
        self.assertEqual([live.file_sha256(self.model),live.file_sha256(Path(self.state['detector']))],before)

    def test_day_reuses_binary_base_and_same_day_is_rejected(self):
        tomorrow = self.root/'OpenBCI-RAW-2026-01-03_12-00-00.txt'
        raw_fixture(tomorrow,'2026-01-03',(11,13))
        result = subprocess.run([sys.executable,str(ROOT/'run.py'),'day','--base-model',self.state['base_model'],
            '--today-input',str(tomorrow),'--subject','SYNTHETIC','--output',str(self.root/'day')],
            capture_output=True,text=True,timeout=180)
        self.assertEqual(result.returncode,0,result.stdout[-3000:]+result.stderr[-3000:])
        state = runner.read_json(next((self.root/'day').rglob('pipeline_state.json')))
        self.assertEqual(state['base_model'],self.state['base_model'])
        result = subprocess.run([sys.executable,str(ROOT/'run.py'),'day','--base-model',self.state['base_model'],
            '--today-input',str(self.history),'--subject','SYNTHETIC','--output',str(self.root/'rejected')],
            capture_output=True,text=True,timeout=90)
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(list((self.root/'rejected').rglob('daily_model.joblib')))


if __name__ == '__main__': unittest.main(verbosity=2)
