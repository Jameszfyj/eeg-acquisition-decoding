import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from handbci.continuous_online import VotingGate
from handbci.model_bundle import ONNXDecoder
from tests import test_native_onnx as native


class PredictionStrideTests(unittest.TestCase):
    setUp = native.NativeONNXTests.setUp
    tearDown = native.NativeONNXTests.tearDown
    add = native.NativeONNXTests.add

    def test_default_stride_and_voting_do_not_require_local_settings(self):
        entry, _, _, _ = self.add(True)
        with patch('handbci.continuous_online.local_settings', return_value={}), \
             patch('handbci.workbench_settings.model_preferences', return_value={}):
            selected, effective, _, gate, _ = self.controller.prepare(dict(
                participant='P01', model_id=entry['model_id'], mode='acp',
                host='127.0.0.1', port=4097, prompted={'enabled': True}))
        self.assertEqual(selected['prediction_stride_seconds'], .1)
        self.assertEqual(effective['window']['stride_seconds'], .1)
        self.assertEqual(effective['session_metadata']['model_stride_seconds'], .5)
        self.assertEqual(gate.settings(), dict(mode='vote',vote_window_seconds=2.,vote_ratio=.65,
            decision_timeout_seconds=3.5,cooldown_seconds=4.,max_gap_seconds=.5))

    def test_local_stride_keeps_manifest_and_embedded_calibration(self):
        entry, selection, cfg, _ = self.add(True)
        manifest = Path(selection['model'])/'manifest.json'
        before = hashlib.sha256(manifest.read_bytes()).hexdigest()
        with patch('handbci.workbench_settings.model_preferences', return_value={'prediction_stride_seconds': .1}):
            selected, effective, _, _, _ = self.controller.prepare(dict(
                participant='P01', model_id=entry['model_id'], mode='acp',
                host='127.0.0.1', port=4097, prompted={'enabled': True}))
        self.assertEqual(effective['window']['stride_seconds'], .1)
        self.assertEqual(effective['session_metadata']['manifest_stride_seconds'], .5)
        self.assertTrue(selected['embedded_calibration_ready'])
        self.assertEqual(before, hashlib.sha256(manifest.read_bytes()).hexdigest())

    def test_ten_hz_windows_remain_causal_and_trigger_unchanged_voting_gate(self):
        _, selection, _, _ = self.add(True)
        fast = ONNXDecoder(selection['model'], prediction_stride_seconds=.1)
        original = ONNXDecoder(selection['model'])
        for decoder in (fast, original):
            decoder.set_prompt_anchor(1000)
        raw = np.random.default_rng(3).normal(size=(32,8000)).astype(np.float32)
        windows = [[], []]
        gate = VotingGate(6)
        decisions = []
        for start in range(0,8000,40):
            for index, decoder in enumerate((fast, original)):
                result = decoder.process(raw[:,start:start+40], np.arange(start,start+40))
                if result is None:
                    continue
                stamp, probabilities = result
                window = decoder.last_window
                windows[index].append((window['model_start_index'], probabilities.copy()))
                self.assertLessEqual(window['source_end_index'], start+39)
                if index == 0:
                    event = gate.update(probabilities, stamp)
                    if event:
                        decisions.append(event)
        np.testing.assert_array_equal(np.diff([w[0] for w in windows[0]]), 100)
        np.testing.assert_array_equal(np.diff([w[0] for w in windows[1]]), 500)
        shared = dict(windows[0])
        for offset, probability in windows[1]:
            np.testing.assert_allclose(shared[offset], probability, rtol=1e-6, atol=1e-7)
        self.assertTrue(decisions)
        self.assertLessEqual(decisions[0]['decision_wait_seconds'], 2.2)

    def test_invalid_runtime_stride_is_rejected(self):
        _, selection, _, _ = self.add(True)
        for value in (True, 0, -.1, float('nan'), .001, .101, 61):
            with self.assertRaisesRegex(ValueError, '预测步长'):
                ONNXDecoder(selection['model'], prediction_stride_seconds=value)


if __name__ == '__main__':
    unittest.main()
