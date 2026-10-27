"""Variable calibration must drive timing, fitting, persistence and reuse alike."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from handbci.calibration_store import CalibrationStore
from handbci.config import load_config
from handbci.labelled_hand_calibration import validate_labels
from handbci.six_class_improved import ImprovedSixClassDecoder, fit_alignment
from handbci.workbench_settings import calibration_protocol

ROOT = Path(__file__).resolve().parents[1]


class UniformModel(torch.nn.Module):
    def forward(self, values):
        return torch.zeros((len(values), 6))


class CalibrationProtocolTests(unittest.TestCase):
    def config(self):
        cfg = load_config(ROOT/'configs/collection_ui_neuracle_ch32.json')
        cfg['device'].update(sample_rate=1000, unit='uV')
        cfg['preprocess'] = dict(output_rate=250., lowpass_hz=100., highpass_hz=4., numtaps=201, trim_seconds=1.)
        cfg['window']['seconds'] = 2.
        cfg.setdefault('session_metadata', {}).update(hand_supervised_calibration=True,
            brain_calibration=dict(rounds=20, imagery_seconds=6., rest_seconds=2.))
        return cfg

    def decoder(self, cfg):
        return ImprovedSixClassDecoder({'_hand_supervised_calibration': True},
            {'baseline': UniformModel()}, cfg)

    def test_default_and_invalid_protocol(self):
        self.assertEqual(calibration_protocol({})['trials'], 18)
        for value in ({'rounds': True}, {'rounds': 0}, {'rounds': 101},
                      {'imagery_seconds': 3.5}, {'imagery_seconds': float('nan')}):
            with self.assertRaises(ValueError):
                calibration_protocol({'session_metadata': {'brain_calibration': value}})
        with self.assertRaises(ValueError):
            fit_alignment(np.zeros((54, 30, 500)), n_trials=120)

    def test_twenty_rounds_six_seconds_fit_save_restore_and_contract(self):
        cfg = self.config()
        decoder = self.decoder(cfg)
        self.assertEqual(np.bincount(decoder.tasks).tolist(), [20]*6)
        decoder.latest_index = 2999
        decoder._prompt('rest')
        decoder.acknowledge(decoder.prompt_id, decoder.session_id)
        decoder.latest_index = 4998
        decoder._calibrate_step()
        self.assertEqual(decoder.phase, 'rest')
        decoder.latest_index = 4999
        decoder._calibrate_step()
        self.assertEqual(decoder.phase, 'imagery')
        self.assertEqual(decoder.remaining, 6.)
        np.testing.assert_array_equal(validate_labels(decoder.tasks, rounds=20), decoder.tasks)
        with self.assertRaises(ValueError):
            validate_labels(decoder.tasks)
        rng = np.random.default_rng(17)
        windows = rng.normal(size=(360, 30, 500)).astype(np.float32)
        # Preserve original three model windows, but keep imagery active for all six seconds.
        for trial in range(120):
            decoder.latest_index = trial*10000 + 3000
            decoder._prompt('imagery')
            decoder.acknowledge(decoder.prompt_id, decoder.session_id)
            start = decoder.started_index
            decoder._trial_windows = lambda trial=trial: windows[trial*3:trial*3+3]
            decoder.latest_index = start+5998
            decoder._calibrate_step()
            self.assertEqual(decoder.completed, trial)
            self.assertEqual(decoder.phase, 'imagery')
            decoder.latest_index = start+5999
            decoder._calibrate_step()
            self.assertEqual(decoder.completed, trial+1)
        state = decoder.snapshot_calibration()
        self.assertTrue(state['ready'])
        self.assertEqual((state['total'], state['round'], state['imagery_seconds']), (120, 20, 6.))
        self.assertEqual(decoder.calibration_info['n_overlapping_windows'], 360)
        np.testing.assert_allclose(decoder.hand_bias, 0., atol=1e-12)
        with tempfile.TemporaryDirectory() as root:
            session_id = '20260101_120000_1234abcd_S01'
            session = Path(root)/session_id
            session.mkdir()
            decoder.save_calibration(session)
            selection = dict(participant_id='S01', model_id='test-model', sha256='test-digest',
                             hand_supervised_calibration=True)
            (session/'model_selection.json').write_text(json.dumps(dict(selection, config=cfg)), encoding='utf-8')
            device = dict(info=dict(type='test', serial='example', cap='test-cap', labels_verified=True),
                          labels=cfg['device']['channels'], settings=dict(fs=1000), unit='uV')
            (session/'device.json').write_text(json.dumps(device), encoding='utf-8')
            saved = CalibrationStore(root).load(session_id, selection, cfg)
            self.assertEqual(saved['summary']['calibration_trials'], 120)
            restored = self.decoder(cfg)
            restored.restore_calibration(saved)
            self.assertTrue(restored.snapshot_calibration()['ready'])
            self.assertEqual(restored.completed, 120)
            old_cfg = copy.deepcopy(cfg)
            old_cfg['session_metadata'].pop('brain_calibration')
            with self.assertRaises(ValueError):
                CalibrationStore(root).load(session_id, selection, old_cfg)

    def test_six_second_task_keeps_source_support_within_imagery(self):
        decoder = self.decoder(self.config())
        decoder.started_index = 10000
        decoder.indices = np.arange(8000, 16000, 4)
        decoder.history = np.random.default_rng(4).normal(size=(30, 2000)).astype(np.float32)
        self.assertEqual(decoder._trial_windows().shape, (3, 30, 500))


if __name__ == '__main__':
    unittest.main()
