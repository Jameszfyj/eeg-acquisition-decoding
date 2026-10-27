import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from handbci.config import load_config
from handbci.prompted_control import PromptedControl
from handbci.studio_v2_runtime import StudioV2Decoder,PROFILE
from handbci.four_class_online import CHANNELS,SCALP

ROOT=Path(__file__).resolve().parents[1]


class FakeSource:
    channels=[CHANNELS[i] for i in SCALP]
    calibration={'kind':'linear-latent-readout'}

    def predict_logits(self,values):
        return np.tile(np.arange(6,dtype=np.float32),(len(values),1))


class StudioV2RuntimeTests(unittest.TestCase):
    def config(self):
        cfg=load_config(ROOT/'configs/collection_ui_neuracle_ch32.json')
        cfg['preprocess']=dict(output_rate=250.,lowpass_hz=100.,highpass_hz=4.,numtaps=201,trim_seconds=1.)
        cfg['window'].update(seconds=2.,stride_seconds=.5)
        return cfg

    def test_uncalibrated_source_is_rejected(self):
        source=FakeSource();source.calibration=None
        with patch('handbci.studio_v2_runtime.load_source',return_value=(source,None)):
            with self.assertRaisesRegex(ValueError,'尚未专用校准'):
                StudioV2Decoder('unused.pt',self.config())

    def test_prepare_grid_live_windows_use_only_received_data(self):
        with patch('handbci.studio_v2_runtime.load_source',return_value=(FakeSource(),None)):
            decoder=StudioV2Decoder('unused.pt',self.config())
        result=None
        decoder.set_prompt_anchor(None)
        for start in range(0,3000,100):
            self.assertIsNone(decoder.process(np.ones((32,100)),np.arange(start,start+100)))
        decoder.set_prompt_anchor(3501)
        for start in range(3000,6100,100):
            value=decoder.process(np.ones((32,100)),np.arange(start,start+100))
            if value is not None:
                result=value
                self.assertLessEqual(decoder.last_window['source_end_index'],start+99)
                self.assertEqual((decoder.last_window['model_start_index']-3500)%500,0)
                self.assertGreaterEqual(decoder.last_window['source_start_index'],2500)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[1].sum(),1.)
        self.assertEqual(decoder.last_window['prepare_source_index'],3500)
        decoder.set_prompt_anchor(7001)
        for start in range(6100,8900,100):
            self.assertIsNone(decoder.process(np.ones((32,100)),np.arange(start,start+100)))
        result=decoder.process(np.ones((32,200)),np.arange(8900,9100))
        self.assertIsNotNone(result)
        self.assertEqual(decoder.last_window['model_start_index'],7000)

    def test_generic_gate_stays_after_cue_and_v2_gate_honours_prepare(self):
        control=object.__new__(PromptedControl)
        control.phase='decoding';control.fs=1000;control.cutoff=6000
        control.cue={'echo_verified':True,'source_cutoff':6000}
        control.prepare_mark={'estimated_source_index':3501}
        window=dict(profile=PROFILE,anchor='prepare',prepare_source_index=3500,
                    model_start_index=3500,source_start_index=2500)
        self.assertTrue(control.eligible(window))
        changed=dict(window,prepare_source_index=8000)
        self.assertFalse(control.eligible(changed))
        self.assertFalse(control.eligible(dict(source_start_index=2500)))
        self.assertTrue(control.eligible(dict(source_start_index=6100)))
        control.phase='cue_hold'
        self.assertFalse(control.eligible(window))

    def test_free_stream_has_full_history_and_recovers_from_gaps(self):
        with patch('handbci.studio_v2_runtime.load_source',return_value=(FakeSource(),None)):
            decoder=StudioV2Decoder('unused.pt',self.config())
        result=decoder.process(np.ones((32,4000)),np.arange(4000))
        self.assertIsNotNone(result)
        self.assertEqual(decoder.last_window['anchor'],'continuous')
        self.assertEqual(decoder.last_window['filter_history_seconds'],1.)
        with self.assertRaises(ValueError):decoder.process(np.ones((32,10)),np.arange(4010,4020))
        decoder.recover_stream(5000,'gap')
        self.assertIsNone(decoder.process(np.ones((32,1000)),np.arange(5000,6000)))

    def test_configured_tenth_second_stride_is_used_in_both_modes(self):
        cfg=self.config();cfg['window']['stride_seconds']=.1
        for prompted in (False,True):
            with patch('handbci.studio_v2_runtime.load_source',return_value=(FakeSource(),None)):
                decoder=StudioV2Decoder('unused.pt',cfg)
            if prompted:decoder.set_prompt_anchor(1000)
            starts=[]
            for start in range(0,7000,20):
                value=decoder.process(np.ones((32,20)),np.arange(start,start+20))
                if value is not None:
                    starts.append(decoder.last_window['model_start_index'])
                    self.assertLessEqual(decoder.last_window['source_end_index'],start+19)
            self.assertGreater(len(starts),30)
            np.testing.assert_array_equal(np.diff(starts),100)


if __name__=='__main__':unittest.main()
