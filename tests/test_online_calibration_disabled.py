import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from handbci.continuous_online import ContinuousController, DEMO_ID
from handbci.six_class_improved import ImprovedSixClassDecoder
from handbci.six_class_right_review import RightHandSixClassDecoder
from handbci.retrained_runtime import RetrainedStreamDecoder
from handbci.studio_v2_runtime import StudioV2Decoder
from tests import test_calibration_protocol as protocol_tests
from tests.test_calibration_protocol import UniformModel
from tests.test_retrained_runtime import fixture
from tests import test_studio_v2_runtime as v2_tests
from tests.test_studio_v2_runtime import FakeSource


class OnlineCalibrationDisabledTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_start_entry_points_reject_calibration_before_opening_devices(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=ContinuousController(Path(tmp))
            request=dict(model_id=DEMO_ID,mode='demo',session_purpose='calibration')
            for start in (c.prepare,c.start,c.start_decoding,c.start_calibration):
                with self.assertRaisesRegex(ValueError,'暂时禁用'):
                    start(request)
                self.assertIsNone(c.worker)
            self.assertFalse(c.eeg_probe_busy)
            details=c.model_details('DEMO',DEMO_ID)
            self.assertTrue(details['online_calibration_disabled'])
            self.assertFalse(details['requires_calibration'])

    def test_improved_and_right_review_models_never_collect_or_fit(self):
        for cls in (ImprovedSixClassDecoder, RightHandSixClassDecoder):
            cfg=protocol_tests.CalibrationProtocolTests().config()
            cfg['session_metadata'].update(online_calibration_disabled=True,right_hand_review=True,
                                          right_hand_review_config={'action':'wide'})
            payload={'_hand_supervised_calibration':True,'_right_hand_review_config':{'action':'wide'}}
            decoder=cls(payload,{'baseline':UniformModel(),'expert':UniformModel()},cfg)
            decoder._calibrate_step=Mock(side_effect=AssertionError('must not calibrate'))
            decoder.raw_calibration.append=Mock(side_effect=AssertionError('must not collect calibration'))
            result=decoder.process(np.ones((32,5000)),np.arange(5000))
            self.assertIsNotNone(result)
            self.assertAlmostEqual(float(result[1].sum()),1.,places=6)
            decoder.recover_stream(6000,'gap')
            self.assertIsNotNone(decoder.process(np.ones((32,5000)),np.arange(6000,11000)))
            self.assertIsNone(decoder.alignment)
            self.assertIsNone(decoder.hand_bias)
            self.assertEqual(decoder.completed,0)
            self.assertEqual(decoder.windows,[])
            self.assertTrue(decoder.snapshot_calibration()['disabled'])
            self.assertFalse(decoder.snapshot_calibration()['required'])
            decoder._calibrate_step.assert_not_called()
            decoder.raw_calibration.append.assert_not_called()
            with self.assertRaisesRegex(ValueError,'重新拟合'):
                decoder.restore_calibration({})
            if cls is RightHandSixClassDecoder:
                self.assertEqual(decoder.last_details['right_hand_review_skipped'],'online_calibration_disabled')

    def test_retrained_model_without_parameters_predicts_without_claiming_calibration(self):
        payload,cfg=fixture()
        cfg['session_metadata']['online_calibration_disabled']=True
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt';torch.save(payload,path)
            decoder=RetrainedStreamDecoder(path,cfg)
            with patch.object(decoder.predictor,'calibrate',side_effect=AssertionError('no fitting')):
                result=decoder.process(np.random.default_rng(22).normal(size=(32,5000)),np.arange(5000))
                self.assertIsNotNone(result)
                self.assertAlmostEqual(float(result[1].sum()),1.,places=6)
                self.assertIsNone(decoder.predictor.calibration)
                self.assertIsNone(decoder.predictor.alignment)
                self.assertIsNone(decoder.predictor.state)
                self.assertEqual(decoder.last_details['selected_head'],'uncalibrated_identity_alignment')
                with self.assertRaisesRegex(ValueError,'不允许拟合'):
                    decoder.fit_calibration(None,None,None)

    def test_v2_uses_original_head_if_missing_and_preserves_existing_head(self):
        for parameters in (None,{'kind':'linear-latent-readout'}):
            source=FakeSource();source.calibration=copy.deepcopy(parameters)
            cfg=v2_tests.StudioV2RuntimeTests().config()
            cfg.setdefault('session_metadata',{})['online_calibration_disabled']=True
            with patch('handbci.studio_v2_runtime.load_source',return_value=(source,None)):
                decoder=StudioV2Decoder('unused.pt',cfg)
            result=decoder.process(np.ones((32,4000)),np.arange(4000))
            self.assertIsNotNone(result)
            self.assertEqual(source.calibration,parameters)
            self.assertEqual(decoder.last_details['selected_head'],'calibrated' if parameters else 'original_source_head')


if __name__ == '__main__':
    unittest.main()
