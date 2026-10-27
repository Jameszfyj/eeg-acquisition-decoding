import tempfile
from pathlib import Path
import unittest

import numpy as np
from onnx import TensorProto,helper,numpy_helper

from examples.create_demo_model import bundle_bytes
from handbci.config import load_config
from handbci.continuous_online import ContinuousController
from handbci.four_class_online import CHANNELS,SCALP
from handbci.model_bundle import ONNXDecoder,import_bundle
from handbci.shared_calibration import ImportedCalibrationDecoder
from handbci.subject_online import prepare_subject_model

ROOT=Path(__file__).resolve().parents[1]


def fixture(calibrated=False):
    cfg=load_config(ROOT/'configs/collection_ui_neuracle_ch32.json')
    cfg['device'].update(source='neuracle_acp',unit='uV',sample_rate=1000,stream_channels=33,host='127.0.0.1',port=4097)
    names=['左旋臂','左握拳','右握拳','右旋臂','双脚','舌头']
    cfg['classes']=[dict(id=i,name=name,task_text=name,action='custom') for i,name in enumerate(names)]
    cfg['preprocess']=dict(output_rate=250.,lowpass_hz=100.,highpass_hz=4.,numtaps=201,trim_seconds=1.)
    cfg['window'].update(seconds=2.,stride_seconds=.5,anchor='prepare')
    cfg['session_metadata']=dict(participant_id='P01')
    manifest=dict(format='handbci.onnx.v1',participant_id='P01',name='native test',config=cfg,
        input=dict(name='eeg',pipeline='ifnet_filterbank',channels=[CHANNELS[i] for i in SCALP]),
        output=dict(name='scores',kind='logits'),studio=dict(arm='B_platform_native',
            profile='l100d250h4_pre_v2',profile_fingerprint='ac09b1dccf7810e4'))
    nodes=[helper.make_node('ReduceMean',['eeg'],['bands'],axes=[2,3],keepdims=0),
           helper.make_node('MatMul',['bands','weights'],['prebias' if calibrated else 'scores'])]
    initializers=[numpy_helper.from_array(np.zeros((2,6),np.float32),'weights')]
    if calibrated:
        bias=np.linspace(-.5,.5,6,dtype=np.float32)
        initializers.append(numpy_helper.from_array(bias,'calib_bias'))
        nodes.append(helper.make_node('Add',['prebias','calib_bias'],['scores']))
        cfg['session_metadata']['calibration']=dict(method='handbci.shared_calibration.fit_bias',
            bias=bias.tolist(),trials=240,sessions=['session1','session2'],protocol=dict(rounds=20,trials=120))
    graph=helper.make_graph(nodes,'native fixture',
        [helper.make_tensor_value_info('eeg',TensorProto.FLOAT,[1,2,30,500])],
        [helper.make_tensor_value_info('scores',TensorProto.FLOAT,[1,6])],initializers)
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)]);model.ir_version=8
    return manifest,model


class NativeONNXTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.registry=self.root/'registry.json'
        self.controller=ContinuousController(self.root/'runs',registry=self.registry)

    def tearDown(self):
        self.tmp.cleanup()

    def add(self,calibrated=False):
        manifest,model=fixture(calibrated)
        entry=import_bundle(bundle_bytes(manifest,model),'P01',registry=self.registry,import_root=self.root/'models')
        selection,cfg=prepare_subject_model('P01',entry['model_id'],registry=self.registry)
        return entry,selection,cfg,ONNXDecoder(selection['model'])

    def test_embedded_bias_is_applied_once_and_does_not_request_calibration(self):
        entry,selection,cfg,decoder=self.add(True)
        self.assertTrue(selection['embedded_calibration_ready'])
        bias=np.asarray(cfg['session_metadata']['calibration']['bias'])
        np.testing.assert_allclose(decoder.predict(np.zeros((1,2,30,500),np.float32)),np.exp(bias)/np.exp(bias).sum())
        request=dict(participant='P01',model_id=entry['model_id'],mode='acp',host='127.0.0.1',port=4097,
                     prompted=dict(enabled=True))
        prepared=self.controller.prepare(request)[0]
        self.assertNotIn('imported_calibration',prepared)
        details=self.controller.model_details('P01',entry['model_id'])
        self.assertFalse(details['requires_prompted_calibration'])
        request.update(calibration_mode='reuse',calibration_source='old')
        with self.assertRaisesRegex(ValueError,'已内置校准'):
            self.controller.prepare(request)

    def test_bad_declared_bias_and_unknown_prepare_contract_are_rejected(self):
        manifest,model=fixture(True)
        manifest['config']['session_metadata']['calibration']['bias'][0]+=1
        with self.assertRaisesRegex(ValueError,'实际加法'):
            import_bundle(bundle_bytes(manifest,model),'P01',registry=self.registry,import_root=self.root/'models')
        manifest,model=fixture();manifest['config']['window']['stride_seconds']=.1
        with self.assertRaisesRegex(ValueError,'准备锚点'):
            import_bundle(bundle_bytes(manifest,model),'P01',registry=self.registry,import_root=self.root/'models')

    def test_prepare_grid_is_causal_and_changes_with_new_prepare_marker(self):
        _,_,_,decoder=self.add()
        decoder.set_prompt_anchor(None)
        rng=np.random.default_rng(12)
        raw=rng.normal(size=(32,9300)).astype(np.float32)
        for start in range(0,3000,100):
            self.assertIsNone(decoder.process(raw[:,start:start+100],np.arange(start,start+100)))
        decoder.set_prompt_anchor(3501)
        count=0
        for start in range(3000,6100,100):
            prediction=decoder.process(raw[:,start:start+100],np.arange(start,start+100))
            if prediction is not None:
                count+=1;window=decoder.last_window
                self.assertEqual((window['model_start_index']-3500)%500,0)
                self.assertLessEqual(window['source_end_index'],start+99)
                self.assertGreaterEqual(window['source_start_index'],2500)
        self.assertGreater(count,0)
        decoder.set_prompt_anchor(7001)
        for start in range(6100,8900,100):
            self.assertIsNone(decoder.process(raw[:,start:start+100],np.arange(start,start+100)))
        self.assertIsNotNone(decoder.process(raw[:,8900:9200],np.arange(8900,9200)))
        self.assertEqual(decoder.last_window['model_start_index'],7000)

    def test_unfitted_native_model_retains_twenty_round_prompted_calibration(self):
        entry,selection,cfg,decoder=self.add()
        request=dict(participant='P01',model_id=entry['model_id'],mode='acp',host='127.0.0.1',port=4097,
                     prompted=dict(enabled=True))
        prepared=self.controller.prepare(request)[0]
        self.assertTrue(prepared['imported_calibration'])
        self.assertEqual(prepared['calibration_trials'],120)
        wrapped=ImportedCalibrationDecoder(decoder,cfg,selection)
        wrapped.set_prompt_anchor(3500)
        self.assertFalse(decoder.prompted)
        wrapped.bias=np.zeros(6)
        wrapped.set_prompt_anchor(3500)
        self.assertEqual(decoder.anchor,3500)
        decoder.recover_stream(4000,'replay')
        self.assertFalse(decoder.prompted)
