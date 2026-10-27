"""Complete ONNX export tests with artificial weights, signals and calibration."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import onnx
from scipy import signal
import torch

from tests.test_retrained_runtime import fixture
from handbci.config import validate_config
from handbci.four_class_online import digest
from handbci.model_bundle import ONNXDecoder,import_bundle
from handbci.retrained_runtime import load_portable,RetrainedStreamDecoder
from handbci.retrained_onnx import CalibratedRouter,export_bundle
from handbci.subject_online import prepare_subject_model
from handbci.continuous_online import ContinuousController


class RetrainedONNXTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        payload,cls.cfg=fixture();payload['training']['n_tasks']=909
        cls.weights=cls.root/'model.pt';torch.save(payload,cls.weights)
        cls.calibration=cls.root/'calibration.npz'
        metadata=dict(method='ea_only',model_sha256=digest(cls.weights),rounds=20,n_tasks=120,
            source_provenance=dict(source_session_id='synthetic-test-session'))
        np.savez_compressed(cls.calibration,metadata=np.asarray(json.dumps(metadata)),alignment=np.eye(30))
        cls.archive=export_bundle(cls.weights,cls.calibration,cls.root/'onnx')

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_import_restores_calibration_and_matches_original_stream(self):
        registry=self.root/'registry.json'
        entry=import_bundle(self.archive.read_bytes(),'P01',registry=registry,import_root=self.root/'imported')
        selection,cfg=prepare_subject_model('P01',entry['model_id'],registry=registry)
        self.assertTrue(selection['embedded_calibration_ready'])
        self.assertFalse(selection['requires_calibration'])
        decoder=ONNXDecoder(selection['model'])
        self.assertEqual(decoder.n_time,700)
        oracle=RetrainedStreamDecoder(self.weights,self.cfg)
        oracle.predictor.restore_calibration(self.calibration)
        raw=np.random.default_rng(3).normal(size=(32,5000)).astype(np.float32)
        for start in range(0,raw.shape[1],100):
            ix=np.arange(start,start+100)
            actual=decoder.process(raw[:,start:start+100],ix)
            expected=oracle.process(raw[:,start:start+100],ix)
            self.assertEqual(actual is None,expected is None)
            if actual is not None:
                np.testing.assert_allclose(actual[1],expected[1],atol=2e-5,rtol=2e-4)
                self.assertEqual(int(actual[1].argmax()),int(expected[1].argmax()))
                self.assertEqual(decoder.last_window,oracle.last_window)
        controller=ContinuousController(self.root/'runs',registry=registry)
        details=controller.model_details('P01',entry['model_id'])
        self.assertFalse(details['requires_prompted_calibration'])
        self.assertEqual([c['hand_action'] for c in details['classes']],['index','ring','thumb','little','pinch','middle'])
        prepared=controller.prepare(dict(participant='P01',model_id=entry['model_id'],mode='acp',
            host='127.0.0.1',port=4097,prompted=dict(enabled=True),output=dict(mode='simulation')))[0]
        self.assertNotIn('imported_calibration',prepared)
        with self.assertRaisesRegex(ValueError,'IFNet窗口'):
            validate_config(cfg)  # Native single-IFNet patch rule remains strict.

    def test_graph_metadata_and_alignment_are_checked(self):
        with zipfile.ZipFile(self.archive) as archive:
            manifest=json.loads(archive.read('manifest.json'));data=archive.read('model.onnx')
        cases=[]
        changed=copy.deepcopy(manifest)
        changed['deployment']['calibration']['alignment_sha256']='0'*64
        cases.append((changed,data))
        graph=onnx.load_model_from_string(data);del graph.metadata_props[:]
        cases.append((manifest,graph.SerializeToString()))
        graph=onnx.load_model_from_string(data)
        for tensor in graph.graph.initializer:
            if tensor.name=='alignment':
                value=onnx.numpy_helper.to_array(tensor).copy();value[0,0]+=1
                tensor.CopyFrom(onnx.numpy_helper.from_array(value,'alignment'))
        cases.append((manifest,graph.SerializeToString()))
        for index,(config,model) in enumerate(cases):
            output=io.BytesIO()
            with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('manifest.json',json.dumps(config));archive.writestr('model.onnx',model)
            with self.assertRaisesRegex(ValueError,'校准|EA'):
                import_bundle(output.getvalue(),'P01',registry=self.root/'bad-registry.json',import_root=self.root/('bad-'+str(index)))
        self.assertFalse((self.root/'bad-registry.json').exists())

    def test_residual_head_formula_is_preserved(self):
        predictor,_=load_portable(self.weights);predictor.restore_calibration(self.calibration)
        dimension=predictor.models['expert'].classifier[0].in_features
        rng=np.random.default_rng(99)
        predictor.state=dict(mean=np.zeros(dimension),scale=np.ones(dimension),
            delta=rng.normal(size=dimension+1)*.01)
        predictor.calibration['method']='head_ridge1'
        network=CalibratedRouter(predictor).eval()
        x=rng.normal(size=(1,30,700)).astype(np.float32)
        windows=np.stack((x[:,:,:500],x[:,:,100:600],x[:,:,200:700]),axis=1)
        windows=signal.filtfilt(*signal.iirnotch(50.,30.,fs=250.),windows,axis=-1).astype(np.float32)
        expected=predictor.predict(windows)['probabilities']
        with torch.inference_mode():actual=network(torch.from_numpy(x)).numpy()
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-4)
