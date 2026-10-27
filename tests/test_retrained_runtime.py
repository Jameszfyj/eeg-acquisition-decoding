"""No personal recordings/weights or hardware required for the new PT runtime."""
import base64
import copy
import io
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np
from scipy import signal
import torch

from handbci.config import load_config
from handbci.model import IFNetClassifier
from handbci.four_class_online import CHANNELS,SCALP,digest
from handbci.six_class_online import ARCH,NAMES
from handbci.six_class_retrained import FORMAT
from handbci.retrained_runtime import (RetrainedStreamDecoder,task_windows,CALIBRATION_METHOD,
                                      register_model,decoder_from_selection)
from handbci.continuous_online import ContinuousController
from handbci.shared_calibration import ImportedCalibrationDecoder
from handbci.subject_online import prepare_subject_model
from handbci.preprocess import StreamPreprocessor

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    cfg=load_config(ROOT/'configs/demo_6class.json')
    cfg['device']=dict(source='neuracle_acp',host='127.0.0.1',port=4097,sample_rate=1000.,
        stream_channels=33,channels=CHANNELS,unit='uV')
    cfg['preprocess']=dict(output_rate=250.,lowpass_hz=100.,highpass_hz=4.,numtaps=201,trim_seconds=1.)
    cfg['window']=dict(seconds=2.,stride_seconds=.1,task_offset_seconds=.6)
    cfg['classes']=[dict(id=i,name=n,task_text=n,action='custom') for i,n in enumerate(NAMES)]
    cfg['session_metadata']=dict(participant_id='P01');cfg['paradigm_id']='synthetic-unit-test-contract'
    torch.manual_seed(31)
    state=IFNetClassifier(**ARCH).state_dict()
    return dict(format=FORMAT,participant_id='P01',config=cfg,class_names=NAMES,
        channel_names=[CHANNELS[i] for i in SCALP],baseline=dict(architecture=ARCH,state_dict=state),
        expert=dict(architecture=ARCH,state_dict=copy.deepcopy(state)),
        future_calibration=dict(rounds=20,tasks=120,labels_required=True),
        training=dict(source_sha256={'trained_source':'a'*64}),online_control_validated=False),cfg


class RetrainedRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_native_import_check_and_calibration_are_model_specific(self):
        payload,cfg=fixture();buffer=io.BytesIO();torch.save(payload,buffer)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            controller=ContinuousController(root/'sessions',registry=root/'registry.json',import_root=root/'imports')
            entry=controller.import_model(dict(participant='P01',data=base64.b64encode(buffer.getvalue()).decode()))
            self.assertEqual(entry['runtime'],'retrained_six')
            selection,cfg=prepare_subject_model('P01',entry['model_id'],registry=root/'registry.json')
            request=dict(participant='P01',model_id=entry['model_id'],mode='acp',host='127.0.0.1',port=4097,
                         calibration_mode='new',output=dict(mode='simulation'))
            checked=controller.check(request)
            self.assertFalse(checked['connected'])
            self.assertEqual(checked['selection']['calibration_method'],CALIBRATION_METHOD)
            self.assertTrue(checked['selection']['requires_calibration'])
            base=RetrainedStreamDecoder(selection['model'],cfg,selection['sha256'])
            wrapper=ImportedCalibrationDecoder(base,cfg,selection)
            rng=np.random.default_rng(44);raw=rng.normal(size=(120,32,4000)).astype(np.float32)
            saved=dict(raw=raw,labels=np.tile(np.arange(6),20),starts=np.arange(120)*5000,
                       summary=dict(id='unit-test',source_sha256={'new_source':'b'*64}))
            wrapper.restore_calibration(saved)
            self.assertTrue(wrapper.snapshot_calibration()['ready'])
            np.testing.assert_array_equal(wrapper.bias,np.zeros(6))
            fresh=rng.normal(size=(32,5000)).astype(np.float32)
            result=None
            for start in range(0,5000,100):
                q=wrapper.process(fresh[:,start:start+100],np.arange(start,start+100))
                if q is not None:result=q
            self.assertIsNotNone(result)
            self.assertAlmostEqual(float(result[1].sum()),1.,places=6)
            self.assertEqual(wrapper.last_details['calibration_method'],CALIBRATION_METHOD)
            self.assertEqual(wrapper.last_window['input_seconds'],2.8)
            output=root/'saved';output.mkdir();wrapper.save_calibration(output)
            self.assertTrue((output/'model_calibration.npz').is_file())
            with self.assertRaises(ValueError):
                base.fit_calibration(raw,saved['labels'],saved['starts'],dict(source_sha256={'data.h5':'a'*64}))
            duplicate=controller.import_model(dict(participant='P01',data=base64.b64encode(buffer.getvalue()).decode()))
            self.assertEqual(duplicate['model_id'],entry['model_id'])

    def test_task_only_filtering_matches_continuous_and_handles_sample_phase(self):
        _,cfg=fixture();rng=np.random.default_rng(15)
        raw=rng.normal(size=(32,9000)).astype(np.float32);start=2003
        local=task_windows(raw[None,:,start:start+3998],np.array([start]),cfg)[0]
        stream=StreamPreprocessor(cfg);values,indices=stream.process(raw,np.arange(9000))
        logical=indices-500;expected=[]
        for offset in (.6,1.,1.4):
            j=np.searchsorted(logical,start+1000*offset)
            expected.append(values[SCALP,j:j+500])
        expected=signal.filtfilt(*signal.iirnotch(50.,30.,fs=250.),np.stack(expected),axis=-1).astype(np.float32)
        np.testing.assert_allclose(local,expected,rtol=1e-6,atol=1e-6)

    def test_rejects_owner_and_preprocessing_mismatch(self):
        payload,cfg=fixture();payload['config']['preprocess']['highpass_hz']=1.
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'model.pt';torch.save(payload,path)
            with self.assertRaises(ValueError):RetrainedStreamDecoder(path,cfg)

    def test_saved_parameters_restore_without_raw_data_or_refitting(self):
        payload,cfg=fixture()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'model.pt';calibration=root/'calibration.npz';registry=root/'registry.json'
            torch.save(payload,path)
            metadata=dict(method='ea_only',model_sha256=digest(path),rounds=20,n_tasks=120)
            np.savez_compressed(calibration,metadata=np.asarray(json.dumps(metadata)),alignment=np.eye(30))
            original=register_model('P01',path,registry)
            ready=register_model('P01',path,registry,calibration_path=calibration)
            self.assertNotEqual(original['model_id'],ready['model_id'])
            selection,cfg=prepare_subject_model('P01',ready['model_id'],registry=registry)
            self.assertTrue(selection['saved_calibration_ready'])
            self.assertFalse(selection['requires_calibration'])
            self.assertEqual([c['hand_action'] for c in cfg['classes']],['index','ring','thumb','little','pinch','middle'])
            controller=ContinuousController(root/'runs',registry=registry)
            request=dict(participant='P01',model_id=ready['model_id'],mode='acp',host='127.0.0.1',port=4097,
                         prompted=dict(enabled=True),output=dict(mode='simulation'))
            prepared=controller.prepare(request)[0]
            self.assertNotIn('imported_calibration',prepared)
            self.assertFalse(controller.model_details('P01',ready['model_id'])['requires_prompted_calibration'])
            decoder=decoder_from_selection(selection,cfg)
            raw=np.random.default_rng(3).normal(size=(32,5000)).astype(np.float32)
            result=decoder.process(raw,np.arange(5000))
            self.assertIsNotNone(result)
            self.assertAlmostEqual(float(result[1].sum()),1.,places=6)
            self.assertEqual(decoder.last_details['selected_head'],'ea_only')
            request['session_purpose']='calibration'
            with self.assertRaisesRegex(ValueError,'已恢复20轮'):
                controller.prepare(request)
            calibration.write_bytes(calibration.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'发生变化'):
                prepare_subject_model('P01',ready['model_id'],registry=registry)
            with self.assertRaisesRegex(ValueError,'校验失败'):
                decoder_from_selection(selection,cfg)
            uncalibrated,_=prepare_subject_model('P01',original['model_id'],registry=registry)
            self.assertTrue(uncalibrated['requires_calibration'])

    def test_mismatched_saved_parameters_do_not_register_model(self):
        payload,_=fixture()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'model.pt';calibration=root/'calibration.npz';registry=root/'registry.json'
            torch.save(payload,path)
            np.savez_compressed(calibration,metadata=np.asarray(json.dumps(dict(method='ea_only',n_tasks=120,
                model_sha256='0'*64))),alignment=np.eye(30))
            with self.assertRaisesRegex(ValueError,'不属于'):
                register_model('P01',path,registry,calibration_path=calibration)
            self.assertFalse(registry.exists())


if __name__=='__main__':unittest.main()
