from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from examples.create_demo_model import build_manifest, build_model, bundle_bytes
from handbci.config import write_json
from handbci.continuous_online import ContinuousController
from handbci.model_bundle import ONNXDecoder, import_bundle
from handbci.shared_calibration import (ImportedCalibrationDecoder, RawCalibrationData,
    SharedCalibrationStore, imported_protocol)
from handbci.subject_online import prepare_subject_model


class SharedCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.registry=self.root/'models.json'
        self.controller=ContinuousController(self.root/'sessions',registry=self.registry,import_root=self.root/'models')
        manifest=build_manifest()
        manifest['config']['device'].update(source='neuracle_acp',unit='uV')
        self.entry=import_bundle(bundle_bytes(manifest),'DEMO_ONNX',registry=self.registry,import_root=self.root/'models')
        self.selection,self.cfg=prepare_subject_model('DEMO_ONNX',self.entry['model_id'],registry=self.registry)
        self.protocol=imported_protocol(self.cfg)
        self.session_id='20260101_120000_1234abcd_DEMO_ONNX'
        self.session=self.root/'sessions'/self.session_id
        self.session.mkdir(parents=True)
        device=dict(info=dict(type='test',serial='example',cap='test-cap',labels_verified=True),
                    labels=self.cfg['device']['channels'],settings=dict(fs=1000),unit='uV')
        write_json(self.session/'device.json',device)

    def base(self):
        return ONNXDecoder(self.selection['model'])

    def make_source(self):
        data=RawCalibrationData(self.cfg,self.protocol)
        for trial in range(self.protocol['trials']):
            start=trial*10000
            raw=np.ones((8,6000),np.float32)
            raw[0]=20
            data.append(raw,np.arange(start,start+6000),start)
            data.finish(trial%2,start)
        write_json(self.session/'calibration.json',dict(complete=True,state=dict(ready=True,completed=40,total=40)))
        data.save(self.session)

    def request(self,**values):
        return dict(participant='DEMO_ONNX',model_id=self.entry['model_id'],mode='acp',host='127.0.0.1',port=4097,
            output=dict(mode='simulation'),decision=dict(mode='vote'),
            prompted=dict(enabled=True,loops=1,marking='software'),**values)

    def test_prompted_import_skips_calibration_even_with_stale_history_selection(self):
        request=self.request()
        selection,cfg,*_=self.controller.prepare(request)
        self.assertFalse(selection['requires_calibration'])
        self.assertTrue(selection['online_calibration_disabled'])
        with patch.object(self.controller,'start',side_effect=lambda value:value):
            self.assertEqual(self.controller.start_decoding(request)['session_purpose'],'decode')
            with self.assertRaisesRegex(ValueError,'暂时禁用'):
                self.controller.start_calibration(request)
        self.make_source()
        details=self.controller.model_details('DEMO_ONNX',self.entry['model_id'])
        self.assertFalse(details['requires_prompted_calibration'])
        self.assertEqual(details['previous_calibration']['id'],self.session_id)
        request.update(calibration_mode='reuse',calibration_source=self.session_id)
        selection,cfg,*_=self.controller.prepare(request)
        self.assertNotIn('calibration_reuse',cfg['session_metadata'])
        self.assertNotIn('imported_calibration',selection)
        with patch.object(self.controller,'start',side_effect=lambda value:value):
            self.assertEqual(self.controller.start_decoding(request)['session_purpose'],'decode')

    def test_shared_raw_replays_with_target_model_and_fits_its_own_bias(self):
        self.make_source()
        saved=SharedCalibrationStore(self.root/'sessions').load(self.session_id,self.cfg,self.protocol,arrays=True)
        wrapper=ImportedCalibrationDecoder(self.base(),self.cfg,dict(self.selection,sha256='a-different-model'))
        wrapper.restore_calibration(saved)
        self.assertTrue(wrapper.snapshot_calibration()['ready'])
        self.assertGreater(np.linalg.norm(wrapper.bias),.01)
        raw=np.ones((8,2000),np.float32)
        raw[0]=20
        reference=self.base().process(raw,np.arange(2000))[1]
        adjusted=wrapper.process(raw,np.arange(2000))[1]
        self.assertLess(abs(adjusted[0]-.5),abs(reference[0]-.5))
        self.assertAlmostEqual(adjusted.sum(),1.)
        # A genuinely different graph must fit its own offsets from the same EEG.
        from onnx import numpy_helper
        model=build_model()
        weights=numpy_helper.to_array(model.graph.initializer[0]).copy()*1.5
        model.graph.initializer[0].CopyFrom(numpy_helper.from_array(weights,'weights'))
        manifest=build_manifest();manifest['config']['device'].update(source='neuracle_acp',unit='uV')
        other=import_bundle(bundle_bytes(manifest,model),'DEMO_ONNX',registry=self.registry,import_root=self.root/'models')
        selection,cfg=prepare_subject_model('DEMO_ONNX',other['model_id'],registry=self.registry)
        second=ImportedCalibrationDecoder(ONNXDecoder(selection['model']),cfg,selection)
        second.restore_calibration(saved)
        self.assertFalse(np.allclose(wrapper.bias,second.bias))

    def test_class_order_channel_order_and_subset_reuse(self):
        self.make_source()
        target=copy.deepcopy(self.cfg)
        target['classes'].reverse()
        for c in target['classes']:c['name']='other_model_'+c['name']
        target['device']['channels'].reverse()
        saved=SharedCalibrationStore(self.root/'sessions').load(self.session_id,target,self.protocol,arrays=True)
        self.assertEqual(saved['labels'][0],1)
        self.assertEqual(saved['labels'][1],0)
        self.assertEqual(saved['raw'].shape,(40,8,6000))

    def test_mismatched_owner_task_timing_units_and_modified_data_rejected(self):
        self.make_source()
        store=SharedCalibrationStore(self.root/'sessions')
        for part,key,value in [('session_metadata','participant_id','OTHER'),('device','unit','V'),('device','sample_rate',500)]:
            cfg=copy.deepcopy(self.cfg);cfg[part][key]=value
            with self.assertRaises(ValueError):store.load(self.session_id,cfg,self.protocol)
        cfg=copy.deepcopy(self.cfg);cfg['classes'][0]['task_text']='another task'
        with self.assertRaises(ValueError):store.load(self.session_id,cfg,self.protocol)
        with self.assertRaises(ValueError):store.load(self.session_id,self.cfg,dict(self.protocol,rest_seconds=3))
        with (self.session/'calibration_raw.npz').open('ab') as handle:handle.write(b'changed')
        with self.assertRaises(ValueError):store.load(self.session_id,self.cfg,self.protocol)

    def test_full_twenty_round_calibration_counts_actual_two_and_six_second_phases(self):
        wrapper=ImportedCalibrationDecoder(self.base(),self.cfg,self.selection)
        index=0
        def feed(count):
            nonlocal index
            result=wrapper.process(np.ones((8,count),np.float32),np.arange(index,index+count))
            index+=count
            return result
        feed(3010)
        for trial in range(40):
            self.assertEqual(wrapper.phase,'rest')
            wrapper.acknowledge(wrapper.prompt_id,wrapper.session_id)
            feed(1999);self.assertEqual(wrapper.phase,'rest')
            feed(1);self.assertEqual(wrapper.phase,'imagery')
            wrapper.acknowledge(wrapper.prompt_id,wrapper.session_id)
            feed(5999);self.assertEqual(wrapper.completed,trial)
            feed(1);self.assertEqual(wrapper.completed,trial+1)
        self.assertTrue(wrapper.snapshot_calibration()['ready'])
        wrapper.save_calibration(self.session)
        saved=SharedCalibrationStore(self.root/'sessions').load(self.session_id,self.cfg,self.protocol,arrays=True)
        self.assertEqual(saved['raw'].shape,(40,8,6000))
        metadata=json.loads((self.session/'calibration.json').read_text(encoding='utf-8'))
        self.assertFalse(metadata['backbone_updated'])

    def test_imported_task_speech_uses_registered_task_texts(self):
        with patch.object(self.controller.speech,'prepare',side_effect=lambda texts: texts):
            self.assertEqual(self.controller.prepare_calibration_audio({'texts':['左手','右手','休息']}),['左手','右手','休息'])
            with self.assertRaises(ValueError):
                self.controller.prepare_calibration_audio({'texts':['not a registered task']})


if __name__=='__main__':unittest.main()
