import base64
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import Request,urlopen
import zipfile

import numpy as np

from examples.create_demo_model import build_manifest,build_model,bundle_bytes
from handbci.continuous_online import ContinuousController,VotingGate
from handbci.model_bundle import import_bundle,ONNXDecoder,validate_manifest,prepare_entry
from handbci.online_web import create_online_server
from handbci.prompted_control import validate_settings,PromptedControl
from handbci.subject_online import prepare_subject_model


class ModelBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.registry=self.root/'registry.json';self.imports=self.root/'models'
        self.c=ContinuousController(self.root/'sessions',registry=self.registry,import_root=self.imports)

    def import_model(self,manifest=None,model=None):
        return import_bundle(bundle_bytes(manifest,model),'DEMO_ONNX',registry=self.registry,import_root=self.imports)

    def test_import_check_and_reorder_subset_prediction(self):
        entry=self.import_model()
        selection,cfg=prepare_subject_model('DEMO_ONNX',entry['model_id'],registry=self.registry)
        self.assertEqual(selection['runtime'],'onnx')
        decoder=ONNXDecoder(selection['model'])
        raw=np.zeros((8,2000),np.float32);raw[0]=10
        stamp,p=decoder.process(raw,np.arange(2000))
        self.assertEqual(p.argmax(),0);self.assertAlmostEqual(p.sum(),1.)
        with self.assertRaises(ValueError):decoder.process(raw[:,:2],np.array([2001,2002]))
        decoder.recover_stream(9000,'gap')
        self.assertIsNone(decoder.process(raw[:,:100],np.arange(9000,9100)))
        self.assertEqual(self.import_model()['model_id'],entry['model_id'])
        self.assertEqual(len(list(self.imports.iterdir())),1)

    def test_missing_wrong_metadata_and_shape_are_rejected_without_registry(self):
        for key,value in [('participant_id','OTHER'),('input',dict(name='eeg',pipeline='raw',channels=['missing']))]:
            m=build_manifest();m[key]=value
            with self.assertRaises(ValueError):self.import_model(m)
        with self.assertRaises(ValueError):self.import_model(model=build_model(times=1999))
        with self.assertRaises(ValueError):self.import_model(model=build_model(n_classes=3))
        self.assertFalse(self.registry.exists())

    def test_zip_path_traversal_and_extra_files_do_not_extract(self):
        for bad in ('../escape.py','subdir/model.onnx','extra.py'):
            out=io.BytesIO()
            with zipfile.ZipFile(out,'w') as z:
                z.writestr('manifest.json',json.dumps(build_manifest()))
                z.writestr(bad,b'not a model')
            with self.assertRaises(ValueError):import_bundle(out.getvalue(),'DEMO_ONNX',registry=self.registry,import_root=self.imports)
        self.assertFalse((self.root/'escape.py').exists())

    def test_modified_manifest_or_weights_rejected(self):
        entry=self.import_model();selection,_=prepare_subject_model('DEMO_ONNX',entry['model_id'],registry=self.registry)
        p=Path(selection['model'])/'manifest.json';m=json.loads(p.read_text(encoding='utf-8'));m['input']['channels'].reverse();p.write_text(json.dumps(m),encoding='utf-8')
        with self.assertRaises(ValueError):prepare_subject_model('DEMO_ONNX',entry['model_id'],registry=self.registry)

    def test_fir_and_filterbank_graph_input_contract(self):
        for pipeline,bands in [('fir',False),('ifnet_filterbank',True)]:
            m=build_manifest();m['input']['pipeline']=pipeline
            # Graph already tested with genuine raw inference; wrong layouts must fail early.
            with self.assertRaises(ValueError):self.import_model(m,build_model(times=2000))

    def test_real_model_requires_units_and_artificial_model_cannot_drive_hand(self):
        m=build_manifest();m['config']['device'].update(source='neuracle_acp',unit='synthetic_uV')
        with self.assertRaises(ValueError):validate_manifest(m)
        entry=self.import_model()
        request=dict(participant='DEMO_ONNX',model_id=entry['model_id'],mode='simulated',output=dict(mode='inspire'))
        with self.assertRaises(ValueError):self.c.check(request)

    def test_external_real_model_does_not_require_project_training_provenance(self):
        m=build_manifest();m['config']['device'].update(source='neuracle_acp',unit='uV')
        m['config'].pop('paradigm_id',None)
        entry=self.import_model(m)
        checked=self.c.check(dict(participant='DEMO_ONNX',model_id=entry['model_id'],mode='acp',host='127.0.0.1',port=4097))
        self.assertFalse(checked['connected'])
        self.assertEqual(len(checked['classes']),2)

    def test_external_weight_paths_are_rejected_before_runtime(self):
        import onnx
        model=build_model()
        tensor=model.graph.initializer[0]
        tensor.data_location=onnx.TensorProto.EXTERNAL
        item=tensor.external_data.add();item.key='location';item.value='../outside.bin'
        with self.assertRaisesRegex(ValueError,'外部权重'):self.import_model(model=model)

    def test_fir_pipeline_performs_real_inference_and_retains_window_support(self):
        m=build_manifest();m['input']['pipeline']='fir'
        entry=self.import_model(m,build_model(times=500))
        selection,_=prepare_subject_model('DEMO_ONNX',entry['model_id'],registry=self.registry)
        decoder=ONNXDecoder(selection['model'])
        result=None
        for start in range(0,5000,100):
            value=decoder.process(np.random.default_rng(start).normal(size=(8,100)),np.arange(start,start+100))
            if value is not None:result=value
        self.assertIsNotNone(result)
        self.assertGreater(decoder.last_window['filter_history_seconds'],0)
        self.assertEqual(decoder.last_window['input_seconds'],2.)

    def test_actual_http_upload_and_full_simulated_decode(self):
        server=create_online_server(self.c,port=0)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:'+str(server.server_port)
        try:
            html=urlopen(base).read().decode()
            import re
            token=re.search('name="online-token" content="([^"]+)"',html).group(1)
            body=dict(participant='DEMO_ONNX',data=base64.b64encode(bundle_bytes()).decode())
            req=Request(base+'/api/import-model',json.dumps(body).encode(),headers={'Content-Type':'application/json','X-Online-Token':token,'Origin':base})
            entry=json.load(urlopen(req,timeout=15))
            request=dict(participant='DEMO_ONNX',model_id=entry['model_id'],mode='simulated',output=dict(mode='simulation'))
            self.assertFalse(self.c.check(request)['connected'])
            self.c.start_decoding(request)
            until=time.monotonic()+7
            while time.monotonic()<until and self.c.snapshot()['n_predictions']<3:
                self.c.snapshot(browser_seen=True);time.sleep(.1)
            self.assertGreaterEqual(self.c.snapshot()['n_predictions'],3,self.c.snapshot())
            self.c.stop();self.c.worker.join(8)
            path=Path(self.c.snapshot()['saved_path'])
            self.assertTrue((path/'predictions.jsonl').is_file())
            self.assertTrue((path/'data.h5').is_file())
        finally:
            self.c.stop()
            if self.c.worker:self.c.worker.join(8)
            server.shutdown();server.server_close();thread.join(2)

    def test_two_class_prompted_plan(self):
        classes=self.c.display_classes(build_manifest()['config']['classes'])
        gate=VotingGate(2)
        settings=validate_settings(dict(enabled=True,loops=3,seed=10,marking='software'),'acp',classes,gate,dict(mode='simulation'))
        control=PromptedControl(self.root,settings,classes,gate,None,lambda:0,1000,False)
        self.assertEqual(len(control.plan),6)
        for i in range(0,6,2):self.assertEqual({x['label'] for x in control.plan[i:i+2]},{0,1})


if __name__=='__main__':unittest.main()
