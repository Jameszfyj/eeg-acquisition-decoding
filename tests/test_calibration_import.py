from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import re
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import zipfile

import h5py
import numpy as np

from examples.create_demo_model import build_manifest,build_model,bundle_bytes
from handbci.calibration_import import CalibrationImports,read_acquisition
from handbci.config import write_json
from handbci.continuous_online import ContinuousController
from handbci.model_bundle import import_bundle,ONNXDecoder
from handbci.online_web import create_online_server
from handbci.shared_calibration import SharedCalibrationStore,ImportedCalibrationDecoder,imported_protocol
from handbci.subject_online import prepare_subject_model


class CalibrationImportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.registry=self.root/'models.json'
        self.controller=ContinuousController(self.root/'sessions',registry=self.registry,import_root=self.root/'models')
        manifest=build_manifest();manifest['config']['device'].update(source='neuracle_acp',unit='uV',sample_rate=250)
        self.entry=import_bundle(bundle_bytes(manifest,build_model(times=500)),'DEMO_ONNX',registry=self.registry,import_root=self.root/'models')
        self.selection,self.cfg=prepare_subject_model('DEMO_ONNX',self.entry['model_id'],registry=self.registry)
        self.protocol=imported_protocol(self.cfg)
        self.imports=CalibrationImports(self.controller)
        self.source=self.root/'acquired_elsewhere';self.source.mkdir()
        self.build_record()

    def build_record(self):
        cfg=copy.deepcopy(self.cfg)
        cfg['device']['channels'].reverse()
        device=dict(synthetic=False,settings=dict(fs=250,channels=8,points=5),labels=cfg['device']['channels'],
                    unit='uV',info=dict(type='Neusen-test',serial='12345678',cap='test-cap',labels_verified=True))
        collection=dict(loops=20,prepare_seconds=2.4,task_seconds=4.,rest_seconds=3.,
                        tasks=[dict(text=c['task_text']) for c in cfg['classes']])
        meta=dict(format='handbci.raw.v1',session_id='external-session',complete=True,participant_id='DEMO_ONNX',
            config=cfg,source='neuracle_acp',device=device,protocol=collection,error='',
            quality=dict(training_compatible=False,reasons=['device_event_time_semantics_unverified'],
                         device_time_gaps=[],receive_faults=[],unverified_trial_ids=[],physical_onset_measured=False))
        n=40*2350+1000
        eeg=np.full((8,n),-999,np.float32);events=[]
        for trial in range(40):
            start=trial*2350+600
            eeg[:,start:start+1000]=trial+1
            for kind,point in [('cue',start),('task_end',start+1000)]:
                events.append(dict(kind=kind,trial_id=trial+1,label=trial%2,sample_index=point,echo_verified=True))
        with h5py.File(self.source/'data.h5','w') as h:
            h.attrs.update(handbci_format='handbci.raw.v1',complete=True,metadata_json=json.dumps(meta))
            h.create_dataset('eeg',data=eeg)
            h.create_dataset('source_indices',data=np.arange(n,dtype=np.int64))
            h.create_dataset('device_time_ms',data=np.arange(n,dtype=np.float64)*4)
            h.create_dataset('events_json',data=[json.dumps(e) for e in events],dtype=h5py.string_dtype('utf-8'))
        write_json(self.source/'session.json',meta)
        return meta

    def change_meta(self, edit):
        meta=json.loads((self.source/'session.json').read_text(encoding='utf-8'))
        edit(meta)
        write_json(self.source/'session.json',meta)
        with h5py.File(self.source/'data.h5','a') as h:h.attrs['metadata_json']=json.dumps(meta)

    def upload(self, zipped=False):
        if zipped:
            out=io.BytesIO()
            with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
                for name in ('session.json','data.h5'):z.write(self.source/name,'some_session/'+name)
                z.writestr('some_session/extra_notes.txt','not extracted')
            files={'session.zip':out.getvalue()}
        else:files={n:(self.source/n).read_bytes() for n in ('session.json','data.h5')}
        opened=self.imports.open(dict(participant='DEMO_ONNX',model_id=self.entry['model_id'],
            files=[dict(name=n,size=len(v)) for n,v in files.items()]))
        for name,content in files.items():
            for start in range(0,len(content),1024**2):
                self.imports.append(opened['id'],name,start,content[start:start+1024**2])
        return opened['id']

    def test_folder_four_second_data_imports_and_reuses_without_preparation_samples(self):
        result=self.imports.finish({'id':self.upload()})
        sid=result['summary']['id']
        self.assertTrue(result['summary']['external_acquisition'])
        self.assertEqual((result['summary']['prepare_seconds'],result['summary']['imagery_seconds']),(2.4,4.))
        saved=SharedCalibrationStore(self.root/'sessions').load(sid,self.cfg,self.protocol,arrays=True)
        self.assertEqual(saved['raw'].shape,(40,8,1000))
        self.assertEqual(float(saved['raw'].min()),1)
        self.assertEqual(float(saved['raw'].max()),40)
        wrapper=ImportedCalibrationDecoder(ONNXDecoder(self.selection['model']),self.cfg,self.selection)
        wrapper.restore_calibration(saved)
        self.assertTrue(wrapper.snapshot_calibration()['ready'])
        self.assertTrue((self.root/'sessions'/sid/'source/data.h5').is_file())
        self.assertTrue((self.source/'data.h5').is_file())
        details=self.controller.model_details('DEMO_ONNX',self.entry['model_id'])
        self.assertEqual(details['previous_calibration']['id'],sid)
        duplicate=self.imports.finish({'id':self.upload(zipped=True)})
        self.assertTrue(duplicate['already_imported'])
        self.assertEqual(duplicate['summary']['id'],sid)
        self.assertFalse(list(self.imports.root.iterdir()))

    def test_rejects_wrong_owner_task_rounds_duration_and_partial_data(self):
        cases=[lambda m:m.update(participant_id='OTHER'),
            lambda m:m['protocol'].update(loops=19),lambda m:m['protocol'].update(task_seconds=3.),
            lambda m:m['protocol']['tasks'][0].update(text='different task'),
            lambda m:m.update(complete=False),lambda m:m['quality']['reasons'].append('unverified_or_missing_labels')]
        for edit in cases:
            self.change_meta(edit)
            with self.assertRaises(ValueError):read_acquisition(self.source,self.cfg,self.protocol)
            self.build_record()

    def test_label_boundaries_gaps_and_metadata_mismatch_are_rejected(self):
        with h5py.File(self.source/'data.h5','a') as h:h['device_time_ms'][2000]+=20
        with self.assertRaisesRegex(ValueError,'缺口'):read_acquisition(self.source,self.cfg,self.protocol)
        self.build_record()
        with h5py.File(self.source/'data.h5','a') as h:
            event=json.loads(h['events_json'].asstr()[1]);event['label']=1;h['events_json'][1]=json.dumps(event)
        with self.assertRaisesRegex(ValueError,'标签'):read_acquisition(self.source,self.cfg,self.protocol)
        self.build_record()
        self.change_meta(lambda m:m['device']['info'].update(labels_verified=False))
        with self.assertRaises(ValueError):read_acquisition(self.source,self.cfg,self.protocol)
        self.build_record()
        (self.source/'session.json').write_text('{}',encoding='utf-8')
        with self.assertRaises(ValueError):read_acquisition(self.source,self.cfg,self.protocol)

    def test_hdf5_external_links_are_not_followed(self):
        with h5py.File(self.source/'data.h5','a') as h:
            del h['eeg'];h['eeg']=h5py.ExternalLink('outside.h5','eeg')
        with self.assertRaisesRegex(ValueError,'外部链接'):read_acquisition(self.source,self.cfg,self.protocol)

    def test_upload_order_size_and_busy_session_guards(self):
        opened=self.imports.open(dict(participant='DEMO_ONNX',model_id=self.entry['model_id'],files=[dict(name='session.zip',size=9)]))
        with self.assertRaises(ValueError):self.imports.append(opened['id'],'../outside',0,b'x')
        with self.assertRaises(ValueError):self.imports.append(opened['id'],'session.zip',1,b'x')
        with self.assertRaises(ValueError):self.imports.finish({'id':opened['id']})
        self.imports.cancel({'id':opened['id']})
        self.controller.state['running']=True
        with self.assertRaises(ValueError):self.upload()

    def test_slightly_short_event_boundary_is_trimmed_inside_imagery(self):
        with h5py.File(self.source/'data.h5','a') as h:
            event=json.loads(h['events_json'].asstr()[1]);event['sample_index']-=1
            h['events_json'][1]=json.dumps(event)
        result=self.imports.finish({'id':self.upload()})
        saved=SharedCalibrationStore(self.root/'sessions').load(result['summary']['id'],self.cfg,self.protocol,arrays=True)
        self.assertEqual(saved['raw'].shape,(40,8,999))
        self.assertEqual(float(saved['raw'].min()),1.)

    def test_variable_preparation_uses_actual_cue_indices(self):
        # The nominal preparation can change, and every trial can have its own delay.
        self.change_meta(lambda m:m['protocol'].update(prepare_seconds=1.7))
        with h5py.File(self.source/'data.h5','a') as h:
            h['eeg'][:]=np.full(h['eeg'].shape,-999,np.float32)
            for trial in range(40):
                start=trial*2350+300+(trial%5)*110
                h['eeg'][:,start:start+1000]=trial+1
                for i,point in [(trial*2,start),(trial*2+1,start+1000)]:
                    event=json.loads(h['events_json'].asstr()[i]);event['sample_index']=point
                    h['events_json'][i]=json.dumps(event)
        result=self.imports.finish({'id':self.upload()})
        saved=SharedCalibrationStore(self.root/'sessions').load(result['summary']['id'],self.cfg,self.protocol,arrays=True)
        self.assertEqual(result['summary']['prepare_seconds'],1.7)
        for trial in range(40):np.testing.assert_array_equal(saved['raw'][trial],trial+1)

    def test_unsafe_and_multiple_session_archives_leave_no_registered_calibration(self):
        for names in [('data.h5','../session.json'),('data.h5','a/session.json','b/session.json')]:
            out=io.BytesIO()
            with zipfile.ZipFile(out,'w') as z:
                for name in names:z.writestr(name,b'invalid')
            content=out.getvalue()
            opened=self.imports.open(dict(participant='DEMO_ONNX',model_id=self.entry['model_id'],files=[dict(name='session.zip',size=len(content))]))
            self.imports.append(opened['id'],'session.zip',0,content)
            with self.assertRaises(ValueError):self.imports.finish({'id':opened['id']})
            self.assertFalse(list((self.root/'sessions').glob('*/calibration.json')))
            self.assertFalse(list(self.imports.root.iterdir()))

    def test_authenticated_chunked_http_import(self):
        server=create_online_server(self.controller,port=0)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            base='http://127.0.0.1:'+str(server.server_port)
            html=urlopen(base).read().decode()
            token=re.search('name="online-token" content="([^"]+)"',html).group(1)
            def post(path,value,raw=False,authenticated=True):
                headers={'Content-Type':'application/octet-stream' if raw else 'application/json'}
                if authenticated:headers['X-Online-Token']=token
                request=Request(base+path,value if raw else json.dumps(value).encode(),headers=headers)
                return json.load(urlopen(request,timeout=30))
            files={n:(self.source/n).read_bytes() for n in ('session.json','data.h5')}
            body=dict(participant='DEMO_ONNX',model_id=self.entry['model_id'],files=[dict(name=n,size=len(v)) for n,v in files.items()])
            with self.assertRaises(HTTPError):post('/api/calibration/import/open',body,authenticated=False)
            opened=post('/api/calibration/import/open',body)
            for name,content in files.items():
                post('/api/calibration/import/chunk?'+urlencode(dict(id=opened['id'],name=name,offset=0)),content,raw=True)
            result=post('/api/calibration/import/finish',{'id':opened['id']})
            self.assertEqual(result['summary']['calibration_trials'],40)
        finally:
            server.shutdown();server.server_close();thread.join(2)


if __name__=='__main__':unittest.main()
