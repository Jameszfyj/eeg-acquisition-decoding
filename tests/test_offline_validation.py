import base64
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.request import Request, urlopen

import h5py
import numpy as np

from examples.create_demo_model import build_manifest, build_model, bundle_bytes
from handbci.model_bundle import import_bundle, ONNXDecoder
from handbci.studio.devices import Packet
from handbci.studio.experiment import validate_protocol
from handbci.studio.server import make_server
from handbci.studio.storage import SessionWriter
from handbci.studio.validation import TrialValidation, import_validation_model, vote_scores

ROOT = Path(__file__).resolve().parents[1]


class OfflineValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.m = build_manifest()
        self.p = validate_protocol(dict(tasks=[dict(text='right'),dict(text='left')], loops=1,
            sound=False, prepare_seconds=1, task_seconds=4, rest_seconds=1))
        self.description = dict(settings=dict(fs=1000,channels=8,points=5),
            labels=self.m['config']['device']['channels'], count_scale=.05,
            unit='synthetic_uV', timestamp_verified=True, synthetic=True)

    def verifier(self, pipeline='raw'):
        m = copy.deepcopy(self.m)
        m['input']['pipeline'] = pipeline
        entry = import_bundle(bundle_bytes(m,build_model(times=2000 if pipeline=='raw' else 500)),
            'DEMO_ONNX', registry=self.root/'models.json', import_root=self.root/'models')
        return TrialValidation(entry,'DEMO_ONNX',self.description,self.p)

    def marks(self, trial=1, prepare=1000, cue=2000, end=6000, label=1, verified=True):
        return {k:dict(kind=k,trial_id=trial,label=label,device_time_ms=t,echo_verified=verified)
                for k,t in [('prepare',prepare),('cue',cue),('task_end',end)]}

    def test_whole_trial_half_second_grid_no_rest_leak_and_name_mapping(self):
        v = self.verifier()
        x = np.zeros((8,9000),np.float32);x[0,:6000]=100;x[2,6000:]=10000
        v.batches.append((x,np.arange(9000),0))
        log = io.StringIO();result=v.evaluate(self.marks(),log)
        rows=[json.loads(s) for s in log.getvalue().splitlines()]
        self.assertEqual([r['input_window']['model_start_index'] for r in rows],[2000,2500,3000,3500,4000])
        self.assertEqual(result['votes'],[5,0])
        self.assertEqual(result['target_class'],0)  # Protocol order is deliberately reversed.
        self.assertTrue(result['correct'])
        self.assertLess(result['last_window']['source_end_index'],6000)

    def test_fir_replay_matches_continuous_preprocessing_and_live_transform(self):
        v = self.verifier('fir')
        x=np.random.default_rng(5).normal(size=(8,9000)).astype(np.float32)*100
        v.batches.append((x,np.arange(9000),0))
        log=io.StringIO();result=v.evaluate(self.marks(),log)
        full=ONNXDecoder(v.folder);y,ix=full.stream.process(x,np.arange(9000))
        rows=[json.loads(s) for s in log.getvalue().splitlines()]
        self.assertEqual(result['windows'],3)
        for row in rows:
            a=np.searchsorted(ix,row['input_window']['model_start_index'])
            expected=full.predict_window(y[full.order,a:a+full.n_time])
            np.testing.assert_allclose(row['probabilities'],expected,atol=1e-7,rtol=1e-6)
            self.assertGreaterEqual(row['input_window']['source_start_index'],2000)

    def test_gap_skips_only_affected_slices_and_later_trials_keep_scoring(self):
        v=self.verifier();session=self.root/'session';session.mkdir();v.start(session)
        x=np.ones((8,15000),np.float32);times=np.arange(15000,dtype=float);times[3500:]+=5
        v.submit('data',(x,times,0))
        for marks in (self.marks(end=6005),self.marks(2,7005,8005,12005),self.marks(3,13005,14005,16005)):
            for row in marks.values():v.submit('event',row)
        result=v.finish()
        self.assertEqual(result['status'],'finished',result)
        self.assertEqual([r['status'] for r in v.results],['completed','completed','transmission_error'])
        self.assertEqual(v.results[0]['skipped_windows'],4)
        self.assertEqual(v.results[0]['windows'],1)
        self.assertEqual(v.results[1]['windows'],5)
        self.assertIsNone(v.results[1]['predicted_class'])  # Exact graph tie abstains.
        self.assertFalse(v.results[1]['correct'])  # Unresolved verified trials stay in the denominator.
        saved=json.loads((session/'onnx_validation/summary.json').read_text(encoding='utf8'))
        self.assertEqual(len(saved['trials']),3)

    def test_estimated_markers_predict_but_do_not_claim_verified_accuracy(self):
        v=self.verifier();x=np.zeros((8,9000),np.float32);x[0]=100
        v.batches.append((x,np.arange(9000),0))
        result=v.evaluate(self.marks(verified=False),io.StringIO())
        self.assertEqual(result['predicted_class'],0)
        self.assertFalse(result['label_verified']);self.assertIsNone(result['correct'])
        with self.assertRaisesRegex(ValueError,'窗口不足'):
            v.evaluate(self.marks(end=2500),io.StringIO())

    def test_fir_gap_support_is_excluded_and_remaining_inputs_match_clean_segment(self):
        v=self.verifier('fir')
        x=np.random.default_rng(33).normal(size=(8,12000)).astype(np.float32)*100
        ix=np.r_[np.arange(4000),np.arange(4100,12000)]
        v.batches.append((x[:,ix],ix.astype(float),0))
        captured=[];predict=v.decoder.predict_window
        def capture(y):
            captured.append(y.copy());return predict(y)
        v.decoder.predict_window=capture
        log=io.StringIO();result=v.evaluate(self.marks(end=10000),log)
        rows=[json.loads(s) for s in log.getvalue().splitlines()]
        valid=[r for r in rows if 'probabilities' in r]
        self.assertEqual(result['skipped_windows'],5)
        self.assertEqual(result['windows'],6)
        clean=ONNXDecoder(v.folder);clean.stream.reset(4100)
        values,retained=clean.stream.process(x[:,4100:],np.arange(4100,12000))
        for y,row in zip(captured,valid):
            window=row['input_window']
            self.assertGreaterEqual(window['source_start_index'],4100)
            a=np.searchsorted(retained,window['model_start_index'])
            np.testing.assert_allclose(y,values[clean.order,a:a+clean.n_time],atol=1e-7,rtol=1e-6)

    def test_sequence_only_gap_and_all_lost_windows_are_recorded(self):
        v=self.verifier();x=np.ones((8,9000),np.float32)
        v.batches.append((x,np.arange(9000),0,[3500]))
        result=v.evaluate(self.marks(),io.StringIO())
        self.assertEqual(result['skipped_windows'],3)
        self.assertEqual(result['windows'],2)
        v.batches.clear()
        v.batches.append((x,np.arange(9000),0,list(range(1000,9000,500))))
        result=v.evaluate(self.marks(),io.StringIO())
        self.assertEqual(result['status'],'transmission_error')
        self.assertEqual(result['reason'],'传输异常')
        self.assertEqual(result['skipped_windows'],5)
        self.assertEqual(result['windows'],0)

    def test_missing_trial_tail_still_votes_on_valid_slices(self):
        v=self.verifier();ix=np.r_[np.arange(4500),np.arange(7000,9000)]
        x=np.zeros((8,len(ix)),np.float32);x[0]=100
        v.batches.append((x,ix.astype(float),0))
        result=v.evaluate(self.marks(),io.StringIO())
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['windows'],2)
        self.assertEqual(result['skipped_windows'],3)
        self.assertEqual(result['votes'],[2,0])

    def test_entire_trial_inside_gap_reports_transmission_error_then_recovers(self):
        v=self.verifier();session=self.root/'all_lost';session.mkdir();v.start(session)
        ix=np.r_[np.arange(1500),np.arange(7000,15000)]
        x=np.zeros((8,len(ix)),np.float32);x[0]=100
        v.submit('data',(x,ix.astype(float),0))
        for marks in (self.marks(),self.marks(2,7500,8000,12000)):
            for row in marks.values():v.submit('event',row)
        result=v.finish()
        self.assertEqual(result['status'],'finished')
        self.assertEqual([r['status'] for r in v.results],['transmission_error','completed'])
        self.assertEqual(v.results[0]['reason'],'传输异常')
        self.assertEqual(v.results[0]['skipped_windows'],5)
        self.assertEqual(v.results[1]['votes'],[5,0])

    def test_delayed_tail_previews_valid_prefix_then_completes_all_votes(self):
        v=self.verifier();session=self.root/'outage';session.mkdir();v.start(session)
        self.addCleanup(v.finish)
        x=np.zeros((8,4500),np.float32);x[0]=100
        v.submit('data',(x,np.arange(4500,dtype=float),0))
        marks=self.marks()
        marks['task_end']['mono_time_ns']=time.perf_counter_ns()
        for row in marks.values():v.submit('event',row)
        deadline=time.monotonic()+3
        while v.snapshot()['status'] != 'waiting_data' and time.monotonic()<deadline:
            time.sleep(.02)
        result=v.snapshot()['latest']
        self.assertEqual(result['status'],'waiting_data',result)
        self.assertEqual(v.snapshot()['completed_trials'],0)
        self.assertTrue(result['provisional'])
        self.assertEqual(result['votes'],[2,0])
        self.assertEqual(result['skipped_windows'],0)
        self.assertEqual(result['pending_windows'],3)
        self.assertEqual(result['gap_count'],0)
        self.assertFalse(result['end_boundary_received'])
        self.assertFalse(v.closed.is_set())
        tail=np.zeros((8,2000),np.float32);tail[0]=100
        v.submit('data',(tail,np.arange(4500,6500,dtype=float),4500))
        v.finish()
        self.assertEqual(len(v.results),1)
        self.assertEqual(v.results[0]['votes'],[5,0])
        self.assertEqual(v.results[0]['skipped_windows'],0)
        self.assertTrue(v.results[0]['end_boundary_received'])
        rows=(session/'onnx_validation'/'predictions.jsonl').read_text().splitlines()
        self.assertEqual(len(rows),5)  # Provisional votes must not be duplicated in the journal.

    def test_no_received_trial_samples_reports_transmission_error_at_finish(self):
        v=self.verifier();session=self.root/'no_trial_data';session.mkdir();v.start(session)
        v.submit('data',(np.zeros((8,1500),np.float32),np.arange(1500,dtype=float),0))
        for row in self.marks().values():v.submit('event',row)
        result=v.finish()
        self.assertEqual(result['status'],'finished')
        self.assertEqual(v.results[0]['status'],'transmission_error')
        self.assertEqual(v.results[0]['reason'],'传输异常')
        self.assertEqual(v.results[0]['windows'],0)

    def test_native_prepare_grid_matches_full_stream_input(self):
        from tests.test_native_onnx import fixture
        m,model=fixture(True)
        entry=import_bundle(bundle_bytes(m,model),'P01',registry=self.root/'native.json',import_root=self.root/'native')
        p=validate_protocol(dict(tasks=[dict(text=c['name']) for c in reversed(m['config']['classes'])],sound=False))
        d=dict(self.description,settings=dict(fs=1000,channels=32,points=5),
               labels=m['config']['device']['channels'],unit='uV')
        v=TrialValidation(entry,'P01',d,p)
        x=np.random.default_rng(9).normal(size=(32,10000)).astype(np.float32)
        v.batches.append((x,np.arange(10000),0))
        full=ONNXDecoder(v.folder);values,ix=full.stream.process(x,np.arange(10000))
        captured=[];predict=v.decoder.predict_window
        def capture(y):
            captured.append(y.copy());return predict(y)
        v.decoder.predict_window=capture
        log=io.StringIO();r=v.evaluate(self.marks(prepare=2001,cue=4500,end=8500),log)
        rows=[json.loads(s) for s in log.getvalue().splitlines()]
        self.assertEqual(r['windows'],10)
        self.assertEqual([q['input_window']['model_start_index'] for q in rows],list(range(2000,6501,500)))
        for y,row in zip(captured,rows):
            a=np.searchsorted(ix,row['input_window']['model_start_index'])
            np.testing.assert_allclose(y,values[full.order,a:a+500],atol=1e-7,rtol=1e-6)

    def test_votes_use_all_windows_and_ties_are_not_arbitrary(self):
        r=vote_scores([[.5,.5],[.9,.1],[.2,.8]])
        self.assertEqual(r['votes'],[1,1]);self.assertEqual(r['abstained_windows'],1)
        self.assertEqual(r['predicted_class'],0)
        self.assertIsNone(vote_scores([[.8,.2],[.2,.8]])['predicted_class'])

    def test_writer_keeps_raw_data_and_persists_validation_sidecar(self):
        v=self.verifier()
        writer=SessionWriter(self.root/'sessions','DEMO_ONNX',self.description,self.p,self.m['config'],validation=v)
        for i in range(0,13000,500):
            counts=np.zeros((8,500),np.int32);counts[0]=2000
            writer.submit('packet',Packet('eeg',time.time_ns(),time.perf_counter_ns(),i,counts=counts,packet_id=(i//500)&255))
        for marks in (self.marks(),self.marks(2,7000,8000,12000,label=0)):
            for row in marks.values():writer.submit('event',row)
        summary=writer.close(True)
        self.assertEqual(summary['samples'],13000)
        self.assertEqual(summary['onnx_validation']['completed_trials'],2)
        self.assertEqual(summary['onnx_validation']['correct_trials'],1)
        self.assertTrue(summary['complete'])
        with h5py.File(writer.path/'data.h5') as h:
            self.assertEqual(h['eeg'].shape,(8,13000))
            self.assertEqual(json.loads(h.attrs['metadata_json'])['onnx_validation']['status'],'finished')
        saved=json.loads((writer.path/'onnx_validation/summary.json').read_text(encoding='utf8'))
        self.assertEqual(saved['stride_seconds'],.5)
        self.assertEqual(len(saved['trials']),2)
        self.assertTrue((writer.path/'onnx_validation/model.onnx').is_file())

    def test_mismatched_contracts_reject_before_recording(self):
        v=self.verifier()
        for key,value in [('unit','uV'),('labels',['missing']*8)]:
            description=dict(self.description,**{key:value})
            with self.assertRaises(ValueError):TrialValidation(v.entry,'DEMO_ONNX',description,self.p)
        with self.assertRaises(ValueError):TrialValidation(v.entry,'OTHER',self.description,self.p)
        wrong=copy.deepcopy(self.p);wrong['tasks'][0]['text']='未知任务'
        with self.assertRaises(ValueError):TrialValidation(v.entry,'DEMO_ONNX',self.description,wrong)

    def test_inference_failure_does_not_invalidate_raw_recording(self):
        v=self.verifier()
        def fail(_):
            raise OSError('injected ONNX failure')
        v.decoder.predict_window=fail
        writer=SessionWriter(self.root/'sessions','DEMO_ONNX',self.description,self.p,self.m['config'],validation=v)
        writer.submit('packet',Packet('eeg',time.time_ns(),time.perf_counter_ns(),0,
            counts=np.ones((8,7000),np.int32),packet_id=0))
        for row in self.marks().values():writer.submit('event',row)
        summary=writer.close(True)
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['samples'],7000)
        self.assertEqual(summary['onnx_validation']['status'],'failed')
        self.assertIn('injected ONNX failure',summary['onnx_validation']['error'])

    def test_http_upload_zip_and_bare_onnx_manifest(self):
        server=make_server(self.root,0)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            state=json.load(urlopen(base+'/api/state'))
            headers={'Content-Type':'application/json','Origin':base,'X-Studio-Token':state['token']}
            body=dict(participant='DEMO_ONNX',filename='test.zip',data=base64.b64encode(bundle_bytes()).decode())
            answer=json.load(urlopen(Request(base+'/api/validation/model',json.dumps(body).encode(),headers=headers)))
            self.assertTrue(answer['ok'])
            self.assertEqual(json.load(urlopen(base+'/api/state'))['validation_model']['model_id'],answer['result']['model_id'])
            data=base64.b64encode(build_model().SerializeToString()).decode()
            with self.assertRaisesRegex(ValueError,'manifest'):
                import_validation_model(self.root,'DEMO_ONNX',data,'model.onnx')
            entry=import_validation_model(self.root,'DEMO_ONNX',data,'model.onnx',self.m)
            self.assertEqual(entry['model_id'],answer['result']['model_id'])
        finally:
            server.shutdown();server.server_close();thread.join(2)


if __name__ == '__main__':
    unittest.main()
