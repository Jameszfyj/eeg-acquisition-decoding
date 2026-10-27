"""No physical device or serial port is opened by these tests."""
import copy
import json
import os
from pathlib import Path
import queue
import socket
import struct
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import h5py
import numpy as np
from handbci.studio.devices import Packet, NeusenAdapter, event_code
from handbci.studio.vendor import acp
from handbci.studio.controller import Controller, DeviceClock
from handbci.studio.experiment import validate_protocol, Experiment
from handbci.studio.storage import SessionWriter
from handbci.studio.triggers import TriggerManager
from handbci.data import load_session
from handbci.config import validate_config, require_compatible
from handbci.acquisition import connect
from handbci.studio.server import make_server
from urllib.request import Request, urlopen
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
P = validate_protocol({'loops':1,'prepare_seconds':.12,'task_seconds':.2,'rest_seconds':.1,
    'block_rest_seconds':0,'initial_rest_seconds':.1,'tail_seconds':.12,'sound':False,'randomize':False})
DESC = {'settings':{'fs':1000,'channels':8,'points':5},'labels':[f'C{i}' for i in range(8)],
    'count_scale':.25,'unit':'synthetic_uV','timestamp_verified':True,'synthetic':True}


def packet(ms=0,n=5):
    return Packet('eeg',time.time_ns(),time.perf_counter_ns(),ms,counts=np.ones((8,n),np.int32),payload=b'raw',raw_device_ms=ms)


class ACPFixture:
    """Real socket + ACP framing on an ephemeral loopback port, fragmented responses."""
    def __init__(self, bad_echo=False):
        self.sock = socket.socket()
        self.sock.bind(('127.0.0.1',0))
        self.port = self.sock.getsockname()[1]
        self.sock.listen(1)
        self.bad_echo, self.configs, self.client = bad_echo, [], None
        self.thread = threading.Thread(target=self.run,daemon=True)
        self.thread.start()

    def send(self, raw):
        for i in range(0,len(raw),7): self.client.sendall(raw[i:i+7])

    def run(self):
        try:
            self.client,_ = self.sock.accept()
            buf=bytearray()
            while True:
                chunk=self.client.recv(65536)
                if not chunk: break
                buf.extend(chunk)
                while len(buf)>=6:
                    h,plen=buf[2],struct.unpack_from('<H',buf,3)[0]
                    total=h+plen+4
                    if len(buf)<total:break
                    f=bytes(buf[:total]);del buf[:total]
                    if f[5]!=acp.PKG_COMMAND:continue
                    module,cmd,seq=f[6:9]
                    payload=f[h:h+plen]
                    hello=cmd==acp.CMD_HELLO
                    if hello: payload=bytes([1,10])+b'NSW3BA0064XX'+struct.pack('<I',12345678)+bytes([32,35])
                    elif cmd==acp.CMD_SETMODPARA:
                        self.configs.append(struct.unpack('<BBHHB',payload))
                        if self.bad_echo: payload=payload[:2]
                    body=bytes([module,cmd,0,seq,1,0])+struct.pack('<II',0,0)
                    self.send(acp.build_frame(acp.TOKEN_HELLO if hello else acp.TOKEN_CMD,20,acp.PKG_RESPONSE,
                        body,payload,acp.TAIL_HELLO if hello else acp.TAIL_CMD))
                    if cmd==acp.CMD_START:
                        ch=self.configs[-1][2];pts=self.configs[-1][4]
                        values=np.zeros((ch,pts,3),np.uint8);values[0,0]=[255,255,255]
                        self.send(acp.build_frame(acp.TOKEN_CMD,17,acp.PKG_ALERT,bytes([2,1,7])+struct.pack('<II',100,100),values.tobytes(),acp.TAIL_CMD))
        except OSError: pass

    def close(self):
        if self.client:
            try:self.client.shutdown(socket.SHUT_RDWR)
            except OSError:pass
            self.client.close()
        self.sock.close()
        self.thread.join(2)


class StudioTests(unittest.TestCase):
    def setUp(self):
        diagnostic = patch.object(Controller, '_save_trigger_check')
        diagnostic.start()
        self.addCleanup(diagnostic.stop)

    def cfg(self):
        cfg=json.loads((ROOT/'configs/demo_6class.json').read_text(encoding='utf-8-sig'))
        cfg['device']['channels']=DESC['labels'];cfg['device']['stream_channels']=9
        return cfg

    def test_clock_wrap_reset_and_event_isolation(self):
        c=DeviceClock();self.assertEqual(c.eeg(2**32-5),(2**32-5,False))
        self.assertEqual(c.event(2),2**32+2)
        self.assertEqual(c.last,2**32-5)
        self.assertEqual(c.eeg(0),(2**32,False))
        c.eeg(100);self.assertTrue(c.eeg(0)[1])

    def test_schedule_complete_rounds_and_static_boundary(self):
        for n in range(2,7):
            p={**P,'tasks':[{'text':str(i),'action':'custom'} for i in range(n)],'loops':3}
            e=Experiment(None,p)
            self.assertEqual([r['trial_id'] for r in e.sequence],list(range(1,n*3+1)))
            for k in range(3):self.assertEqual(sorted(r['label'] for r in e.sequence[k*n:(k+1)*n]),list(range(n)))

    def test_condition_validation_and_compatibility(self):
        p=validate_protocol({'condition':'  闭眼 + 轻音乐  ','condition_notes':'  安静坐姿\n第2次采集  '})
        self.assertEqual(p['condition'],'闭眼 + 轻音乐')
        self.assertEqual(p['condition_notes'],'安静坐姿\n第2次采集')
        for value in ('', '   ', 'x'*81, '闭眼\n常规', None, 123):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):validate_protocol({'condition':value})
        for value in ('x'*1001, None, [], '\x00'):
            with self.assertRaises(ValueError):validate_protocol({'condition_notes':value})
        a=self.cfg();b=copy.deepcopy(a)
        require_compatible(a,b)  # Existing records with no condition remain compatible with each other.
        a['session_metadata']={'recording_context':{'condition':'常规','notes':'备注A'}}
        b['session_metadata']={'recording_context':{'condition':'闭眼','notes':'备注A'}}
        with self.assertRaisesRegex(ValueError,'recording_condition'):require_compatible(a,b)
        b['session_metadata']['recording_context']={'condition':'常规','notes':'备注B'}
        require_compatible(a,b)  # Notes do not create new conditions.

    def test_condition_persists_in_incomplete_manifest_and_h5(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as root:
            p={**P,'condition':'自定义：闭眼','condition_notes':'坐姿\n环境较安静'}
            w=SessionWriter(root,'CONTEXT',DESC,p,self.cfg())
            expected={'condition':p['condition'],'notes':p['condition_notes']}
            initial=json.loads((w.path/'session.json').read_text(encoding='utf-8'))
            self.assertEqual(initial['recording_context'],expected)
            self.assertFalse(initial['complete'])
            w.submit('packet',packet());summary=w.close(False,'中止')
            self.assertEqual(summary['recording_context'],expected)
            self.assertEqual(json.loads((w.path/'session.json').read_text(encoding='utf-8'))['recording_context'],expected)
            with h5py.File(w.path/'data.h5') as h:
                self.assertEqual(json.loads(h.attrs['recording_context_json']),expected)
                self.assertEqual(json.loads(h.attrs['metadata_json'])['recording_context'],expected)

    def test_acp_real_link_echo_and_impedance_restore(self):
        f=ACPFixture();packets=[];a=NeusenAdapter(packets.append,lambda *x:None)
        try:
            a.connect({'host':'127.0.0.1','port':f.port,'channels':8})
            end=time.monotonic()+2
            while not packets and time.monotonic()<end:time.sleep(.01)
            self.assertEqual(packets[0].counts[0,0],-1)
            a.impedance(True);a.impedance(False)
            self.assertEqual(f.configs[0],f.configs[-1])
            self.assertEqual(f.configs[1][1:],(13,8,250,1))
        finally:a.disconnect();f.close()

    def test_acp_bad_echo_rejected(self):
        f=ACPFixture(True);a=NeusenAdapter(lambda x:None,lambda *x:None)
        try:
            with self.assertRaisesRegex(ValueError,'7-byte'):a.connect({'host':'127.0.0.1','port':f.port,'channels':8})
            self.assertFalse(a.connected)
        finally:f.close()

    def test_parser_crc_tail_header_and_unknown_event(self):
        seen=[];errors=[];details=[]
        l=acp.Link('',0,seen.append,lambda k,d:(errors.append(k),details.append((k,d))))
        body=bytes([13,1,1])+struct.pack('<II',5,5)
        frame=acp.build_frame(acp.TOKEN_CMD,17,acp.PKG_ALERT,body,b'unknown',acp.TAIL_CMD)
        corrupt=bytearray(frame);corrupt[-4]^=1
        tail=acp.build_frame(acp.TOKEN_CMD,17,acp.PKG_ALERT,body,b'?',0)
        short=acp.build_frame(acp.TOKEN_CMD,6,acp.PKG_ALERT,b'',b'',acp.TAIL_CMD)
        l._buf.extend(corrupt+tail+short+frame);l._drain(time.time_ns())
        self.assertEqual(len(seen),1)
        self.assertTrue({'crc_error','tail_error','header_error'}<=set(errors))
        crc_detail=next(d for k,d in details if k=='crc_error')
        self.assertEqual(crc_detail['candidate_frame_hex'],corrupt.hex())
        self.assertNotEqual(crc_detail['received_crc'],crc_detail['calculated_crc'])
        self.assertIsNone(event_code(seen[0].payload));self.assertEqual(event_code(bytes([1,1,42])),42)

    def test_writer_scale_raw_gap_abort_and_empty(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as root:
            w=SessionWriter(root,'P01',DESC,P,self.cfg())
            w.submit('packet',packet(0));w.submit('packet',packet(6))
            summary=w.close(False,'中止')
            self.assertFalse(summary['complete']);self.assertTrue(summary['device_time_gaps'])
            with h5py.File(w.path/'data.h5') as h:
                np.testing.assert_equal(h['counts'][:],1);np.testing.assert_equal(h['eeg'][:],.25)
                self.assertEqual(h['device_time_ms'][5],6)
            with self.assertRaises(ValueError):load_session(w.path)
            w2=SessionWriter(root,'P01',DESC,P,self.cfg());self.assertNotEqual(w.path,w2.path)
            self.assertFalse(w2.close(True)['complete'])

    def test_writer_io_failure_and_invalid_config(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as root:
            with self.assertRaises(ValueError):SessionWriter(root,'P',dict(DESC,count_scale=float('nan')),P,self.cfg())
            w=SessionWriter(root,'P',DESC,P,self.cfg());bad=packet();bad.counts=np.zeros((7,5),np.int32)
            w.submit('packet',bad)
            self.assertTrue(w.finished.wait(3))
            with self.assertRaises(RuntimeError):w.close(True)
            self.assertTrue((w.path/'failure.json').is_file())

    def test_full_queue_stops_explicitly(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as root:
            w=SessionWriter(root,'P',DESC,P,self.cfg())
            with patch.object(w.queue,'put_nowait',side_effect=queue.Full):
                with self.assertRaisesRegex(RuntimeError,'队列'):w.submit('event',{})
            with self.assertRaises(RuntimeError):w.close(True)
            self.assertFalse(json.loads((w.path/'session.json').read_text(encoding='utf-8'))['complete'])

    def test_trigger_ack_is_not_echo_and_state_nonblocking(self):
        class Box:
            def send_event(self,code):pass
        entered=threading.Event();release=threading.Event()
        def wait(*args):entered.set();release.wait(2);return None
        t=TriggerManager(lambda x:None,wait,lambda:None);t.mode='serial';t.box=Box()
        errors=[]
        def run():
            try:t.send(1,require_echo=True)
            except ValueError as e:errors.append(str(e))
        thread=threading.Thread(target=run);thread.start();self.assertTrue(entered.wait(1))
        begin=time.monotonic();self.assertFalse(t.state()['echo_verified']);self.assertLess(time.monotonic()-begin,.1)
        release.set();thread.join(3);self.assertTrue(errors);self.assertTrue(t.last['acknowledged'])
        with self.assertRaises(ValueError):TriggerManager(None,None,lambda:None).connect('simulated')

    def test_failed_transport_change_never_falls_back(self):
        c=Controller(ROOT)
        c.trigger_connect('software')
        self.assertTrue(c.trigger_selected)
        with self.assertRaises(ValueError):c.trigger_connect('serial','invalid')
        self.assertFalse(c.trigger_selected)
        self.assertEqual(c.trigger.state()['mode'],'serial')
        self.assertFalse(c.trigger.state()['connected'])

    def test_missing_echo_continues_and_journals_unverified_trials(self):
        c=Controller(ROOT)
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as root:
            c.sessions_root=Path(root);c.connect('synthetic',{'channels':8})
            try:
                time.sleep(.1);c.trigger_connect('simulated');c.trigger_test()
                with patch.object(c.device,'emit_marker',return_value=None) as send:
                    c.start('MISSING_ECHO',{**P,'task_seconds':.3})
                    e=c.experiment;e.thread.join(8)
                self.assertFalse(e.thread.is_alive())
                self.assertTrue(c.last_summary['complete'],c.last_summary)
                self.assertFalse(c.last_summary['training_compatible'])
                self.assertEqual(c.last_summary['unverified_trial_ids'],[1,2])
                self.assertTrue(c.last_summary['marking_warning_count'])
                self.assertFalse(c.error)
                self.assertTrue(c.warning)
                rows=[json.loads(x) for x in (Path(c.last_summary['path'])/'events.jsonl').read_text(encoding='utf-8').splitlines()]
                self.assertTrue(any(e.get('acknowledged') and not e.get('echo_verified') for e in rows))
                marks=[r for r in rows if r['kind'] in ('session_start','prepare','cue','task_end','rest','session_end')]
                # All sends happen while previous echoes are still absent;
                # waiting for two seconds must not block the send worker.
                self.assertEqual(len(marks),10)
                self.assertEqual(send.call_count,10)
                self.assertTrue(all(r.get('acknowledged') for r in marks))
                self.assertTrue(all(not r.get('echo_verified') for r in marks))
                with self.assertRaises(ValueError):load_session(c.last_summary['path'])
            finally:c.disconnect()

    def test_trigger_warnings_do_not_cancel_but_eeg_down_does(self):
        c=Controller(ROOT)
        exp=Experiment(c,P);c.experiment=exp
        for kind in ('down','stalled','frame_error','wireless_event_unaligned'):
            c.status_event(kind,{'source':'triggerbox_acp','error':'injected trigger fault'})
            self.assertFalse(exp.stop.is_set())
            self.assertFalse(c.error)
        self.assertTrue(c.warning)
        c.status_event('down',{'error':'injected EEG disconnect'})
        self.assertTrue(exp.stop.is_set())
        self.assertEqual(c.error,'injected EEG disconnect')

    def test_same_code_echo_after_next_send_cannot_verify_earlier_mark(self):
        c=Controller(ROOT)
        after=time.perf_counter_ns()
        c.echoes.append({'event_id':9,'mono_time_ns':after+300,'device_time_ms':42,'wall_time_ns':1})
        self.assertIsNone(c.wait_echo(9,after,0,before=after+200))
        self.assertEqual(c.wait_echo(9,after+200,0)['device_time_ms'],42)

    def test_expired_marker_is_not_sent_and_recording_is_not_cancelled(self):
        c=Controller(ROOT)
        c.trigger.mode='serial'
        c.marks.put({'kind':'cue','event_id':1,'trial_id':1,'label':0,
                     'mono_time_ns':time.perf_counter_ns()-300_000_000,'device_time_ms':123})
        c.marks.join();c.confirmations.join()
        self.assertFalse(c.error)
        self.assertTrue(c.warning)
        self.assertFalse(c.trigger.last)

    def test_six_classes_short_run_completes_with_delays_and_missing_echoes(self):
        c=Controller(ROOT)
        timers=[]
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as root:
            c.sessions_root=Path(root);c.connect('synthetic',{'channels':8})
            try:
                time.sleep(.1);c.trigger_connect('simulated');c.trigger_test()
                sent=[];end_count=[0]
                def emit(code):
                    sent.append(code)
                    if code==9:
                        end_count[0]+=1
                        if end_count[0]%7==0:return
                    if code==2:return
                    delay=.303 if code==1 else 0
                    hardware_ms=c.device.last_device_ms
                    def deliver():
                        c.device.on_packet(Packet('event',time.time_ns(),time.perf_counter_ns(),hardware_ms,
                            code=code,module=13,payload=bytes([1,1,code]),
                            detail={'format':'simulated_delayed_echo','physical_hardware':False}))
                    if delay:
                        timer=threading.Timer(delay,deliver);timer.daemon=True
                        timers.append(timer);timer.start()
                    else:deliver()
                p={**P,'tasks':[{'text':str(i),'action':'custom'} for i in range(6)],'loops':2,
                   'prepare_seconds':.1,'task_seconds':.1,'rest_seconds':0,'block_rest_seconds':0}
                with patch.object(c.device,'emit_marker',side_effect=emit):
                    c.start('CONTINUITY_SIM',p);exp=c.experiment
                    c.status_event('wireless_event_unaligned',{'source':'triggerbox_acp','error':'injected unmatched event'})
                    exp.thread.join(8)
                    self.assertFalse(exp.thread.is_alive(),c.state())
                summary=c.last_summary
                self.assertTrue(summary['complete'],summary)
                self.assertFalse(c.error)
                self.assertEqual(len(sent),51)
                expected_unverified=sorted(set(range(2,13,6))|set(range(7,13,7)))
                self.assertEqual(summary['unverified_trial_ids'],expected_unverified)
                self.assertIn('unverified_or_missing_labels',summary['reasons'])
                self.assertFalse(summary['training_compatible'])
                with h5py.File(Path(summary['path'])/'data.h5') as h:
                    rows=[json.loads(x) for x in h['events_json'][:]]
                    stamps=h['device_time_ms'][:]
                cues=[r for r in rows if r['kind']=='cue']
                ends=[r for r in rows if r['kind']=='task_end']
                self.assertEqual([r['trial_id'] for r in cues],list(range(1,13)))
                self.assertEqual(len(ends),12)
                self.assertEqual(sum(not r.get('echo_verified') for r in cues),2)
                self.assertEqual(sum(not r.get('echo_verified') for r in ends),1)
                self.assertTrue(all(r.get('echo_verified') for r in cues if r['label']==0))
                self.assertTrue(np.all(np.diff(stamps)==1))
                self.continuity_result={'classes':6,'rounds':2,'trials_completed':12,
                    'delayed_cues_verified':2,'missing_cues_flagged':2,'missing_task_ends_flagged':1,
                    'markers_sent':len(sent),'recording_completed':True,'continuous_eeg':True,
                    'training_compatible':False,'unverified_trial_ids':expected_unverified,
                    'simulation_only':True,'real_devices_contacted':False}
            finally:
                for timer in timers:timer.cancel()
                if c.experiment:
                    c.abort('模拟测试结束');c.experiment.thread.join(8)
                c.disconnect()

    def test_packet_stall_and_subject_lease_abort(self):
        for failure in ('stall','subject'):
            c=Controller(ROOT)
            with tempfile.TemporaryDirectory(dir=ROOT/'tests') as root:
                c.sessions_root=Path(root);c.connect('synthetic',{'channels':8})
                try:
                    time.sleep(.1);c.trigger_connect('software')
                    c.start('FAULT',{**P,'task_seconds':3},require_subject=failure=='subject');e=c.experiment
                    if failure=='stall':c.device.stop.set()
                    e.thread.join(8)
                    self.assertFalse(c.last_summary['complete'])
                    self.assertTrue(c.last_summary['error'])
                finally:c.disconnect()

    def test_writer_failure_does_not_deadlock_marker_or_finalize(self):
        c=Controller(ROOT)
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as root:
            c.sessions_root=Path(root);c.connect('synthetic',{'channels':8})
            try:
                time.sleep(.1);c.trigger_connect('simulated');c.trigger_test()
                c.start('DISK_FAIL',{**P,'task_seconds':3});e=c.experiment;writer=c.writer
                with patch.object(writer.queue,'put_nowait',side_effect=queue.Full):
                    e.thread.join(8)
                self.assertFalse(e.thread.is_alive())
                self.assertTrue(c.marker.is_alive())
                self.assertFalse(c.last_summary['complete'])
            finally:c.disconnect()

    def test_writer_does_not_align_into_time_gap(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as root:
            w=SessionWriter(root,'GAP',DESC,P,self.cfg())
            w.submit('packet',packet(0));w.submit('packet',packet(10))
            w.submit('event',{'kind':'cue','device_time_ms':7,'echo_verified':True})
            w.close(False)
            self.assertIsNone(w.events[0]['sample_index'])
            self.assertEqual(w.events[0]['alignment'],'device_clock_gap')

    def test_http_same_origin_token_and_loopback(self):
        server=make_server(ROOT,0)
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            state=json.load(urlopen(base+'/api/state'))
            self.assertEqual(server.server_address[0],'127.0.0.1')
            self.assertIsNone(state['device'])
            headers={'Content-Type':'application/json','Origin':base,'X-Studio-Token':state['token']}
            for bad in ({},dict(headers,Origin='http://example.com'),dict(headers,Host='example.com')):
                with self.assertRaises(HTTPError) as error:urlopen(Request(base+'/api/disconnect',b'{}',headers=bad))
                self.assertEqual(error.exception.code,403)
            response=json.load(urlopen(Request(base+'/api/disconnect',b'{}',headers=headers)))
            self.assertTrue(response['ok'])
        finally:server.shutdown();server.server_close();worker.join(3)

    def test_controller_closed_loop_and_abort(self):
        c=Controller(ROOT)
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as root:
            c.sessions_root=Path(root)
            c.connect('synthetic',{'channels':8})
            try:
                time.sleep(.15);c.trigger_connect('simulated');self.assertTrue(c.trigger_test()['echo_verified'])
                run_protocol={**P,'condition':'闭眼 + 自定义环境','condition_notes':'这是人工模拟验收'}
                c.start('P_SIM',run_protocol);e=c.experiment
                run_protocol['condition']='开始后修改输入字典'
                with self.assertRaises(ValueError):c.impedance(True)
                e.thread.join(8);self.assertFalse(e.thread.is_alive())
                self.assertTrue(c.last_summary['complete'],c.last_summary)
                r=c.review(Path(c.last_summary['path']).name)
                expected={'condition':'闭眼 + 自定义环境','notes':'这是人工模拟验收'}
                self.assertEqual(r['metadata']['recording_context'],expected)
                self.assertEqual(r['metadata']['config']['session_metadata']['recording_context'],expected)
                self.assertEqual(c.sessions()[0]['condition'],expected['condition'])
                events=r['events'];self.assertEqual(len([e for e in events if e['kind']=='cue']),2)
                cues=[e for e in events if e['kind']=='cue'];self.assertTrue(all(e['echo_verified'] for e in cues))
                self.assertTrue(all(e['marking']=='simulated_echo' for e in cues))
                self.assertEqual(c.subject_state['revision'],6)
                c.start('P_ABORT',P);e=c.experiment;c.abort();e.thread.join(8)
                self.assertFalse(c.last_summary['complete'])
            finally:c.disconnect()

    def test_compatible_training_and_clock_gap_defense(self):
        cfg=self.cfg();cfg['device']['source']='neuracle_acp';validate_config(cfg)
        with self.assertRaises(ValueError):connect(None,cfg)
        cfg['paradigm_id']='a';other=copy.deepcopy(cfg);other['paradigm_id']='b'
        with self.assertRaises(ValueError):require_compatible(cfg,other)
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as root:
            p={**P,'task_seconds':2.5};w=SessionWriter(root,'P_TRAIN',DESC,p,cfg)
            w.submit('packet',packet(0,6000))
            for kind,label,trial,t in [('cue',0,1,100),('task_end',0,1,2600),('cue',1,2,3000),('task_end',1,2,5500)]:
                w.submit('event',{'kind':kind,'label':label,'trial_id':trial,'device_time_ms':t,'echo_verified':True})
            self.assertTrue(w.close(True)['training_compatible'])
            self.assertEqual(load_session(w.path)['eeg'].shape,(8,6000))
            with h5py.File(w.path/'data.h5','a') as h:h['device_time_ms'][3000:]+=2
            with self.assertRaisesRegex(ValueError,'缺口'):load_session(w.path)


if __name__=='__main__':unittest.main(verbosity=2)
