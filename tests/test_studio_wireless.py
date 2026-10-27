"""Wireless input tests use fake links, never physical hardware."""
import json
from pathlib import Path
import struct
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from handbci.studio.controller import Controller
from handbci.studio.devices import NeusenAdapter, Packet, event_code
from handbci.studio.storage import SessionWriter
from handbci.studio.vendor import acp
from handbci.studio.wireless_trigger import WirelessTriggerInput
from tests.test_studio import DESC, P, ROOT, packet


class FakeLink:
    def __init__(self, host='', port=4097, on_frame=None, on_status=None, **kwargs):
        self.on_frame, self.on_status = on_frame, on_status
        self.state = 'idle'
        self.code = 0
        self.kind = 'NST3TB0003'
        self.requests = []
        self.device = None
        self.bad_code = False

    def open(self): self.state = 'connected'
    def close(self): self.state = 'closed'
    def do_hello(self): return acp.DeviceInfo(type_code=self.kind)

    def request(self, cmd, builder):
        raw = builder(5)
        self.requests.append(raw)
        if cmd == acp.CMD_SENDSYNCCODE:
            self.code = struct.unpack('<i', raw[9:-4])[0]
            if self.device:
                self.device.clock_pair = (10010, 10, time.perf_counter_ns())
        return acp.Response(1, cmd, 0, 5, 1, 0, 50010,
                            struct.pack('<i', self.code+(1 if self.bad_code else 0)), sync_ts_ms=10)


def device():
    link = FakeLink()
    d = SimpleNamespace(id='neusen_acp', running=True, impedance_active=False,
                        synthetic=False,
                        info={'host':'192.168.1.20','port':4097}, link=link, clock_pair=None)
    link.device = d
    return d


def event(code=240, shared=10, uptime=50010, kind=2):
    return acp.Frame(time.time_ns(), time.perf_counter_ns(), uptime, 1, 13, 1,
                     bytes([1, kind])+struct.pack('<H', code), sync_ts_ms=shared)


class WirelessTests(unittest.TestCase):
    def setUp(self):
        diagnostic = patch.object(Controller, '_save_trigger_check')
        diagnostic.start()
        self.addCleanup(diagnostic.stop)

    def connect_input(self):
        d, packets, faults = device(), [], []
        box = FakeLink()
        w = WirelessTriggerInput(d, packets.append, lambda *x: faults.append(x))
        with patch('handbci.studio.wireless_trigger.acp.Link', return_value=box):
            w.connect('192.168.1.242')
        self.addCleanup(w.close)
        return d, box, w, packets, faults

    def test_shared_counter_is_distinct_from_uptime_and_crc_checked(self):
        seen, errors = [], []
        link = acp.Link('', 0, seen.append, lambda *x: errors.append(x))
        raw = acp.build_frame(acp.TOKEN_CMD, 17, acp.PKG_ALERT,
            bytes([13,1,1])+struct.pack('<II',17,50017), b'\x01\x02\xf0\x00', acp.TAIL_CMD)
        for b in raw:
            link._buf.append(b)
            link._drain(time.time_ns())
        self.assertEqual((seen[0].sync_ts_ms, seen[0].dev_ts_ms), (17,50017))
        bad = bytearray(raw); bad[-4] ^= 1
        link._buf.extend(bad); link._drain(time.time_ns())
        self.assertEqual(len(seen),1)
        self.assertTrue(errors)

    def test_sync_both_devices_then_independent_output_event(self):
        d, box, w, packets, faults = self.connect_input()
        self.assertTrue(w.synced)
        for link in (d.link,box):
            self.assertEqual([r[7] for r in link.requests], [9,8])
            self.assertEqual(link.code,w.sync_code)
        self.assertIsNone(event_code(event().payload))  # Amplifier parser remains TriggerIn-only.
        w._frame(event())
        p = packets[0]
        self.assertEqual((p.kind,p.code,p.device_ms,p.raw_device_ms),('event',240,10010,50010))
        self.assertEqual(p.detail['sync_time_ms'],10)
        self.assertEqual(p.detail['source'],'triggerbox_acp')
        self.assertFalse(faults)

    def test_missing_or_wrong_identity_and_sync_rejected(self):
        for reason in ('identity','sync'):
            d, box = device(), FakeLink()
            if reason == 'identity': box.kind='NSW3BA0064'
            else: box.bad_code=True
            w=WirelessTriggerInput(d,lambda p:None,lambda *x:None)
            with patch('handbci.studio.wireless_trigger.acp.Link',return_value=box):
                with self.assertRaises(ValueError):w.connect('192.168.1.242')
            self.assertFalse(w.synced)
            self.assertEqual(box.state,'closed')

    def test_invalid_clock_stale_eeg_and_event_time_do_not_become_labels(self):
        for failure in ('unsynced','stale','clock','late','stopped'):
            d, box, w, packets, faults = self.connect_input()
            f = event()
            if failure=='unsynced':w.synced=False
            if failure=='stale':d.clock_pair=(10010,10,f.mono_ns-600_000_000)
            if failure=='clock':d.clock_pair=(10011,10,f.mono_ns)
            if failure=='late':f.sync_ts_ms=1000
            if failure=='stopped':d.running=False
            w._frame(f)
            self.assertEqual(packets[-1].kind,'other',failure)
            self.assertIsNone(packets[-1].code)
            self.assertEqual(w.synced, failure != 'unsynced')
            self.assertTrue(faults)
            w.close()

    def test_unrelated_sensor_event_not_recognized(self):
        d, box, w, packets, faults=self.connect_input()
        w._frame(event(kind=3))
        self.assertIsNone(packets[-1].code)
        self.assertEqual(packets[-1].kind,'other')

    def buffered_input(self, old_age_ms=304, event_shared=231277, history_shared=231275):
        """Real adapter buffer with no sockets: clocks from the 2026-10-23 failure."""
        packets, faults = [], []
        d = NeusenAdapter(lambda p: None, lambda *x: None)
        d.running = True
        now = time.perf_counter_ns()
        payload = bytes(32*5*3)
        d._frame(acp.Frame(time.time_ns(), now-int(old_age_ms*1e6), 337275, 1, 2, 1,
                           payload, sync_ts_ms=history_shared))
        d._frame(acp.Frame(time.time_ns(), now-1_000_000, 337585, 2, 2, 1,
                           payload, sync_ts_ms=231585))
        w = WirelessTriggerInput(d, packets.append, lambda *x: faults.append(x))
        w.synced, w.sync_code, w.eeg_offset = True, 123, 106000
        f = event(code=9, shared=event_shared, uptime=528931)
        f.mono_ns = now
        return d, w, f, packets, faults

    def test_delayed_radio_event_uses_exact_buffered_hardware_time(self):
        d, w, f, packets, faults = self.buffered_input()
        w._frame(f)
        self.assertFalse(faults)
        p = packets[-1]
        self.assertEqual((p.kind,p.code,p.device_ms), ('event',9,337277))
        self.assertEqual(p.raw_device_ms,528931)
        self.assertEqual(p.detail['event_vs_latest_eeg_ms'],-308)
        self.assertEqual(p.detail['arrival_delay_ms'],304)
        self.assertEqual(p.detail['matched_eeg_packet_ms'],337275)
        self.assertEqual(p.detail['alignment_basis'],'buffered_device_clock')
        # Reading or replaying an event must never advance/retime the EEG clock.
        self.assertEqual(d.clock_snapshot()[0][0],337585)

    def test_late_event_needs_recent_matching_eeg_not_only_a_larger_threshold(self):
        for kwargs in ({'old_age_ms':1600}, {'event_shared':231370}, {'history_shared':231274},
                       {'event_shared':233277}, {'event_shared':230000}):
            d, w, f, packets, faults = self.buffered_input(**kwargs)
            w._frame(f)
            self.assertEqual(packets[-1].kind,'other',kwargs)
            self.assertIsNone(packets[-1].code,kwargs)
            self.assertTrue(faults,kwargs)
            self.assertTrue(w.synced,kwargs)

    def test_bad_event_does_not_disable_next_valid_event(self):
        d, w, f, packets, faults = self.buffered_input(event_shared=231370)
        w._frame(f)
        self.assertEqual(packets[-1].kind,'other')
        self.assertTrue(w.synced)
        f.sync_ts_ms = 231277
        w._frame(f)
        self.assertEqual((packets[-1].kind,packets[-1].device_ms),('event',337277))
        self.assertTrue(packets[-1].detail['clock_verified'])

    def test_clock_buffer_cleared_on_device_disconnect(self):
        d, w, f, packets, faults = self.buffered_input()
        self.assertEqual(len(d.clock_snapshot()[1]),2)
        d.disconnect()
        self.assertEqual(d.clock_snapshot(),(None,()))

    def test_controller_requires_wireless_input_not_embedded_ack(self):
        c=Controller(ROOT)
        c.device=device()
        box=FakeLink()
        emitted=[True]
        class Serial:
            device_name='TriggerBox.Titing'
            def connect(self, port):pass
            def disconnect(self):pass
            def send_event(self, code):
                if emitted[0]:box.on_frame(event(code))
                return b'\x01\x02'+bytes([code,0])
        def factory(*args,**kwargs):
            box.on_frame=args[2];box.on_status=args[3]
            return box
        with patch('handbci.vendor.trigger_box.TriggerBoxConnection',Serial), \
             patch('handbci.studio.wireless_trigger.acp.Link',side_effect=factory):
            try:
                c.trigger_connect('serial','COM3',wireless_host='192.168.1.242')
                row=c.trigger_test()
                self.assertTrue(row['echo_verified'])
                self.assertEqual(row['observed_device_ms'],10010)
                self.assertEqual(row['verification'],'hardware_wireless_echo')
                emitted[0]=False
                with patch.object(c.trigger,'wait_echo',return_value=None):
                    with self.assertRaisesRegex(ValueError,'无线事件流'):c.trigger_test()
                self.assertTrue(c.trigger.last['acknowledged'])
                self.assertFalse(c.trigger.state()['echo_verified'])
            finally:c.trigger.disconnect()

    def test_raw_counters_persist_for_reanalysis(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as root:
            cfg=json.loads((ROOT/'configs/demo_6class.json').read_text(encoding='utf-8-sig'))
            cfg['device']['channels']=DESC['labels'];cfg['device']['stream_channels']=9
            w=SessionWriter(root,'RADIO',DESC,P,cfg)
            p=packet(10000)
            p.detail={'sync_time_ms':0}
            w.submit('packet',p)
            w.submit('event',{'kind':'device_event','event_id':240,'device_time_ms':10002,
                'raw_device_time_ms':50002,'detail':{'sync_time_ms':2,'source':'triggerbox_acp'}})
            w.close(False)
            with h5py.File(w.path/'data.h5') as h:
                np.testing.assert_array_equal(h['shared_hardware_time_ms'][:],np.arange(5))
                self.assertEqual(h['trigger'][2],240)
                e=json.loads(h['events_json'][0])
                self.assertEqual(e['raw_device_time_ms'],50002)
                self.assertEqual(e['sample_index'],2)


class DiagnosticTests(unittest.TestCase):
    def test_pass_and_failure_saved_without_http_token(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as root:
            c=Controller(root)
            with patch.object(c.trigger,'test',return_value={'echo_verified':True}):
                c.trigger_test()
            with patch.object(c.trigger,'test',side_effect=ValueError('无线事件缺失')):
                with self.assertRaises(ValueError):c.trigger_test()
            raw=(Path(root)/'runtime/studio/diagnostics/trigger_checks.jsonl').read_text(encoding='utf-8')
            rows=[json.loads(line) for line in raw.splitlines()]
            self.assertEqual([r['error'] for r in rows],['','无线事件缺失'])
            self.assertNotIn('token',raw)


if __name__=='__main__':unittest.main(verbosity=2)
