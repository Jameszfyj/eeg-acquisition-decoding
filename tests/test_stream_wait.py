"""Buffered short stalls keep the stream; stale samples never drive the hand."""
import io
import json
import struct
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from handbci.continuous_online import ContinuousController, VotingGate
from handbci.prompted_control import PromptedControl
from handbci.studio.devices import Packet
from handbci.studio.vendor import acp


class BufferedAdapter:
    labels = [f'CH{i}' for i in range(32)]
    labels_verified = True
    unit, count_scale = 'uV', 1.
    settings = dict(fs=1000, channels=32, points=5)

    def __init__(self, packet, status, resume=True, corrupt=False, gap_points=0, wrong_id=False):
        self.packet, self.status, self.resume, self.corrupt = packet, status, resume, corrupt
        self.gap_points, self.wrong_id = gap_points, wrong_id
        self.stop = threading.Event()
        self.disconnect_count = 0

    def connect(self, options):
        self.thread = threading.Thread(target=self.publish, daemon=True)
        self.thread.start()

    def publish(self):
        deadline=time.monotonic()
        for start in range(0, 2000, 5):
            deadline += .005
            if 100 <= start < 100+self.gap_points:
                continue
            if self.corrupt and start == 100:
                frame=bytearray(acp.build_frame(acp.TOKEN_CMD,17,acp.PKG_ALERT,
                    bytes([2,2,start//5])+struct.pack('<II',100000+start,100000+start),bytes(480),acp.TAIL_CMD))
                frame[40]^=1
                self.status('crc_error',dict(candidate_frame_hex=frame.hex(),payload_length=480))
                continue
            if start == 200:
                if not self.corrupt and not self.gap_points and self.stop.wait(1.3 if self.resume else 6):
                    return
            if self.stop.is_set():
                return
            self.packet(Packet('eeg', time.time_ns(), time.perf_counter_ns(),
                100000+start, counts=np.zeros((32,5), np.int32),
                packet_id=(start//5+int(self.wrong_id and start == 100))%256))
            if self.stop.wait(max(0,deadline-time.monotonic())):
                return

    def describe(self):
        return dict(settings=self.settings, labels=self.labels, unit=self.unit)

    def disconnect(self):
        self.disconnect_count += 1
        self.stop.set()
        self.thread.join(2)


class StreamWaitTests(unittest.TestCase):
    def run_stream(self, resume, corrupt=False, gap_points=0, wrong_id=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        controller = ContinuousController(folder/'runs')
        calls, resets, adapters = [], [], []
        class Decoder:
            stream = type('Stream', (), dict(fs=1000.))()
            def process(self, raw, ix):
                calls.append(ix.copy())
                if ix[-1] >= 1799:
                    controller.stop_event.set()
                return None
            def recover_stream(self, first, reason):
                resets.append(first)
        def adapter(*args):
            item = BufferedAdapter(*args, resume=resume, corrupt=corrupt,
                                   gap_points=gap_points, wrong_id=wrong_id)
            adapters.append(item)
            return item
        cfg = dict(device=dict(host='127.0.0.1', port=12345, sample_rate=1000,
                               channels=BufferedAdapter.labels, unit='uV'))
        error = None
        with patch('handbci.studio.devices.NeusenAdapter', side_effect=adapter):
            try:
                controller._acp_loop(cfg, Decoder(), VotingGate(6), io.StringIO(), io.StringIO(), folder)
            except RuntimeError as exc:
                error = str(exc)
        return folder, controller, calls, resets, adapters[0], error

    def test_one_second_pause_waits_then_resumes_without_replaying_backlog(self):
        folder, controller, calls, resets, adapter, error = self.run_stream(True)
        self.assertIsNone(error)
        self.assertEqual(adapter.disconnect_count, 1)  # final explicit stop only
        self.assertEqual(len(resets), 1)
        self.assertGreaterEqual(resets[0], 1500)
        for ix in calls:
            self.assertTrue(ix[-1] < 200 or ix[0] >= resets[0])
        events=[json.loads(s) for s in (folder/'protocol_events.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertIn('stream_wait', [e['kind'] for e in events])
        self.assertIn('stream_resumed', [e['kind'] for e in events])
        self.assertFalse(controller.eeg_fault_active.is_set())
        with h5py.File(folder/'raw_acp.h5') as h:
            self.assertTrue(h.attrs['complete'])
            np.testing.assert_array_equal(np.diff(h['source_index'][:]), 1)

    def test_permanent_stall_still_stops_with_saved_evidence(self):
        folder, controller, calls, resets, adapter, error = self.run_stream(False)
        self.assertIn('连续4秒', error)
        self.assertEqual(resets, [])
        self.assertTrue(controller.eeg_fault_active.is_set())
        with h5py.File(folder/'raw_acp.h5') as h:
            self.assertFalse(h.attrs['complete'])

    def test_isolated_crc_keeps_connection_and_preserves_missing_source_samples(self):
        folder, controller, calls, resets, adapter, error = self.run_stream(True, corrupt=True)
        self.assertIsNone(error)
        self.assertGreaterEqual(len(resets),2)
        self.assertFalse(controller.eeg_fault_active.is_set())
        with h5py.File(folder/'raw_acp.h5') as h:
            indices=h['source_index'][:]
            self.assertTrue(h.attrs['contains_discontinuities'])
            self.assertFalse(np.any((indices>=100)&(indices<105)))
            self.assertEqual(int(np.max(np.diff(indices))),6)

    def test_complete_missing_packets_resume_with_a_real_gap_and_fresh_model_batches(self):
        folder, controller, calls, resets, adapter, error = self.run_stream(True, gap_points=10)
        self.assertIsNone(error)
        self.assertEqual(adapter.disconnect_count, 1)
        self.assertEqual(resets[0], 110)
        self.assertGreater(resets[-1], 110)
        self.assertTrue(any(ix[0] >= resets[-1] for ix in calls))
        for ix in calls:
            np.testing.assert_array_equal(np.diff(ix), 1)
            self.assertTrue(ix[-1] < 100 or ix[0] >= resets[-1])
        self.assertFalse(controller.eeg_fault_active.is_set())
        self.assertEqual(controller.state['gap_count'], 1)
        with h5py.File(folder/'raw_acp.h5') as h:
            indices = h['source_index'][:]
            self.assertTrue(h.attrs['complete'])
            self.assertTrue(h.attrs['contains_discontinuities'])
            self.assertFalse(np.any((indices >= 100) & (indices < 110)))
            self.assertEqual(int(np.max(np.diff(indices))), 11)
            np.testing.assert_array_equal(h['device_ms'][:]-100000, indices)
        events = [json.loads(s) for s in (folder/'protocol_events.jsonl').read_text(encoding='utf-8').splitlines()]
        recovered = next(e['detail'] for e in events if e['kind'] == 'stream_recovery')
        self.assertEqual(recovered['reason'], 'packet_gap')
        self.assertEqual(recovered['missing_samples'], 10)
        self.assertEqual(recovered['repaired_samples'], 0)
        self.assertIn('stream_resumed', [e['kind'] for e in events])

    def test_sequence_only_fault_still_stops_and_preserves_offending_frame(self):
        folder, controller, calls, resets, adapter, error = self.run_stream(True, wrong_id=True)
        self.assertIn('不一致', error)
        self.assertEqual(resets, [])
        self.assertTrue(controller.eeg_fault_active.is_set())
        self.assertTrue(all(ix[-1] < 100 for ix in calls))
        with h5py.File(folder/'raw_acp.h5') as h:
            self.assertFalse(h.attrs['complete'])
            self.assertEqual(h['source_index'][-1], 104)
            self.assertEqual(h['packet_id'][-1], 21)


class PromptStreamWaitTests(unittest.TestCase):
    def control(self, folder):
        class Markers:
            error = ''
            def __init__(self):self.events=[]
            def record(self, event):self.events.append(event)
            def submit(self, kind, event_id, **kwargs):
                return dict(kind=kind, event_id=event_id, verification_pending=False, **kwargs)
        classes=[dict(id=i,hand_action=str(i),action_name=str(i),name=str(i)) for i in range(6)]
        control=PromptedControl(folder,dict(loops=1,seed=3,marking='software',prepare_seconds=2.,rest_seconds=2.),
            classes,VotingGate(6),Markers(),lambda:100,1000,False)
        control.tick(True)
        return control

    def test_interrupted_attempt_is_retained_and_retry_gets_fresh_prompt_and_id(self):
        with tempfile.TemporaryDirectory() as directory:
            control=self.control(Path(directory))
            target, prompt = control.current['label'], control.prompt_id
            control.suspend_stream('test pause')
            self.assertEqual(control.phase,'waiting_stream')
            self.assertFalse(control.trials[0]['complete'])
            self.assertEqual(control.trials[0]['execution_status'],'stream_interrupted')
            self.assertEqual(len(control.plan),7)
            with self.assertRaises(ValueError):
                control.acknowledge(dict(session_id=control.session_id,prompt_id=prompt,visible=True))
            control.resume_stream()
            self.assertEqual(control.current['label'],target)
            self.assertEqual(control.current['trial_id'],2)
            self.assertNotEqual(control.prompt_id,prompt)
            self.assertEqual(control.phase,'awaiting_display')
            self.assertIsNone(control.prepare_mark)

    def test_inflight_action_is_not_reissued_after_receive_pause(self):
        with tempfile.TemporaryDirectory() as directory:
            control=self.control(Path(directory))
            control.phase='executing';control.current['command_sequence']=8
            current=control.current
            control.suspend_stream('test pause');control.resume_stream()
            self.assertEqual(control.phase,'executing')
            self.assertIs(control.current,current)
            self.assertEqual(control.current['command_sequence'],8)
            self.assertEqual(control.trials,[])


if __name__ == '__main__':unittest.main()
