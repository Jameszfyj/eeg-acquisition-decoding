"""Software marking must never open a TriggerBox; all EEG here is synthetic."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from urllib.request import Request, urlopen

import h5py
import numpy as np

from examples.create_demo_model import build_manifest, bundle_bytes
from handbci.continuous_online import ContinuousController, VotingGate
from handbci.model_bundle import import_bundle
from handbci.prompted_control import OnlineMarkers, PromptedControl, validate_settings
from handbci.studio.controller import Controller
from handbci.studio.devices import Packet
from handbci.studio.server import make_server
from handbci.studio.triggers import TriggerManager

ROOT = Path(__file__).resolve().parents[1]
CLASSES = [dict(id=i, name=name, task_text=name, hand_action=name, action_name=name)
           for i, name in enumerate(('left', 'right'))]
SETTINGS = dict(enabled=True, marking='software', loops=1, seed=1,
                prepare_seconds=.5, rest_seconds=0., trigger_port='invalid',
                trigger_host='stale-trigger-host', trigger_acp_port=4097)


class SoftwareMarkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def test_software_mode_ignores_hardware_fields_and_needs_no_echo(self):
        with patch('handbci.vendor.trigger_box.TriggerBoxConnection') as usb, \
             patch('handbci.studio.wireless_trigger.WirelessTriggerInput') as wifi:
            trigger = TriggerManager(None, Mock(), Mock())
            state = trigger.connect('software', 'invalid', wireless_host='stale')
            self.assertTrue(state['connected'])
            self.assertEqual(state['port'], '')
            self.assertIsNone(state['wireless'])
            self.assertFalse(state['echo_verified'])
            self.assertTrue(trigger.test()['ready'])
            usb.assert_not_called();wifi.assert_not_called()

    def test_offline_software_mark_uses_issue_time_not_schedule_or_worker(self):
        c = Controller(ROOT)
        c.writer = Mock()
        c.latest = Packet('eeg', 1, 1_000_000_000, 5000., counts=np.zeros((8, 5)))
        c.sample_clock.observe(c.latest.device_ms, 5, c.latest.mono_ns)
        try:
            with patch('handbci.studio.controller.time.perf_counter_ns', return_value=1_020_000_000), \
                 patch('handbci.studio.controller.time.time_ns', return_value=7_000_000_000), \
                 patch.object(c.marks, 'put_nowait') as queue_send, patch.object(c.trigger, 'send') as send:
                row = c.mark('cue', 1, trial_id=1, planned_mono_ns=900_000_000)
            self.assertEqual(row['device_time_ms'], 5024.)
            self.assertEqual(row['send_mono_ns'], row['mono_time_ns'])
            self.assertEqual(row['send_wall_ns'], row['wall_time_ns'])
            self.assertEqual(row['marker_time_source'], 'host_command_issue')
            self.assertFalse(row['verification_pending'])
            self.assertFalse(row['echo_verified'])
            c.writer.submit.assert_called_once_with('event', row)
            queue_send.assert_not_called();send.assert_not_called()
        finally:
            c.writer = None

    def test_online_immediate_marking_and_gap_aware_export_without_hardware(self):
        markers = OnlineMarkers(self.path, None, SETTINGS, Mock(), Mock())
        with patch.object(markers.trigger, 'send') as send, patch.object(markers.trigger, 'test') as check:
            markers.connect()
            self.assertIsNone(markers.worker)
            control = PromptedControl(self.path, SETTINGS, CLASSES, VotingGate(2), markers, lambda: 0, 1000, False)
            cue = markers.submit('cue', 1, trial_id=1, estimated_source_index=2)
            markers.submit('task_end', 9, trial_id=1, estimated_source_index=7)
            missing = markers.submit('cue', 2, trial_id=2, estimated_source_index=4)
            self.assertFalse(cue['verification_pending'])
            self.assertEqual(cue['send_mono_ns'], cue['mono_time_ns'])
            self.assertEqual(cue['marker_time_source'], 'host_command_issue')
            self.assertTrue(markers.requests.empty())
            send.assert_not_called();check.assert_not_called()
            journal = [json.loads(line) for line in (self.path/'events.jsonl').read_text(encoding='utf8').splitlines()]
            self.assertEqual(len(journal), 3)
            self.assertTrue(all(not e['verification_pending'] for e in journal))
            with h5py.File(self.path/'raw.h5', 'w') as h:
                indices = np.array([0, 1, 2, 3, 6, 7, 8])
                h.create_dataset('device_ms', data=indices.astype(float))
                h.create_dataset('source_index', data=indices)
                control.finalize(h)
                self.assertEqual(h['trigger'][:].tolist(), [0, 0, 1, 0, 0, 9, 0])
            events = [json.loads(line) for line in (self.path/'events.jsonl').read_text(encoding='utf8').splitlines()]
            lost = next(e for e in events if e.get('trial_id') == 2)
            self.assertIsNone(lost['sample_index'])
            self.assertEqual(lost['alignment'], 'source_index_gap')
            self.assertFalse(missing['echo_verified'])

    def test_online_software_preflight_cannot_require_or_probe_trigger(self):
        c = ContinuousController(self.path/'runs')
        gate = VotingGate(2)
        request = dict(prompted=SETTINGS, output=dict(mode='simulation'))
        normalized = validate_settings(SETTINGS, 'acp', CLASSES, gate, request['output'])
        self.assertEqual(normalized['trigger_port'], '')
        self.assertEqual(normalized['trigger_host'], '')
        with patch.object(c, 'prepare', return_value=({}, {'device': {'host': 'stale-trigger-host'}}, CLASSES, gate, 'acp')), \
             patch('handbci.trigger_probe.run_trigger_check') as probe:
            result = c.check_trigger_connection(request)
        self.assertTrue(result['ready']);self.assertFalse(result['echo_verified'])
        self.assertFalse(c.eeg_probe_busy);self.assertTrue(c.eeg_probe_done.is_set())
        probe.assert_not_called()

    def test_offline_http_software_start_records_and_validates_without_trigger_connect(self):
        server = make_server(ROOT, 0)
        c = server.controller
        c.sessions_root = self.path/'sessions'
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f'http://127.0.0.1:{server.server_port}'
        state = json.load(urlopen(base+'/api/state'))
        headers = {'Content-Type': 'application/json', 'Origin': base, 'X-Studio-Token': state['token']}
        def post(route, body):
            return json.load(urlopen(Request(base+'/api/'+route, json.dumps(body).encode(), headers)))['result']
        try:
            post('connect', dict(adapter='synthetic', options=dict(channels=8)))
            deadline = time.monotonic()+2
            while c.latest is None and time.monotonic() < deadline:
                time.sleep(.01)
            manifest = build_manifest()
            c.device.labels = manifest['config']['device']['channels']
            c.validation_model = import_bundle(bundle_bytes(manifest), 'DEMO_ONNX',
                registry=self.path/'registry.json', import_root=self.path/'models')
            protocol = dict(tasks=[dict(text='left'), dict(text='right')], loops=1, randomize=False,
                prepare_seconds=.2, task_seconds=2.6, rest_seconds=.15,
                initial_rest_seconds=.1, block_rest_seconds=0., tail_seconds=.15, sound=False)
            self.assertFalse(c.trigger_selected)
            with patch.object(c.trigger, 'send') as send, patch.object(c.trigger, 'test') as check:
                result = post('start', dict(participant='DEMO_ONNX', protocol=protocol, marking='software',
                    validation=True, validation_model_id=c.validation_model['model_id']))
                experiment = c.experiment
                deadline = time.monotonic()+15
                while experiment.thread.is_alive() and time.monotonic() < deadline:
                    post('subject-heartbeat', {})
                    experiment.thread.join(.1)
                self.assertFalse(experiment.thread.is_alive())
                self.assertTrue(c.last_summary['complete'], c.last_summary)
                send.assert_not_called();check.assert_not_called()
            folder = Path(result['path'])
            events = [json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf8').splitlines()]
            cues = [e for e in events if e['kind'] == 'cue']
            self.assertEqual(len(cues), 2)
            with h5py.File(folder/'data.h5', 'r') as h:
                for cue in cues:
                    self.assertIsNotNone(cue['sample_index'])
                    self.assertEqual(h['trigger'][cue['sample_index']], cue['event_id'])
                    self.assertFalse(cue['echo_verified'])
                    self.assertEqual(cue['marker_time_source'], 'host_command_issue')
            validation = json.loads((folder/'onnx_validation/summary.json').read_text(encoding='utf8'))
            self.assertEqual([r['status'] for r in validation['trials']], ['completed', 'completed'])
            self.assertTrue(all(r['windows'] > 0 and not r['label_verified'] for r in validation['trials']))
            with self.assertRaisesRegex(ValueError, '当前选择'):
                c.start('DEMO_ONNX', protocol, marking='serial')
            c.trigger.mode = 'serial'
            with self.assertRaisesRegex(ValueError, '回传自检'):
                c.start('DEMO_ONNX', protocol, marking='serial')
        finally:
            c.abort('test cleanup')
            if c.experiment:
                c.experiment.thread.join(5)
            c.disconnect()
            server.shutdown();server.server_close();worker.join(3)


if __name__ == '__main__':
    unittest.main()
