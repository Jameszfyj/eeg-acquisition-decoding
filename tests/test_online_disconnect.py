import json
from pathlib import Path
import re
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from handbci.continuous_online import ContinuousController, DEMO_ID
from handbci.online_web import create_online_server

ROOT = Path(__file__).resolve().parents[1]


class OnlineDisconnectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.c = ContinuousController(self.root/'runs')

    def test_disconnect_waits_for_record_sealing_and_blocks_reconnect(self):
        released, finalizing = threading.Event(), threading.Event()
        saved = self.root/'sealed.json'
        def run():
            self.c.stop_event.wait(2)
            finalizing.set()
            released.wait(2)
            saved.write_text('{"saved":true}')
            with self.c.condition:
                self.c.state.update(running=False, saved_path=str(saved))
        self.c.worker = threading.Thread(target=run)
        self.c.state['running'] = True
        self.c.worker.start()
        try:
            self.assertEqual(self.c.disconnect_devices()['device_connection']['phase'], 'disconnecting')
            self.assertTrue(finalizing.wait(1))
            self.assertFalse(saved.exists())
            self.assertEqual(self.c.snapshot()['device_connection']['phase'], 'disconnecting')
            for method in (self.c.start, self.c.check_eeg_connection, self.c.check_trigger_connection,
                           self.c.discover_eeg_devices, self.c.discover_trigger_devices, self.c.start_hand_test):
                with self.assertRaisesRegex(ValueError, '断开设备'):
                    method({})
            self.c.disconnect_devices()  # Repeated clicks cannot skip cleanup.
        finally:
            released.set();self.c.worker.join(2)
        state = self.c.snapshot()
        self.assertEqual(state['device_connection']['phase'], 'disconnected')
        self.assertEqual(Path(state['saved_path']), saved)
        self.assertTrue(saved.exists())
        self.c.start(dict(mode='demo', model_id=DEMO_ID))
        self.c.disconnect_devices();self.c.worker.join(5)
        self.assertEqual(self.c.snapshot()['device_connection']['phase'], 'disconnected')
        self.assertTrue((Path(self.c.state['saved_path'])/'report.json').is_file())

    def test_pending_probe_is_cancelled_and_not_claimed_closed_early(self):
        self.c.eeg_probe_busy = True
        self.c.eeg_probe_done.clear()
        self.c.state['eeg_connection_test'] = dict(phase='testing')
        self.c.disconnect_devices()
        self.assertTrue(self.c.eeg_probe_cancel.is_set())
        self.assertEqual(self.c.state['eeg_connection_test']['phase'], 'cancelling')
        self.assertEqual(self.c.snapshot()['device_connection']['phase'], 'disconnecting')
        self.c.eeg_probe_busy = False
        self.assertEqual(self.c.snapshot()['device_connection']['phase'], 'disconnecting')
        self.c.eeg_probe_done.set()
        self.assertEqual(self.c.snapshot()['device_connection']['phase'], 'disconnected')

    def test_http_disconnect_requires_token_and_leaves_workbench_running(self):
        server = create_online_server(self.c, port=0, assets=ROOT/'handbci/continuous_ui')
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval':.01}, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            html = urlopen(base+'/').read().decode('utf-8')
            self.assertIn('id="disconnectDevices"', html)
            token = re.search(r'name="online-token" content="([^"]+)"', html)[1]
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(base+'/api/disconnect', data=b'{}', headers={'Content-Type':'application/json'}))
            self.assertEqual(error.exception.code, 403)
            for _ in range(2):
                response = json.load(urlopen(Request(base+'/api/disconnect', data=b'{}',
                    headers={'Content-Type':'application/json','X-Online-Token':token})))
                self.assertEqual(response['device_connection']['phase'], 'disconnected')
            self.assertFalse(json.load(urlopen(base+'/api/state'))['running'])
            self.assertIn('断开连接', urlopen(base+'/').read().decode('utf-8'))
        finally:
            server.shutdown();server.server_close();thread.join(2)


if __name__ == '__main__':
    unittest.main()
