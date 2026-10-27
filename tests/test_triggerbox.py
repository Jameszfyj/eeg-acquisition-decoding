"""模拟串口协议与真实TCP接收链路；不证明任何实体TriggerBox兼容。"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import threading
import unittest
import urllib.error
import urllib.request
import uuid
from unittest.mock import patch

import h5py

from handbci.acquisition import collect
from handbci.collection_ui import CollectionController, create_ui_server
from handbci.config import load_config
from handbci.data import load_session
from handbci.tcp_simulator import TcpEEGSimulator
from handbci.triggerbox import configure_triggerbox, probe_triggerbox, serial_ports

ROOT = Path(__file__).resolve().parents[1]


class FakeSerial:
    def __init__(self, sink=None, *, reply=True, short_write=False):
        self.sink, self.reply, self.short_write = sink, reply, short_write
        self.is_open, self.buffer, self.commands = True, bytearray(), []

    def reset_input_buffer(self):
        self.buffer.clear()

    def reset_output_buffer(self):
        pass

    def write(self, message):
        self.commands.append(message)
        if self.short_write:
            return len(message) - 1
        assert message[:1] == b"\x01" and len(message) == 4 + message[2] + 256 * message[3]
        function = message[1]
        payload = b"TEST_FIXTURE_NOT_HARDWARE\x00" if function == 4 else b""
        if function == 225 and self.sink:
            self.sink(message[4])
        if self.reply:
            self.buffer.extend(bytes([1, function, len(payload), 0]) + payload)
        return len(message)

    def flush(self):
        pass

    def read(self, size):
        # Fragment replies into at most two bytes, exercising the real driver.
        chunk = bytes(self.buffer[:min(size, 2)])
        del self.buffer[:len(chunk)]
        return chunk

    def close(self):
        self.is_open = False


class TriggerBoxTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "configs/collection_ui_6class.json")
        self.cfg["classes"] = self.cfg["classes"][:2]
        self.cfg["device"]["port"] = 0
        self.cfg["window"].update(seconds=0.5, task_offset_seconds=0.1)
        self.cfg["collection"].update(loops=1, initial_rest_seconds=0.05, prepare_seconds=0.05,
                                      task_seconds=0.7, rest_seconds=0.05, tail_seconds=0.1)
        self.work = Path(os.environ.get("HANDBCI_TRIGGER_TEST_ROOT", str(ROOT / "runtime/triggerbox_tests"))) / uuid.uuid4().hex[:10]

    def test_probe_checks_identity_without_sending_event_and_closes_port(self):
        fake = FakeSerial()
        with patch("serial.Serial", return_value=fake) as constructor:
            result = probe_triggerbox(" com7 ")
        self.assertTrue(result["ok"])
        self.assertFalse(result["eeg_verified"])
        self.assertEqual(result["port"], "COM7")
        self.assertEqual([m[1] for m in fake.commands], [4])
        constructor.assert_called_once_with(port="COM7", baudrate=115200, timeout=1.0, write_timeout=1.0)
        self.assertFalse(fake.is_open)

    def test_probe_timeout_and_partial_write_close_port(self):
        for fake in (FakeSerial(reply=False), FakeSerial(short_write=True)):
            with self.subTest(fake=fake), patch("serial.Serial", return_value=fake):
                with self.assertRaisesRegex(ValueError, "连接失败"):
                    probe_triggerbox("COM7")
                self.assertFalse(fake.is_open)

    def test_missing_dependency_keeps_enumeration_readable(self):
        with patch("handbci.triggerbox.require_serial", side_effect=ValueError("缺少pyserial")):
            result = serial_ports()
            self.assertFalse(result["available"])
            self.assertIn("pyserial", result["message"])
        # Software/synthetic operation does not import or require serial.
        with patch("handbci.triggerbox.require_serial", side_effect=AssertionError("must not import")):
            self.assertEqual(configure_triggerbox(self.cfg)["mode"], "software")

    def test_configuration_rejects_missing_invalid_or_synthetic_hardware_port(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["device"]["source"] = "neuracle"
        for port in ("", "8712", "COM0", "COM3;anything", None):
            with self.subTest(port=port), self.assertRaises(ValueError):
                configure_triggerbox(cfg, {"mode": "hardware", "port": port})
        with self.assertRaisesRegex(ValueError, "模拟源"):
            configure_triggerbox(self.cfg, {"mode": "hardware", "port": "COM7"})
        configure_triggerbox(cfg, {"mode": "hardware", "port": "com7"})
        self.assertEqual(cfg["device"]["trigger_port"], "COM7")
        configure_triggerbox(cfg, {"mode": "software", "port": "COM7"})
        self.assertEqual(cfg["device"]["trigger_port"], "")

    def run_stream(self, emit):
        with TcpEEGSimulator(self.cfg, progress=lambda _: None) as simulator:
            cfg = copy.deepcopy(self.cfg)
            # Exercise the hardware path using injected serial and artificial TCP.
            # All outputs carry an explicit test-fixture identity and stay in test storage.
            cfg["device"].update(source="neuracle", port=simulator.port, trigger_port="COM7")
            cfg["session_metadata"] = {"participant_id": "TRIGGER_TEST_FIXTURE",
                                       "test_fixture": "fake_serial_and_synthetic_tcp_not_real_eeg"}
            sink = simulator.send_event if emit == "correct" else (
                (lambda code: simulator.send_event(9)) if emit == "wrong" else None)
            fake = FakeSerial(sink)
            with patch("serial.Serial", return_value=fake):
                if emit == "correct":
                    path = collect(cfg, self.work, on_cue=lambda _: None, on_phase=lambda _: {})
                else:
                    with self.assertRaisesRegex(TimeoutError, "不自动退回软件标记"):
                        collect(cfg, self.work, on_cue=lambda _: None, on_phase=lambda _: {})
                    path = next(self.work.iterdir())
            self.assertFalse(fake.is_open)
            return path, fake

    def test_hardware_events_align_to_actual_received_trigger_samples(self):
        path, fake = self.run_stream("correct")
        record = load_session(path)
        info = record["config"]["session_metadata"]["triggerbox"]
        self.assertTrue(info["eeg_verified"])
        self.assertEqual(info["status"], "completed")
        self.assertEqual(info["verified_events"], 8)
        self.assertEqual(info["port"], "COM7")
        self.assertIn("NOT_HARDWARE", info["device_name"])
        for event in record["events"]:
            self.assertEqual(event["marking"], "hardware_trigger")
            self.assertEqual(record["trigger"][event["sample_index"]], event["event_id"])
            self.assertLessEqual(event["trigger_send_wall_ns"], event["trigger_ack_wall_ns"])
            self.assertLessEqual(event["trigger_ack_wall_ns"], event["trigger_observed_wall_ns"])
        report = json.loads((path / "collection_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["triggerbox"], info)
        self.assertEqual(report["n_trials"], 2)
        self.assertEqual(len([m for m in fake.commands if m[1] == 225]), 8)
        self.assertIn("trigger_send_wall_ns", (path / "events.csv").read_text(encoding="utf-8-sig"))

    def test_ack_without_eeg_trigger_does_not_complete_or_fallback(self):
        for emit in ("missing", "wrong"):
            with self.subTest(emit=emit):
                self.work = self.work / emit
                path, _ = self.run_stream(emit)
                meta = json.loads((path / "session.json").read_text(encoding="utf-8"))
                self.assertFalse(meta["complete"])
                self.assertEqual(meta["config"]["session_metadata"]["triggerbox"]["verified_events"], 0)
                self.assertEqual((path / "events.jsonl").read_text(encoding="utf-8"), "")
                with h5py.File(path / "data.h5") as data:
                    self.assertGreater(data["eeg"].shape[1], 0)

    def test_controller_excludes_probe_and_collection_concurrency(self):
        controller = CollectionController(self.work)
        entered, release = threading.Event(), threading.Event()
        def probe(port):
            entered.set()
            release.wait(2)
            return {"ok": True}
        with patch("handbci.collection_ui.probe_triggerbox", side_effect=probe):
            thread = threading.Thread(target=controller.test_triggerbox, args=("COM7",))
            thread.start()
            self.assertTrue(entered.wait(1))
            try:
                self.assertTrue(controller.snapshot()["trigger_probe_busy"])
                with self.assertRaises(ValueError):
                    controller.start({"config": self.cfg})
                with self.assertRaises(ValueError):
                    controller.test_triggerbox("COM7")
            finally:
                release.set()
                thread.join(2)
        self.assertFalse(controller.snapshot()["trigger_probe_busy"])
        controller.state["running"] = True
        with self.assertRaises(ValueError):
            controller.test_triggerbox("COM7")

    def test_web_start_applies_selected_hardware_or_clears_old_port(self):
        for mode in ("hardware", "software"):
            with self.subTest(mode=mode):
                cfg = copy.deepcopy(self.cfg)
                cfg["device"].update(source="neuracle", trigger_port="COM8")
                controller = CollectionController(self.work)
                with patch.object(controller, "_run") as run:
                    controller.start({"config": cfg, "triggerbox": {"mode": mode, "port": "com7"}})
                    controller.worker.join(2)
                saved = run.call_args.args[0]
                self.assertEqual(saved["device"]["trigger_port"], "COM7" if mode == "hardware" else "")
                self.assertEqual(controller.snapshot()["triggerbox"]["mode"], mode)

    def test_http_probe_requires_token_and_returns_readable_failure(self):
        controller = CollectionController(self.work)
        server = create_ui_server(controller)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            html = urllib.request.urlopen(url).read().decode()
            token = re.search(r'name="collection-token" content="([^"]+)"', html).group(1)
            ports = json.loads(urllib.request.urlopen(url + "/api/triggerbox/ports").read())
            self.assertIsInstance(ports["ports"], list)
            for supplied, status in (("", 403), (token, 400)):
                req = urllib.request.Request(url + "/api/triggerbox/test", data=b'{"port":"8712"}',
                    headers={"Content-Type": "application/json", "X-Collection-Token": supplied})
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(req)
                self.assertEqual(error.exception.code, status)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)


if __name__ == "__main__":
    unittest.main()
