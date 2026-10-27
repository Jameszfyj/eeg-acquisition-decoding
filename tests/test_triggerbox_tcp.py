"""Test network DCP protocol and actual EEG receive path, using local fixtures only."""
import copy
import json
import os
from pathlib import Path
import socket
import struct
import threading
import unittest
import uuid
from unittest.mock import patch

from handbci.acquisition import collect
from handbci.collection_ui import CollectionController
from handbci.config import load_config
from handbci.data import load_session
from handbci.tcp_simulator import TcpEEGSimulator
from handbci.triggerbox import configure_triggerbox, probe_network_triggerbox, validate_network
from handbci.vendor.trigger_box_tcp import TcpTriggerBoxConnection

ROOT = Path(__file__).resolve().parents[1]


class DCPFixture:
    def __init__(self, sink=None, fault=None):
        self.sink, self.fault = sink, fault
        self.requests, self.errors = [], []
        self.stop = threading.Event()
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        self.sock.listen(2)
        self.sock.settimeout(.1)

    def exact(self, conn, n):
        result = bytearray()
        while len(result) < n and not self.stop.is_set():
            try:
                data = conn.recv(n - len(result))
            except socket.timeout:
                continue
            if not data:
                return None
            result.extend(data)
        return bytes(result) if len(result) == n else None

    def serve(self):
        try:
            while not self.stop.is_set():
                try:
                    conn, _ = self.sock.accept()
                except socket.timeout:
                    continue
                with conn:
                    conn.settimeout(.1)
                    while not self.stop.is_set():
                        header = self.exact(conn, 4)
                        if header is None:
                            break
                        device, function, size = struct.unpack("<BBH", header)
                        data = self.exact(conn, size)
                        if data is None:
                            break
                        self.requests.append((device, function, data))
                        if self.fault == "timeout":
                            self.stop.wait(.4)
                            continue
                        payload = b"TriggerBox.TEST_FIXTURE\x00" if function == 4 else b""
                        if self.fault == "badname":
                            payload = b"OtherDevice"
                        if function == 225 and self.sink:
                            self.sink(data[0])
                        reply_device = 2 if self.fault == "device" else 1
                        reply_function = 5 if self.fault == "function" else function
                        length = 6000 if self.fault == "length" else len(payload)
                        reply = struct.pack("<BBH", reply_device, reply_function, length) + payload
                        for i in range(0, len(reply), 2):
                            conn.sendall(reply[i:i + 2])
                            self.stop.wait(.001)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as exc:
            self.errors.append(repr(exc))

    def __enter__(self):
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(2)
        self.sock.close()
        assert not self.thread.is_alive() and not self.errors, self.errors


class TCPTriggerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "configs/collection_ui_6class.json")
        self.cfg["device"]["port"] = 0
        self.cfg["classes"] = self.cfg["classes"][:2]
        self.cfg["window"].update(seconds=.5, task_offset_seconds=.1)
        self.cfg["collection"].update(loops=1, initial_rest_seconds=.1, prepare_seconds=.05,
                                      task_seconds=.7, rest_seconds=.05, tail_seconds=.1)
        self.work = Path(os.environ.get("HANDBCI_TRIGGER_TEST_ROOT", str(ROOT / "runtime/triggerbox_tcp_tests"))) / uuid.uuid4().hex[:10]

    def test_handshake_is_read_only_and_does_not_require_serial(self):
        with DCPFixture() as service, patch("handbci.triggerbox.require_serial", side_effect=AssertionError("no serial")):
            result = probe_network_triggerbox({"host": "127.0.0.1", "port": service.port})
            self.assertTrue(result["ok"])
            self.assertFalse(result["eeg_verified"])
            self.assertEqual(service.requests, [(1, 4, b"")])

    def test_fragmented_replies_and_single_event_bytes(self):
        with DCPFixture() as service:
            client = TcpTriggerBoxConnection()
            try:
                client.connect("127.0.0.1", service.port)
                client.send_event(42)
                self.assertEqual(service.requests, [(1, 4, b""), (1, 225, b"*")])
            finally:
                client.disconnect()
            self.assertFalse(client.connected)

    def test_web_probe_uses_network_settings_and_only_queries_name(self):
        log = self.work / "probes.jsonl"
        controller = CollectionController(self.work, probe_log=log)
        with DCPFixture() as service:
            result = controller.test_triggerbox(None, {"mode": "network", "host": "127.0.0.1", "port": service.port})
            self.assertTrue(result["ok"])
            self.assertEqual(service.requests, [(1, 4, b"")])
        self.assertFalse(controller.snapshot()["trigger_probe_busy"])
        record = json.loads(log.read_text(encoding="utf-8"))
        self.assertTrue(record["read_only"])
        self.assertTrue(record["ok"])
        self.assertEqual(controller.snapshot()["trigger_probe_result"], record)

    def test_wrong_protocol_bad_identity_oversize_and_timeout_close_socket(self):
        for fault in ("device", "function", "length", "timeout", "badname"):
            with self.subTest(fault=fault), DCPFixture(fault=fault) as service:
                client = TcpTriggerBoxConnection(timeout=.15)
                with self.assertRaises((ValueError, TimeoutError, ConnectionError)):
                    client.connect("127.0.0.1", service.port)
                self.assertFalse(client.connected)

    def test_invalid_event_codes_do_not_write(self):
        with DCPFixture() as service:
            client = TcpTriggerBoxConnection()
            client.connect("127.0.0.1", service.port)
            try:
                for code in (True, -1, 256, 1.5):
                    with self.assertRaises(ValueError):
                        client.send_event(code)
                self.assertEqual(service.requests, [(1, 4, b"")])
            finally:
                client.disconnect()

    def test_reserved_ports_public_ips_and_synthetic_source_are_rejected(self):
        for host, port in (("8.8.8.8", 4321), ("0.0.0.0", 4321), ("192.168.1.242", 4097),
                           ("127.0.0.1", 8712), ("192.168.1.242", True), ("bad", 4321)):
            with self.subTest(host=host, port=port), self.assertRaises(ValueError):
                validate_network(host, port)
        with self.assertRaisesRegex(ValueError, "模拟源"):
            configure_triggerbox(self.cfg, {"mode": "network", "host": "127.0.0.1", "port": 4321})

    def test_network_settings_survive_web_start_and_can_be_cleared(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["device"].update(source="neuracle", trigger_port="COM7")
        settings = {"mode": "network", "host": "192.168.1.242", "port": 4321}
        controller = CollectionController(self.work)
        with patch.object(controller, "_run") as run:
            controller.start({"config": cfg, "triggerbox": settings})
            controller.worker.join(2)
        saved = run.call_args.args[0]
        self.assertEqual(saved["device"]["trigger_port"], "")
        self.assertEqual(saved["device"]["trigger_connection"]["host"], settings["host"])
        self.assertEqual(configure_triggerbox(saved)["mode"], "network")
        configure_triggerbox(saved, {"mode": "software"})
        self.assertNotIn("trigger_connection", saved["device"])

    def test_network_events_reach_real_tcp_eeg_and_saved_metadata(self):
        with TcpEEGSimulator(self.cfg, progress=lambda _: None) as eeg:
            with DCPFixture(sink=eeg.send_event) as trigger:
                cfg = copy.deepcopy(self.cfg)
                cfg["device"].update(source="neuracle", port=eeg.port)
                cfg["session_metadata"] = {"participant_id": "NETWORK_SYNTHETIC_FIXTURE", "synthetic_test_fixture": True}
                configure_triggerbox(cfg, {"mode": "network", "host": "127.0.0.1", "port": trigger.port})
                path = collect(cfg, self.work, on_cue=lambda _: None, on_phase=lambda _: {})
        record = load_session(path)
        info = record["config"]["session_metadata"]["triggerbox"]
        self.assertEqual(info["transport"], "tcp")
        self.assertEqual(info["verified_events"], 8)
        self.assertEqual(info["status"], "completed")
        for event in record["events"]:
            self.assertEqual(event["marking"], "hardware_trigger")
            self.assertEqual(event["trigger_transport"], "tcp")
            self.assertEqual(event["trigger_endpoint"], info["endpoint"])
            self.assertEqual(record["trigger"][event["sample_index"]], event["event_id"])
        self.assertEqual(json.loads((path / "collection_report.json").read_text(encoding="utf-8"))["triggerbox"], info)

    def test_network_ack_without_eeg_event_stops_without_fallback(self):
        with TcpEEGSimulator(self.cfg, progress=lambda _: None) as eeg, DCPFixture() as trigger:
            cfg = copy.deepcopy(self.cfg)
            cfg["device"].update(source="neuracle", port=eeg.port)
            cfg["session_metadata"] = {"participant_id": "NETWORK_SYNTHETIC_FIXTURE", "synthetic_test_fixture": True}
            configure_triggerbox(cfg, {"mode": "network", "host": "127.0.0.1", "port": trigger.port})
            with self.assertRaisesRegex(TimeoutError, "不自动退回软件标记"):
                collect(cfg, self.work, on_cue=lambda _: None, on_phase=lambda _: {})
        path = next(self.work.iterdir())
        metadata = json.loads((path / "session.json").read_text(encoding="utf-8"))
        self.assertFalse(metadata["complete"])
        self.assertEqual(metadata["config"]["session_metadata"]["triggerbox"]["verified_events"], 0)
        self.assertEqual((path / "events.jsonl").read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
