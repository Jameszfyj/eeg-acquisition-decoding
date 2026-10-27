"""经真实loopback TCP验证接收、采样索引、原始trigger和采集失败标记。"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import socket
import time
import unittest
import uuid

import numpy as np

from handbci.acquisition import collect, connect, make_recorder
from handbci.config import load_config, validate_config
from handbci.data import build_dataset, load_session
from handbci.tcp_simulator import TcpEEGSimulator, run_tcp_collection_demo
from handbci.vendor.neuracle_recorder import decode_trigger_values

ROOT = Path(__file__).resolve().parents[1]


class TcpSimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(ROOT / "configs/simulated_tcp_6class.json")
        cls.work = ROOT / "runtime/tcp_tests" / uuid.uuid4().hex[:10]

    def test_fragmented_socket_data_and_uint32_trigger_reach_real_receiver(self):
        cfg = copy.deepcopy(self.cfg)
        with TcpEEGSimulator(cfg, port=0, fragment_bytes=113, progress=lambda _: None) as server:
            cfg["device"]["port"] = server.port
            recorder = make_recorder(cfg)
            try:
                connect(recorder, cfg)
                deadline = time.monotonic() + 4
                while recorder.raw_buffer.latest_source_index is None:
                    if time.monotonic() > deadline:
                        self.fail("真实TCP连接未收到数据。")
                    recorder.raw_buffer.wait_for_new_data(0.02)
                before = recorder.raw_buffer.latest_source_index
                server.send_event(3)
                while True:
                    raw, indices, lost = recorder.raw_buffer.read_after_source_index(before)
                    if np.any(decode_trigger_values(raw[-1]) == 3):
                        break
                    if time.monotonic() > deadline:
                        self.fail("模拟trigger没有经过TCP到达接收器。")
                    recorder.raw_buffer.wait_for_new_data(0.02)
                self.assertFalse(lost)
                self.assertEqual(raw.shape[0], 9)
                self.assertTrue(np.isfinite(raw).all())
                np.testing.assert_array_equal(np.diff(indices), 1)
                self.assertEqual(server.stats()["connections"], 1)
                self.assertTrue(server.connected.is_set())
            finally:
                recorder.disconnect()
        # 服务退出后同一地址可重新监听，不留下后台服务。
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", server.port))

    def test_six_class_collection_has_observed_triggers_and_trainable_windows(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["window"].update(seconds=0.5, stride_seconds=0.5, task_offset_seconds=0.1)
        cfg["collection"].update(initial_rest_seconds=1.2, task_seconds=1.0, rest_seconds=0.1)
        report = run_tcp_collection_demo(cfg, self.work / "complete", port=0, progress=lambda _: None)
        record = load_session(report["session"])
        self.assertEqual(report["transport"], "TCP")
        self.assertEqual(record["metadata"]["source"], "synthetic_tcp")
        self.assertEqual(report["n_trials"], 6)
        self.assertEqual(report["n_events"], 12)
        self.assertTrue(report["raw_trigger_matches_events"])
        self.assertEqual({e["marking"] for e in record["events"]}, {"synthetic_tcp_trigger"})
        self.assertEqual(np.count_nonzero(record["trigger"]), 12)
        self.assertLess(abs(report["signal_duration_seconds"] - report["wall_elapsed_seconds"]), 1.0)
        np.testing.assert_array_equal(np.diff(record["indices"]), 1)
        ds = build_dataset([report["session"]])
        self.assertEqual(set(ds.y.tolist()), set(range(6)))

    def test_port_collision_does_not_connect_to_an_existing_service(self):
        with TcpEEGSimulator(self.cfg, port=0, progress=lambda _: None) as first:
            with self.assertRaises(OSError):
                TcpEEGSimulator(self.cfg, port=first.port, progress=lambda _: None).start()
            self.assertEqual(first.stats()["connections"], 0)

    def test_interruption_keeps_incomplete_record_and_releases_port(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["collection"]["initial_rest_seconds"] = 0.1
        with TcpEEGSimulator(cfg, port=0, progress=lambda _: None) as server:
            cfg["device"]["port"] = server.port

            def interrupt_on_task(message):
                if "试次" in message:
                    raise KeyboardInterrupt()

            with self.assertRaises(KeyboardInterrupt):
                collect(cfg, self.work / "interrupted", on_cue=interrupt_on_task,
                        simulated_trigger=server.send_event)
        sessions = list((self.work / "interrupted").iterdir())
        self.assertEqual(len(sessions), 1)
        meta = json.loads((sessions[0] / "session.json").read_text(encoding="utf-8"))
        self.assertFalse(meta["complete"])
        self.assertEqual(meta["source"], "synthetic_tcp")
        with self.assertRaises(ValueError):
            load_session(sessions[0])

    def test_simulation_cannot_be_labelled_as_real_hardware(self):
        for field, value in (("host", "0.0.0.0"), ("unit", "uV")):
            cfg = copy.deepcopy(self.cfg)
            cfg["device"][field] = value
            with self.assertRaises(ValueError):
                validate_config(cfg)
        real = copy.deepcopy(self.cfg)
        real["device"]["source"] = "neuracle"
        with self.assertRaises(ValueError):
            collect(real, self.work / "invalid", simulated_trigger=lambda _: None)
        with self.assertRaises(ValueError):
            collect(self.cfg, self.work / "invalid", trigger_port="COM3")


if __name__ == "__main__":
    unittest.main()
