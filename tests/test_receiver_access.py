"""Endpoint ownership tests; only ephemeral loopback sockets, no hardware access."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np

from handbci import acquisition, receiver_access, runtime, stream_diagnostics
from handbci.config import load_config
from handbci.receiver_access import ReceiverLease
from handbci.tcp_simulator import TcpEEGSimulator

ROOT = Path(__file__).resolve().parents[1]


class ReceiverAccessTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory(prefix="receiver_access_", dir=ROOT)
        self.addCleanup(self.work.cleanup)
        self.folder = Path(self.work.name)
        self.lock_root = self.folder / "locks"
        self.root_patch = patch.object(receiver_access, "ROOT", self.lock_root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]

    def lease(self, host="127.0.0.1"):
        return ReceiverLease(host, self.port)

    def test_same_process_alias_and_stale_metadata(self):
        with self.lease():
            with self.assertRaisesRegex(ValueError, "已有"):
                self.lease("localhost").acquire()
        # A persistent lock file is metadata, not evidence of a live owner.
        self.assertEqual(len(list(self.lock_root.glob("*.lock"))), 1)
        with self.lease():
            pass

    def test_exception_and_failed_check_release_ownership(self):
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            with self.lease():
                raise RuntimeError("simulated")
        with patch.object(receiver_access, "existing_clients", side_effect=RuntimeError("table failed")):
            with self.assertRaisesRegex(RuntimeError, "table failed"):
                self.lease().acquire()
        with self.lease():
            pass

    def test_other_process_blocked_and_normal_or_forced_exit_releases(self):
        source = """import sys
from handbci.receiver_access import ReceiverLease
with ReceiverLease('127.0.0.1', int(sys.argv[1]), lock_root=sys.argv[2]):
    print('READY', flush=True)
    sys.stdin.readline()
"""
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        for force in (False, True):
            with self.subTest(force=force):
                child = subprocess.Popen([sys.executable, "-B", "-u", "-c", source,
                                          str(self.port), str(self.lock_root)], cwd=ROOT,
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True, env=env,
                                         creationflags=flags)
                try:
                    self.assertEqual(child.stdout.readline().strip(), "READY")
                    with self.assertRaisesRegex(ValueError, "另一进程"):
                        self.lease().acquire()
                    if force:
                        child.terminate()  # Only this test's child process.
                        child.communicate(timeout=10)
                    else:
                        _, stderr = child.communicate("finish\n", timeout=10)
                        self.assertEqual(child.returncode, 0, stderr)
                    with self.lease():
                        pass
                finally:
                    if child.poll() is None:
                        child.kill()
                        child.communicate(timeout=10)

    @unittest.skipUnless(os.name == "nt", "Windows endpoint table integration")
    def test_existing_real_tcp_client_rejected_without_opening_second_connection(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", self.port))
            listener.listen(2)
            with socket.create_connection(("127.0.0.1", self.port)) as client:
                with listener.accept()[0]:
                    with self.assertRaisesRegex(ValueError, "PID"):
                        self.lease().acquire()
                    listener.settimeout(.1)
                    with self.assertRaises(socket.timeout):
                        listener.accept()
            with self.lease():
                pass

    def config(self):
        cfg = load_config(ROOT / "configs/collection_ui_neuracle_ch32.json")
        cfg["device"].update(source="neuracle", host="127.0.0.1", port=self.port)
        cfg["device"].pop("trigger_connection", None)
        cfg["device"]["trigger_port"] = ""
        return cfg

    def test_busy_collection_refuses_before_tcp_or_trigger_open(self):
        cfg = self.config()
        with self.lease(), patch.object(acquisition, "connect") as connect, \
                patch.object(acquisition, "open_network_triggerbox") as trigger:
            with self.assertRaisesRegex(ValueError, "已有"):
                acquisition.collect(cfg, self.folder / "records", on_cue=lambda *_: None)
        connect.assert_not_called()
        trigger.assert_not_called()
        self.assertFalse(list((self.folder / "records").rglob("data.h5")))

    def test_collection_connection_failure_releases_ownership(self):
        with patch.object(acquisition, "connect", side_effect=ConnectionRefusedError("fixture refusal")):
            with self.assertRaisesRegex(ConnectionRefusedError, "fixture refusal"):
                acquisition.collect(self.config(), self.folder / "records", on_cue=lambda *_: None)
        with self.lease():
            pass
        saved = list((self.folder / "records").rglob("session.json"))
        self.assertEqual(len(saved), 1)
        self.assertFalse(json.loads(saved[0].read_text(encoding="utf-8"))["complete"])

    def test_guarded_32_channel_collection_finishes_and_releases(self):
        cfg = self.config()
        cfg["classes"] = cfg["classes"][:2]
        cfg["window"].update(seconds=.5, task_offset_seconds=.1)
        cfg["collection"].update(loops=1, initial_rest_seconds=.05, prepare_seconds=.05,
                                  task_seconds=.7, rest_seconds=.05, tail_seconds=.1)
        cfg.setdefault("session_metadata", {})["test_origin"] = "loopback_fixture_not_hardware"
        cfg["device"].update(source="synthetic_tcp", unit="synthetic_uV")
        server = TcpEEGSimulator(cfg, port=self.port, progress=lambda *_: None).start()
        try:
            # Exercise the physical-source guard, but the only endpoint is our
            # explicitly identified ephemeral loopback fixture.
            cfg["device"]["source"] = "neuracle"
            saved = acquisition.collect(cfg, self.folder / "records", on_cue=lambda *_: None)
        finally:
            server.stop()
        with h5py.File(saved / "data.h5") as data:
            count = data["eeg"].shape[1]
            self.assertEqual(data["eeg"].shape[0], 32)
            self.assertGreater(count, 1000)
            self.assertTrue(np.all(np.diff(data["source_indices"][:]) == 1))
            self.assertLessEqual(server.samples_sent - count, server.chunk_samples)
            self.assertGreaterEqual(server.samples_sent - count, 0)
        self.assertTrue(json.loads((saved / "session.json").read_text(encoding="utf-8"))["complete"])
        with self.lease():
            pass

    def test_busy_live_and_raw_diagnostic_refuse_before_work(self):
        with self.lease(), patch.object(runtime, "_run_live") as live, \
                patch.object(stream_diagnostics, "_capture_stream") as diagnostic:
            with self.assertRaisesRegex(ValueError, "已有"):
                runtime.run_live("unused-model", self.config(), self.folder)
            with self.assertRaisesRegex(ValueError, "已有"):
                stream_diagnostics.capture_stream(host="127.0.0.1", port=self.port,
                    n_eeg=32, declared_fs=1000, seconds=1, output_root=self.folder)
        live.assert_not_called()
        diagnostic.assert_not_called()

    def test_synthetic_live_bypasses_physical_endpoint_lease(self):
        cfg = self.config()
        cfg["device"]["source"] = "synthetic_tcp"
        with self.lease(), patch.object(runtime, "_run_live", return_value="simulation") as live:
            self.assertEqual(runtime.run_live("unused", cfg, self.folder), "simulation")
        live.assert_called_once()



if __name__ == "__main__":
    unittest.main(verbosity=2)
