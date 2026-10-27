"""本地人工数据服务：按Neuracle接收器的二进制格式，经真实TCP连接发送。"""
from __future__ import annotations

import copy
import queue
import socket
import threading
import time
import uuid

import numpy as np

from .config import validate_config, write_json


class TcpEEGSimulator:
    """仅监听loopback；无设备、无串口、无机械手通信。每次启动供一个客户端采集。"""

    def __init__(self, config, *, port=None, fragment_bytes=0, progress=print):
        self.config = validate_config(config)
        if self.config["device"]["source"] != "synthetic_tcp":
            raise ValueError("模拟服务需要source=synthetic_tcp。")
        if len(config["classes"]) > 6:
            raise ValueError("此模拟器最多生成六类人工任务信号。")
        self.host = "127.0.0.1"
        self.port = int(config["device"].get("port", 18712) if port is None else port)
        if not 0 <= self.port <= 65535:
            raise ValueError("端口须在0至65535之间，0表示自动选择空闲端口。")
        if int(fragment_bytes) < 0:
            raise ValueError("分片大小不能为负。")
        self.fragment_bytes = int(fragment_bytes)
        self.progress = progress
        self.fs = float(config["device"]["sample_rate"])
        self.n_eeg = len(config["device"]["channels"])
        self.chunk_samples = max(1, round(0.04 * self.fs))
        self.frame_dtype = np.dtype([(f"eeg_{i}", "<f4") for i in range(self.n_eeg)]
                                    + [("trigger", "<u4")])
        self.stop_event = threading.Event()
        self.connected = threading.Event()
        self._events = queue.Queue()
        self._lock = threading.Lock()
        self._listener = self._client = self._thread = None
        self.error = None
        self.samples_sent = self.bytes_sent = self.connections = 0

    def start(self):
        if self._thread is not None:
            raise RuntimeError("同一个模拟服务对象不能重复启动。")
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            listener.bind((self.host, self.port))
            listener.listen(1)
            listener.settimeout(0.2)
        except OSError as exc:
            listener.close()
            raise OSError(f"无法监听模拟端口{self.host}:{self.port}；端口可能已占用：{exc}") from exc
        self._listener = listener
        self.port = listener.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, name="handbci-tcp-simulator", daemon=True)
        self._thread.start()
        self.progress(f"[模拟设备] 正在监听 {self.host}:{self.port}；"
                      f"{self.n_eeg}路人工EEG + 1路trigger，{self.fs:g} Hz，每包约40 ms。")
        return self

    def send_event(self, code):
        self.raise_if_failed()
        code = int(code)
        if code not in set(range(1, len(self.config["classes"]) + 1)) | {9, 10, 12, 13, 14}:
            raise ValueError("模拟事件码必须为类别码或已定义的采集阶段码。")
        if not self.connected.is_set():
            raise RuntimeError("尚无采集客户端连接模拟端口。")
        # 只决定人工信号与原始trigger；采集端仍须从TCP流观察到该码才记下标签。
        self._events.put(code)

    def _serve(self):
        try:
            while not self.stop_event.is_set():
                try:
                    client, _ = self._listener.accept()
                    break
                except socket.timeout:
                    continue
            else:
                return
            self._client = client
            with client:
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                client.settimeout(1.0)
                with self._lock:
                    self.connections += 1
                self.connected.set()
                self.progress("[模拟设备] 采集端已通过TCP连接，开始连续发送人工信号。")
                self._send_stream(client)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # 采集端结束后关闭连接是正常退出。
        except BaseException as exc:
            if not self.stop_event.is_set():
                self.error = exc
        finally:
            self.connected.clear()
            if self._listener is not None:
                self._listener.close()

    def _send_stream(self, client):
        rng = np.random.default_rng(self.config["collection"].get("seed", 42))
        frequencies = np.linspace(6, 35, len(self.config["classes"]))
        active_label = None
        sample_index = 0
        deadline = time.monotonic()
        while not self.stop_event.is_set():
            try:
                code = self._events.get_nowait()
                active_label = code - 1 if 1 <= code <= len(self.config["classes"]) else None
            except queue.Empty:
                code = 0
            times = np.arange(sample_index, sample_index + self.chunk_samples) / self.fs
            eeg = rng.normal(0, 0.3, (self.n_eeg, self.chunk_samples))
            if active_label is not None:
                spatial = 0.5 + np.roll(np.linspace(0.2, 1.5, self.n_eeg), active_label % self.n_eeg)
                eeg += 4 * spatial[:, None] * np.sin(2 * np.pi * frequencies[active_label] * times)
            frames = np.zeros(self.chunk_samples, dtype=self.frame_dtype)
            for ch in range(self.n_eeg):
                frames[f"eeg_{ch}"] = eeg[ch]
            frames["trigger"][0] = code
            packet = frames.tobytes()
            size = self.fragment_bytes or len(packet)
            for offset in range(0, len(packet), size):
                client.sendall(packet[offset:offset + size])
            with self._lock:
                self.samples_sent += self.chunk_samples
                self.bytes_sent += len(packet)
            sample_index += self.chunk_samples
            deadline += self.chunk_samples / self.fs
            self.stop_event.wait(max(0, deadline - time.monotonic()))

    def raise_if_failed(self):
        if self.error is not None:
            raise RuntimeError(f"本地TCP模拟服务异常：{self.error}") from self.error

    def stats(self):
        with self._lock:
            return {"host": self.host, "port": self.port, "connections": self.connections,
                    "samples_sent": self.samples_sent, "bytes_sent": self.bytes_sent,
                    "packet_samples": self.chunk_samples,
                    "wire_format": f"little-endian: {self.n_eeg} float32 EEG + uint32 trigger per sample"}

    def stop(self):
        self.stop_event.set()
        for sock in (self._client, self._listener):
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                sock.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RuntimeError("TCP模拟服务未能及时退出。")

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def run_tcp_collection_demo(config, output_root, *, port=None, progress=print):
    """一键启动真实端口服务并走现有collect，随后核对收到的原始事件。"""
    from .acquisition import collect
    from .data import load_session, new_session, trials_from_events
    from .vendor.neuracle_recorder import decode_trigger_values

    cfg = copy.deepcopy(validate_config(config))
    with TcpEEGSimulator(cfg, port=port, progress=progress) as server:
        cfg["device"]["port"] = server.port
        root = new_session(output_root, "tcp_collect_" + time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6])
        write_json(root / "config.json", cfg)
        expected = (cfg["collection"]["initial_rest_seconds"] + 1 +
                    cfg["collection"]["loops"] * len(cfg["classes"]) *
                    (cfg["collection"]["task_seconds"] + cfg["collection"]["rest_seconds"]))
        progress(f"[采集端] 使用现有Neuracle TCP接收器；预计约{expected:.0f}秒。")
        progress("所有信号和触发均为人工模拟，无需做实际想象动作。")
        start = time.monotonic()
        session = collect(cfg, root / "records", on_cue=progress, simulated_trigger=server.send_event)
        elapsed = time.monotonic() - start
        server.raise_if_failed()
        record = load_session(session)
        decoded = decode_trigger_values(record["trigger"])
        first = int(record["indices"][0])
        matches = all(0 <= event["sample_index"] - first < len(decoded) and
                      int(decoded[event["sample_index"] - first]) == event["event_id"]
                      for event in record["events"])
        if not matches:
            raise RuntimeError("采集标签与TCP原始trigger不一致。")
        trials = trials_from_events(record["events"], len(cfg["classes"]))
        report = {"source": "synthetic_tcp", "transport": "TCP", "physical_device_tested": False,
                  "output": str(root.resolve()), "session": str(session.resolve()),
                  "server": server.stats(), "sample_rate": cfg["device"]["sample_rate"],
                  "eeg_channels": record["eeg"].shape[0], "samples_received": len(decoded),
                  "signal_duration_seconds": len(decoded) / cfg["device"]["sample_rate"],
                  "wall_elapsed_seconds": elapsed, "n_trials": len(trials),
                  "class_counts": {c["name"]: sum(t["label"] == c["id"] for t in trials) for c in cfg["classes"]},
                  "n_events": len(record["events"]), "raw_trigger_matches_events": matches,
                  "marking": "synthetic_tcp_trigger", "sample_indices_contiguous": True,
                  "notice": "人工信号经真实TCP端口接收；不代表设备联机成功或真实六类脑电可识别。"}
        write_json(root / "ACQUISITION_REPORT.json", report)
        progress(f"采集完成。报告：{root.resolve() / 'ACQUISITION_REPORT.json'}")
        return report


def serve_standalone(config, *, port=None, progress=print):
    """独立端口调试：没有任务同步控制，发送连续背景噪声。Ctrl+C结束。"""
    with TcpEEGSimulator(config, port=port, progress=progress) as server:
        progress("独立模式发送连续人工背景信号；如需六类任务与标记同步，请运行collect-tcp-demo。")
        try:
            while server._thread.is_alive():
                server.raise_if_failed()
                server.stop_event.wait(0.2)
        except KeyboardInterrupt:
            progress("模拟服务已停止。")
        server.raise_if_failed()
        return server.stats()
