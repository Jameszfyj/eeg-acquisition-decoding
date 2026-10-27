"""Device adapters emit packets; experiment/UI/storage do not speak device protocols."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import socket
import struct
import threading
import time
import os

import numpy as np

from .vendor import acp, layout


@dataclass
class Packet:
    kind: str
    wall_ns: int
    mono_ns: int
    device_ms: float
    counts: np.ndarray | None = None
    packet_id: int = 0
    module: int = 2
    payload: bytes = b""
    code: int | None = None
    detail: dict | None = None
    raw_device_ms: float | None = None


class DeviceAdapter(ABC):
    id = "base"
    title = "设备"
    synthetic = False
    capabilities = {"impedance": False, "events": False}
    count_scale = 1.0
    unit = "device_native"
    timestamp_verified = False

    def __init__(self, on_packet, on_status):
        self.on_packet, self.on_status = on_packet, on_status
        self.connected, self.running = False, False
        self.settings = {"channels": 32, "fs": 1000, "points": 5, "gain": 12}
        self.info = {}
        self.labels = []
        self.positions = []
        self.impedance_active = False

    @classmethod
    def discover(cls):
        return []

    def describe(self):
        return {"id": self.id, "title": self.title, "synthetic": self.synthetic,
                "capabilities": self.capabilities, "connected": self.connected,
                "running": self.running, "settings": dict(self.settings),
                "info": self.info, "labels": self.labels, "positions": self.positions,
                "impedance_active": self.impedance_active, "count_scale": self.count_scale,
                "unit": self.unit, "timestamp_verified": self.timestamp_verified}

    @abstractmethod
    def connect(self, options): ...
    @abstractmethod
    def disconnect(self): ...
    @abstractmethod
    def impedance(self, enabled): ...


def settings(options):
    result = {k: int(options.get(k, default)) for k, default in
              (("channels", 32), ("fs", 1000), ("points", 5), ("gain", 12))}
    if result["channels"] not in (8, 16, 32, 64):
        raise ValueError("当前适配器支持8/16/32/64通道；其他设备可通过新适配器扩展。")
    if result["fs"] not in (250, 500, 1000, 2000) or not 1 <= result["points"] <= 16:
        raise ValueError("采样率请选择250/500/1000/2000 Hz，每包点数为1至16。")
    if result["gain"] != 12:
        raise ValueError("当前Neusen协议配置保留已核对的增益编码12。")
    return result


def electrode_layout(channels, model="", cap=None):
    candidates = [Path(os.environ["STUDIO_CAP_CONFIG"])] if os.environ.get("STUDIO_CAP_CONFIG") else []
    project = Path(__file__).resolve().parents[2]
    candidates += [project/'Conf/setting.json']
    profile_file = Path(os.environ.get('HAND_BCI_LAYOUT_FILE', str(project/'configs/channel_layouts.json')))
    if profile_file.is_file():
        import json
        document = json.loads(profile_file.read_text(encoding='utf-8-sig'))
        if document.get('format') != 'handbci.channel-layouts.v1':
            raise ValueError('通道配置格式不正确。')
        matches = [x for x in document['layouts'] if len(x['channels']) == channels and (not cap or x['name'] == cap)]
        # A confirmed, explicitly supplied physical order takes precedence.
        if len(matches) == 1 and matches[0].get('confirmed') is True:
            item = matches[0]
            if len(set(item['channels'])) != channels:
                raise ValueError('设备物理通道表不能重复。')
            return list(item['channels']), list(item.get('positions', [])), item['name'], True
    conf = next((p for p in candidates if p.is_file()), Path(layout.DEFAULT_CONF))
    item = layout.Layout(path=str(conf), device_type=model, channels=channels, cap_name=cap)
    valid = item.ok and len(item.physical) == channels
    if cap and (not valid or item.cap_name != cap):
        raise ValueError("所选脑电帽与通道数不一致。")
    return (item.physical if valid else [f"CH{i+1:02d}" for i in range(channels)],
            item.positions if valid else [], item.cap_name if valid else "unverified", valid)


def event_code(payload):
    """ACP EventDataFormat + EventType + value. Unknown formats stay raw.

    The enums and [format,type,value] layout come from the supplied decompilation.
    Physical compatibility is established by a round-trip test, never by ACK alone.
    """
    if len(payload) < 3 or payload[1] != 1:  # TriggerIn only
        return None
    if payload[0] == 1 and len(payload[2:]) in (1, 2, 4):
        value = int.from_bytes(payload[2:], "little")
        return value if 0 <= value <= 255 else None
    if payload[0] == 2:
        try:
            text = payload[2:].rstrip(b"\x00").decode("ascii")
            return int(text) if text.isdecimal() and 0 <= int(text) <= 255 else None
        except UnicodeError:
            pass
    return None


class NeusenAdapter(DeviceAdapter):
    id, title = "neusen_acp", "博睿康 Neusen · ACP直连"
    count_scale, unit = acp.UV_PER_COUNT, "uV"
    capabilities = {"impedance": True, "events": True, "raw_packets": True,
                    "event_verification": "requires_round_trip"}

    def __init__(self, *args):
        super().__init__(*args)
        self.link = None
        self._backup = None
        self.labels_verified = False
        self.clock_pair = None
        self._clock_lock = threading.Lock()
        self._clock_history = deque(maxlen=4096)

    def clock_snapshot(self):
        """Recent EEG packet clocks, independent of controller/UI queues.

        Entries are (uptime_ms, shared_ms, arrival_mono_ns, packet_span_ms).
        Only decoded EEG packets populate this buffer; event arrival does not.
        """
        with self._clock_lock:
            return self.clock_pair, tuple(self._clock_history)

    def _clear_clock(self):
        with self._clock_lock:
            self.clock_pair = None
            self._clock_history.clear()

    @classmethod
    def discover(cls, known_hosts=()):
        return acp.discover(timeout=3.0, repeat=.5, extra_hosts=known_hosts)

    def _status(self, kind, detail):
        if kind in ("down", "stalled"):
            self.running = False
        self.on_status(kind, detail)

    def connect(self, options):
        self._clear_clock()
        self.settings = settings(options)
        host = str(options.get("host", "")).strip()
        port = int(options.get("port", 4097))
        if not host or not 1 <= port <= 65535 or port == 8712:
            raise ValueError("填写设备IP及ACP端口（默认4097）；8712是厂家转发口。")
        link = acp.Link(host, port, self._frame, self._status)
        self.link = link
        try:
            link.open()
            link.do_stop()
            time.sleep(.6)
            info = link.do_hello()
            self.info = info.as_dict()
            self.labels, self.positions, cap, self.labels_verified = electrode_layout(
                self.settings["channels"], info.type_code, options.get("cap") or None)
            self.info.update(cap=cap, labels_verified=self.labels_verified, host=host, port=port)
            self._configure(acp.WM_SIGNAL)
            link.do_start()
            self.connected, self.running = True, True
        except BaseException:
            self.disconnect()
            raise

    def _configure(self, mode):
        s = self.settings
        actual = self.link.do_set_mod_para(s["gain"], mode, s["channels"], s["fs"], s["points"])
        if tuple(actual) != (s["channels"], s["fs"], s["points"]):
            raise ValueError(f"设备回显不一致：请求{s}，返回{actual}，未开始正式采集。")

    def _frame(self, frame):
        if frame.module == acp.MOD_EVENT:
            self.on_packet(Packet("event", frame.wall_ns, frame.mono_ns, frame.dev_ts_ms,
                                  packet_id=frame.packet_id, module=frame.module,
                                  payload=frame.payload, code=event_code(frame.payload),
                                  detail={"payload_hex": frame.payload.hex(), "format": "acp_event"}))
            return
        if frame.module == acp.MOD_IMPEDANCE:
            ch = self.settings["channels"]
            if len(frame.payload) >= ch*2:
                raw = np.frombuffer(frame.payload[:ch*2], dtype="<u2")
                self.on_packet(Packet("impedance", frame.wall_ns, frame.mono_ns, frame.dev_ts_ms,
                    packet_id=frame.packet_id, module=3, payload=frame.payload,
                    detail={"kohm": [None if x == 65535 else float(x)/10 for x in raw]}))
            return
        if frame.module != acp.MOD_EEG:
            self.on_packet(Packet("other", frame.wall_ns, frame.mono_ns, frame.dev_ts_ms,
                                  module=frame.module, payload=frame.payload))
            return
        s = self.settings
        if len(frame.payload) != s["channels"]*s["points"]*3:
            self.on_packet(Packet("invalid", frame.wall_ns, frame.mono_ns, frame.dev_ts_ms,
                packet_id=frame.packet_id, module=frame.module, payload=frame.payload))
            self.on_status("payload_mismatch", {"bytes": len(frame.payload), "settings": dict(s)})
            return
        b = np.frombuffer(frame.payload, np.uint8).reshape(s["channels"], s["points"], 3).astype(np.int32)
        values = b[:, :, 0] | (b[:, :, 1] << 8) | (b[:, :, 2] << 16)
        values = np.where(values & 0x800000, values-0x1000000, values).astype(np.int32)
        with self._clock_lock:
            self.clock_pair = (frame.dev_ts_ms, frame.sync_ts_ms, frame.mono_ns)
            self._clock_history.append((*self.clock_pair, s['points']*1000/s['fs']))
        self.on_packet(Packet("eeg", frame.wall_ns, frame.mono_ns, frame.dev_ts_ms,
                              counts=values, packet_id=frame.packet_id, payload=frame.payload,
                              detail={"sync_time_ms": frame.sync_ts_ms}))

    def impedance(self, enabled):
        if not self.connected:
            raise ValueError("请先连接设备。")
        if bool(enabled) == self.impedance_active:
            return
        self.link.do_stop()
        self.running = False
        try:
            if enabled:
                self._backup = dict(self.settings)
                self.settings.update(fs=250, points=1)
                self._configure(acp.WM_IMPEDANCE_ONLINE)
            else:
                self.settings = dict(self._backup or self.settings)
                self._configure(acp.WM_SIGNAL)
            self.link.do_start()
            self.impedance_active, self.running = bool(enabled), True
        except BaseException:
            self.on_status("configuration_error", {"message": "阻抗模式切换失败，请断开后重新连接。"})
            raise

    def disconnect(self):
        self._clear_clock()
        if self.link:
            try:
                self.link.do_stop()
            finally:
                self.link.close()
                self.link = None
        self.connected = self.running = self.impedance_active = False


class SyntheticAdapter(DeviceAdapter):
    id, title, synthetic = "synthetic", "人工模拟设备（无外设）", True
    capabilities = {"impedance": True, "events": True, "raw_packets": True}
    count_scale, unit, timestamp_verified = 0.05, "synthetic_uV", True

    def __init__(self, *args):
        super().__init__(*args)
        self.stop = threading.Event()
        self.thread = None
        self.index = 0
        self.last_device_ms = 0
        self._backup = None

    def connect(self, options):
        self.settings = settings(options)
        self.labels, self.positions, cap, verified = electrode_layout(self.settings["channels"])
        self.info = {"model": "人工信号源", "cap": cap, "labels_verified": verified,
                     "physical_hardware": False}
        self.connected = self.running = True
        self.stop.clear()
        self.thread = threading.Thread(target=self._run, daemon=True, name="studio-synthetic")
        self.thread.start()

    def _run(self):
        rng, seq, deadline = np.random.default_rng(42), 0, time.perf_counter()
        while not self.stop.is_set():
            s = dict(self.settings)
            ch, n, fs = s["channels"], s["points"], s["fs"]
            t0 = self.last_device_ms
            times = t0/1000 + np.arange(n)/fs
            eeg = np.asarray(400*np.sin(2*np.pi*10*times[None, :] + np.arange(ch)[:, None]*.2)
                             + rng.normal(0, 25, (ch, n)), dtype=np.int32)
            wall, mono = time.time_ns(), time.perf_counter_ns()
            raw = (eeg.astype(np.uint32) & 0xFFFFFF)
            packed = np.stack([raw & 255, raw >> 8 & 255, raw >> 16 & 255], axis=-1).astype(np.uint8).tobytes()
            if self.impedance_active:
                self.on_packet(Packet("impedance", wall, mono, t0, module=3,
                                      detail={"kohm": [float(2+i%8) for i in range(ch)]}))
            self.on_packet(Packet("eeg", wall, mono, t0, counts=eeg, packet_id=seq, payload=packed))
            self.last_device_ms += n*1000/fs
            self.index += n
            seq = (seq+1) & 255
            deadline += n/fs
            self.stop.wait(max(0, deadline-time.perf_counter()))

    def emit_marker(self, code):
        self.on_packet(Packet("event", time.time_ns(), time.perf_counter_ns(), self.last_device_ms,
                              code=code, module=13, payload=bytes([1, 1, code]),
                              detail={"format": "simulated_acp_event", "physical_hardware": False}))

    def impedance(self, enabled):
        self.impedance_active = bool(enabled)

    def disconnect(self):
        self.stop.set()
        if self.thread:
            self.thread.join(3)
        self.connected = self.running = False


ADAPTERS = {cls.id: cls for cls in (SyntheticAdapter, NeusenAdapter)}


def register_adapter(adapter):
    if not issubclass(adapter, DeviceAdapter) or adapter.id in ADAPTERS:
        raise ValueError("设备适配器必须实现DeviceAdapter并使用唯一id。")
    ADAPTERS[adapter.id] = adapter
