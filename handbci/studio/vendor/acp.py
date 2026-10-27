"""Neuracle ACP v2 link layer (server side).

Reconstructed from Neuracle.EEGRecorder.* :
  discovery  UDP 1020, 18-byte hybrid query
  session    TCP 4097, one socket for control + data
  frame      token 0x0FF0 (hello 0x0550) / CRC16-CCITT over buf[2:-4] / tail 0xF00F

Design rule for this app: NOTHING is synthesised. A frame is delivered with the
wall-clock time at which it was parsed; if the link goes quiet, no frames are
emitted and the caller records a gap marker instead.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

# ------------------------------------------------------------------ constants

TOKEN_CMD = 0x0FF0
TOKEN_HELLO = 0x0550
TOKEN_BCAST = 0x5555
TAIL_CMD = 0xF00F
TAIL_HELLO = 0x5005

PKG_COMMAND = 1
PKG_RESPONSE = 2
PKG_ALERT = 3
PKG_NOTIFICATION = 4
PKG_BROADCAST = 5

CMD_HELLO = 1
CMD_RESET = 2
CMD_SETMODPARA = 3
CMD_SETDEVPARA = 4
CMD_START = 5
CMD_GETDEVSTATUS = 6
CMD_GETMODSTATUS = 7
CMD_ASKSYNCCODE = 8
CMD_SENDSYNCCODE = 9
CMD_SETPROMPT = 10

MOD_DEVICE = 1
MOD_EEG = 2
MOD_IMPEDANCE = 3
MOD_ACC = 7
MOD_GYRO = 8
MOD_MAG = 9
MOD_EVENT = 13

WM_NORMAL = 1
WM_WAVE = 4
WM_SHORT = 5
WM_LEADOFF = 6
WM_POWERDOWN = 7
WM_TIMESTAMP = 8
WM_RLD = 9
WM_RLDALL = 10
WM_IMPEDANCE_OFFLINE = 12
WM_IMPEDANCE_ONLINE = 13
WM_SIGNAL = 14
WM_RAW_IMPEDANCE_OFFLINE = 17
WM_RAW_IMPEDANCE_ONLINE = 18

DEFAULT_DISCOVERY_PORT = 1020
DEFAULT_DATA_PORT = 4097

# physical scaling declared by the amplifier (DeviceSettings.xml)
PHYS_MIN_UV = -375000.0
PHYS_MAX_UV = 375000.0
DIG_MIN = -8388608
DIG_MAX = 8388607
UV_PER_COUNT = (PHYS_MAX_UV - PHYS_MIN_UV) / (DIG_MAX - DIG_MIN)


# ------------------------------------------------------------------ CRC

def _crc_table() -> List[int]:
    table = []
    for dividend in range(256):
        rem = dividend << 8
        for _ in range(8):
            rem = ((rem << 1) ^ 0x1021) & 0xFFFF if rem & 0x8000 else (rem << 1) & 0xFFFF
        table.append(rem)
    return table


_CRC_TABLE = _crc_table()


def crc16(data: bytes) -> int:
    rem = 0xFFFF
    for b in data:
        rem = ((rem << 8) ^ _CRC_TABLE[((rem >> 8) ^ b) & 0xFF]) & 0xFFFF
    return rem


def frame_crc(frame: bytes) -> int:
    return crc16(frame[2:len(frame) - 4])


# ------------------------------------------------------------------ frame build

def build_frame(token: int, header_len: int, pkg_type: int,
                body: bytes, payload: bytes, tail: int) -> bytes:
    head = struct.pack('<HBHB', token, header_len, len(payload), pkg_type) + body
    frame = bytearray(head + payload + b'\x00\x00' + struct.pack('<H', tail))
    struct.pack_into('<H', frame, len(frame) - 4, frame_crc(bytes(frame)))
    return bytes(frame)


def command(token: int, module: int, cmd: int, seq: int,
            payload: bytes = b'', tail: int = TAIL_CMD) -> bytes:
    return build_frame(token, 9, PKG_COMMAND, bytes([module, cmd, seq & 0xFF]), payload, tail)


def hello(seq: int = 1) -> bytes:
    return command(TOKEN_HELLO, MOD_DEVICE, CMD_HELLO, seq, b'', TAIL_HELLO)


def reset(seq: int = 1) -> bytes:
    return command(TOKEN_CMD, MOD_DEVICE, CMD_RESET, seq)


def start(seq: int = 1) -> bytes:
    return command(TOKEN_CMD, MOD_DEVICE, CMD_START, seq)


def set_mod_para(gain: int, work_mode: int, channels: int, fs: int,
                 points: int, seq: int = 1) -> bytes:
    payload = struct.pack('<BBHHB', gain, work_mode, channels, fs, points)
    return command(TOKEN_CMD, MOD_EEG, CMD_SETMODPARA, seq, payload)


def notification(packet_id: int, ts_ms: int) -> bytes:
    body = bytes([MOD_DEVICE, 1, packet_id & 0xFF]) + struct.pack('<II', ts_ms, ts_ms)
    return build_frame(TOKEN_CMD, 17, PKG_NOTIFICATION, body, b'', TAIL_CMD)


def discovery_query(seq: int = 1) -> bytes:
    bc = build_frame(TOKEN_CMD, 9, PKG_BROADCAST, bytes([MOD_DEVICE, 2, seq]), b'', TAIL_CMD)
    return bytes([0x55, 0x55, 0x02, 0x00, 0x02]) + bc


# ------------------------------------------------------------------ models

@dataclass
class DeviceInfo:
    type_code: str = ''
    serial: int = 0
    major: int = 0
    minor: int = 0
    hw: int = 0
    fw: int = 0

    def as_dict(self) -> Dict:
        return {
            'type': self.type_code,
            'serial': f'{self.serial:08X}',
            'acp': f'{self.major}.{self.minor}',
            'hw': self.hw,
            'fw': self.fw,
        }


def parse_device_info(payload: bytes) -> DeviceInfo:
    if len(payload) < 20:
        raise ValueError('device info payload too short')
    return DeviceInfo(
        type_code=payload[2:14].decode('ascii', 'replace').strip('\x00 ').rstrip('X'),
        serial=struct.unpack_from('<I', payload, 14)[0],
        major=payload[0],
        minor=payload[1],
        hw=payload[18],
        fw=payload[19],
    )


@dataclass
class Frame:
    """One received data frame, timestamped at parse time."""
    wall_ns: int
    mono_ns: int
    dev_ts_ms: int
    packet_id: int
    module: int
    alert_type: int
    payload: bytes
    # Hardware-synchronised counter at offset 9; uptime above is offset 13.
    sync_ts_ms: Optional[int] = None

    @property
    def payload_len(self) -> int:
        return len(self.payload)


@dataclass
class Response:
    module: int
    command: int
    packet_id: int
    sequence: int
    result: int
    error_code: int
    dev_ts_ms: int
    payload: bytes
    sync_ts_ms: Optional[int] = None


# ------------------------------------------------------------------ discovery

def local_ipv4s() -> List[str]:
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ips)


def parse_announcement(raw: bytes) -> Optional[Tuple[DeviceInfo, str]]:
    """Returns (info, announced_ip) for both v2 and legacy replies."""
    off = raw.find(b'\xF0\x0F')
    if off >= 0 and len(raw) - off >= 13:
        fr = raw[off:]
        if fr[5] == PKG_BROADCAST and fr[7] == 3:
            plen = struct.unpack_from('<H', fr, 3)[0]
            payload = fr[9:9 + plen]
            if len(payload) >= 20:
                return parse_device_info(payload), ''
    if len(raw) >= 20 and struct.unpack_from('<H', raw, 0)[0] == TOKEN_BCAST and raw[4] == 3:
        return DeviceInfo(type_code='NSW1BA32'), '.'.join(str(b) for b in raw[16:20])
    return None


def discover(port: int = DEFAULT_DISCOVERY_PORT, timeout: float = 3.0,
             repeat: float = 1.0, extra_hosts=()) -> List[Dict]:
    """Broadcast the v2 query; returns one entry per responding device."""
    found: Dict[str, Dict] = {}
    sequence = time.monotonic_ns() % 255 + 1

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    bound = False
    try:
        sock.bind(('', port))       # devices answer to port 1020
        bound = True
    except OSError:
        pass
    sock.settimeout(0.2)

    targets = {'255.255.255.255'}
    targets.update(extra_hosts)
    for ip in local_ipv4s():
        targets.add(ip.rsplit('.', 1)[0] + '.255')

    started = time.time()
    last_send = 0.0
    while time.time() - started < timeout:
        if time.time() - last_send >= repeat:
            query = discovery_query(sequence)
            sequence = sequence % 255 + 1
            for target in targets:
                try:
                    sock.sendto(query, (target, port))
                except OSError:
                    pass
            last_send = time.time()
        try:
            raw, addr = sock.recvfrom(4096)
        except socket.timeout:
            continue
        parsed = parse_announcement(raw)
        if not parsed:
            continue
        info, announced_ip = parsed
        ip = announced_ip or addr[0]
        found[ip] = {'ip': ip, 'bound_port': bound, 'reply_from': addr[0], **info.as_dict()}
    sock.close()
    return sorted(found.values(), key=lambda d: d['ip'])


# ------------------------------------------------------------------ link

class Link:
    """TCP session. Calls on_frame() from the reader thread; never invents data."""

    def __init__(self, ip: str, port: int, on_frame: Callable[[Frame], None],
                 on_event: Callable[[str, Dict], None],
                 silent_ms: int = 1000, down_ms: int = 4000):
        self.ip = ip
        self.port = port
        self.on_frame = on_frame
        self.on_event = on_event
        self.silent_ms = silent_ms
        self.down_ms = down_ms

        self.sock: Optional[socket.socket] = None
        self._buf = bytearray()
        self._rx_tail = bytearray()
        self._rx_total = 0
        self._seq = 0
        self._responses: Dict[Tuple[int, int], Response] = {}
        self._resp_cv = threading.Condition()
        self._reader: Optional[threading.Thread] = None
        self._alive = False
        self._keepalive: Optional[threading.Thread] = None
        self._last_rx_ns = 0
        self._silent = False
        self._state = 'idle'
        self._lock = threading.Lock()

    # -- lifecycle
    def open(self, timeout: float = 3.0) -> None:
        sock = socket.create_connection((self.ip, self.port), timeout=timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(0.2)
        self.sock = sock
        self._alive = True
        self._buf.clear()
        self._rx_tail.clear()
        self._rx_total = 0
        self._last_rx_ns = time.time_ns()
        self._silent = False
        self._reader = threading.Thread(target=self._read_loop, name='acp-reader', daemon=True)
        self._reader.start()
        self._keepalive = threading.Thread(target=self._keepalive_loop, name='acp-keepalive', daemon=True)
        self._keepalive.start()
        self._set_state('connected')

    def close(self) -> None:
        self._alive = False
        with self._lock:
            sock, self.sock = self.sock, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        self._set_state('closed')
        for worker in (self._reader, self._keepalive):
            if worker is not None and worker is not threading.current_thread():
                worker.join(2)

    def _set_state(self, state: str) -> None:
        if state != self._state:
            self._state = state

    @property
    def state(self) -> str:
        return self._state

    # -- wire
    def next_seq(self) -> int:
        self._seq = (self._seq + 1) & 0xFF
        return self._seq

    def send(self, data: bytes) -> None:
        sock = self.sock
        if sock is None:
            raise ConnectionError('link closed')
        sock.sendall(data)

    def _keepalive_loop(self) -> None:
        packet_id = 0
        while self._alive:
            try:
                self.send(notification(packet_id, int(time.time() * 1000) & 0xFFFFFFFF))
                packet_id = (packet_id + 1) & 0xFF
            except OSError:
                return
            time.sleep(1.0)

    def _read_loop(self) -> None:
        down_fired = False
        while self._alive:
            sock = self.sock
            if sock is None:
                break
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                idle_ns = time.time_ns() - self._last_rx_ns
                if not self._silent and idle_ns >= self.silent_ms * 1_000_000:
                    self._silent = True
                    self.on_event('silent', {'idle_ms': idle_ns // 1_000_000})
                if not down_fired and idle_ns >= self.down_ms * 1_000_000:
                    down_fired = True
                    self.on_event('stalled', {'idle_ms': idle_ns // 1_000_000})
                continue
            except OSError as exc:
                if self._alive:
                    self.on_event('down', {'error': str(exc)})
                break
            if not chunk:
                if self._alive:
                    self.on_event('down', {'error': 'peer closed'})
                break

            if self._silent:
                self._silent = False
                self.on_event('resumed', {})
            down_fired = False
            now_ns = time.time_ns()
            self._last_rx_ns = now_ns
            self._received(chunk, now_ns)

        if self._alive:
            self._alive = False

    def _received(self, chunk: bytes, now_ns: int) -> None:
        # Preserve a bounded copy before any framing or resynchronisation. These
        # are socket bytes, so a later CRC report can be independently replayed.
        self._rx_total += len(chunk)
        self._rx_tail.extend(chunk)
        if len(self._rx_tail) > 65536:
            del self._rx_tail[:-65536]
        self._buf.extend(chunk)
        self._drain(now_ns)

    def _drain(self, now_ns: int) -> None:
        buf = self._buf
        while True:
            if len(buf) < 6:
                return
            token = struct.unpack_from('<H', buf, 0)[0]
            if token not in (TOKEN_CMD, TOKEN_HELLO):
                del buf[0]
                continue
            header_len = buf[2]
            payload_len = struct.unpack_from('<H', buf, 3)[0]
            total = header_len + payload_len + 4
            if header_len < 6 or total > 8192:
                self.on_event('header_error', {'header_length': header_len, 'total': total})
                del buf[0]
                continue
            if len(buf) < total:
                return
            frame = bytes(buf[:total])
            calculated_crc = frame_crc(frame)
            received_crc = struct.unpack_from('<H', frame, total - 4)[0]
            if calculated_crc != received_crc:
                self.on_event('crc_error', {'discarded_bytes': 1,
                    'candidate_length': total, 'header_length': header_len,
                    'payload_length': payload_len, 'package': frame[5],
                    'received_crc': received_crc, 'calculated_crc': calculated_crc,
                    'candidate_frame_hex': frame.hex(),
                    'socket_bytes_received': self._rx_total,
                    'candidate_stream_offset': self._rx_total-len(buf) if self._rx_total >= len(buf) else None,
                    'raw_receive_tail_start': self._rx_total-len(self._rx_tail),
                    'raw_receive_tail_hex': bytes(self._rx_tail).hex()})
                del buf[0]
                continue
            expected_tail = TAIL_HELLO if token == TOKEN_HELLO else TAIL_CMD
            if struct.unpack_from('<H', frame, total-2)[0] != expected_tail:
                self.on_event('tail_error', {'discarded_bytes': 1})
                del buf[0]
                continue
            if (frame[5] == PKG_RESPONSE and header_len != 20) or (frame[5] == PKG_ALERT and header_len != 17):
                self.on_event('header_error', {'header_length': header_len, 'package': frame[5]})
                del buf[:total]
                continue
            del buf[:total]
            self._dispatch(frame, now_ns)

    def _dispatch(self, frame: bytes, now_ns: int) -> None:
        pkg = frame[5]
        if pkg == PKG_RESPONSE:
            resp = Response(
                module=frame[6], command=frame[7], packet_id=frame[8], sequence=frame[9],
                result=frame[10], error_code=frame[11],
                dev_ts_ms=struct.unpack_from('<I', frame, 16)[0],
                sync_ts_ms=struct.unpack_from('<I', frame, 12)[0],
                payload=frame[20:20 + struct.unpack_from('<H', frame, 3)[0]],
            )
            with self._resp_cv:
                self._responses[(resp.command, resp.sequence)] = resp
                self._resp_cv.notify_all()
        elif pkg == PKG_ALERT:
            plen = struct.unpack_from('<H', frame, 3)[0]
            try:
                self.on_frame(Frame(
                    wall_ns=now_ns,
                    mono_ns=time.perf_counter_ns(),
                    dev_ts_ms=struct.unpack_from('<I', frame, 13)[0],
                    sync_ts_ms=struct.unpack_from('<I', frame, 9)[0],
                    packet_id=frame[8],
                    module=frame[6],
                    alert_type=frame[7],
                    payload=frame[17:17 + plen],
                ))
            except Exception as exc:  # a consumer bug must not kill the link
                self.on_event('frame_error', {'error': f'{type(exc).__name__}: {exc}'})

    # -- commands
    def wait_response(self, cmd: int, seq: int, timeout: float = 3.0) -> Optional[Response]:
        deadline = time.time() + timeout
        with self._resp_cv:
            while True:
                resp = self._responses.pop((cmd, seq), None)
                if resp is not None:
                    return resp
                left = deadline - time.time()
                if left <= 0:
                    return None
                self._resp_cv.wait(left)

    def request(self, cmd: int, builder: Callable[[int], bytes],
                timeout: float = 3.0) -> Optional[Response]:
        seq = self.next_seq()
        self.send(builder(seq))
        return self.wait_response(cmd, seq, timeout)

    def do_hello(self) -> DeviceInfo:
        resp = self.request(CMD_HELLO, hello)
        if resp is None:
            raise RuntimeError('no Hello response')
        if resp.result != 1:
            raise RuntimeError(f'Hello rejected (result={resp.result}, error={resp.error_code})')
        if len(resp.payload) < 20:
            raise RuntimeError('Hello payload too short')
        return parse_device_info(resp.payload)

    def do_set_mod_para(self, gain: int, work_mode: int, channels: int, fs: int,
                        points: int) -> Tuple[int, int, int]:
        resp = self.request(CMD_SETMODPARA,
                            lambda s: set_mod_para(gain, work_mode, channels, fs, points, s))
        if resp is None:
            raise RuntimeError('no SetModPara response')
        if resp.result != 1:
            raise RuntimeError(f'SetModPara rejected (result={resp.result}, error={resp.error_code})')
        if len(resp.payload) == 7:
            g, wm, ch, rate, pts = struct.unpack('<BBHHB', resp.payload)
            if (g, wm, ch, rate, pts) != (gain, work_mode, channels, fs, points):
                raise ValueError('EEG configuration echo does not match the request')
            return ch, rate, pts
        raise ValueError('EEG configuration response has no complete 7-byte echo')

    def do_start(self) -> None:
        resp = self.request(CMD_START, start)
        if resp is None:
            raise RuntimeError('no Start response')
        if resp.result != 1:
            raise RuntimeError(f'Start rejected (result={resp.result}, error={resp.error_code})')

    def do_stop(self) -> None:
        try:
            self.send(reset(self.next_seq()))
        except OSError:
            pass
