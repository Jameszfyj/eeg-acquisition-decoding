"""Neuracle DCP over TCP, per the manufacturer's TriggerBox manual §3.1.

Only the DCP device-name and single-event commands are used. ACP/4097 is not
this protocol. Model compatibility is established by handshake, then by EEG events.
"""
from __future__ import annotations

import socket
import struct
import time


class TcpTriggerBoxConnection:
    def __init__(self, timeout=2.0):
        self.timeout = float(timeout)
        self._socket = None
        self._device_name = None

    @property
    def connected(self):
        return self._socket is not None

    @property
    def device_name(self):
        return self._device_name

    def connect(self, host, port):
        if self.connected:
            raise RuntimeError("TriggerBox已经连接。")
        try:
            self._socket = socket.create_connection((host, int(port)), timeout=self.timeout)
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            response = self._command(4, b"")
            name = response.rstrip(b"\x00").decode("ascii", errors="strict")
            if not name or len(name) > 128 or not all(32 <= ord(c) < 127 for c in name):
                raise ValueError("DCP设备名称无效，未确认TriggerBox身份。")
            if not any(token in name.casefold() for token in ("trigger", "nst3tb", "titing")):
                raise ValueError(f"DCP回复的设备名不是已识别的TriggerBox：{name}")
            self._device_name = name
        except BaseException:
            self.disconnect()
            raise

    def _read_exactly(self, size, deadline):
        result = bytearray()
        while len(result) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("TriggerBox DCP回复超时。")
            self._socket.settimeout(remaining)
            block = self._socket.recv(size - len(result))
            if not block:
                raise ConnectionError("TriggerBox在回复完成前断开了网络连接。")
            result.extend(block)
        return bytes(result)

    def _command(self, function_id, payload):
        if not self.connected:
            raise RuntimeError("TriggerBox网络未连接。")
        deadline = time.monotonic() + self.timeout
        self._socket.settimeout(self.timeout)
        self._socket.sendall(struct.pack("<BBH", 1, function_id, len(payload)) + payload)
        device, function, length = struct.unpack("<BBH", self._read_exactly(4, deadline))
        if device != 1:
            raise ValueError("回复不是DCP设备1；请核对打标接口，不能使用4097设备管理端口。")
        if length > 4096:
            raise ValueError("DCP回复长度异常，停止读取。")
        if function == 131:
            data = self._read_exactly(length, deadline)
            raise RuntimeError(f"TriggerBox返回DCP错误：{data.hex() or '无错误码'}")
        if function != function_id:
            raise ValueError(f"DCP回复功能码{function}与请求{function_id}不一致。")
        return self._read_exactly(length, deadline)

    def send_event(self, code):
        if isinstance(code, bool) or not isinstance(code, int) or not 0 <= code <= 255:
            raise ValueError("事件码须为0至255的整数。")
        try:
            return self._command(225, bytes([code]))
        except BaseException:
            self.disconnect()
            raise

    def disconnect(self):
        sock, self._socket = self._socket, None
        self._device_name = None
        if sock is not None:
            sock.close()
