"""Trigger transports are independent from EEG adapters."""
from __future__ import annotations

import threading
import time

from ..triggerbox import serial_ports, validate_port


class TriggerManager:
    def __init__(self, on_event, wait_echo, get_device, on_packet=None, on_status=None):
        self.on_event, self.wait_echo, self.get_device = on_event, wait_echo, get_device
        self.lock = threading.RLock()
        self.box = None
        self.mode = "software"
        self.name, self.port = "", ""
        self.verified = False
        self.last = {}
        self.wireless = None
        self.on_packet, self.on_status = on_packet, on_status

    def state(self):
        return {"mode": self.mode, "connected": self.mode == "software" or self.box is not None,
                    "name": self.name, "port": self.port,
                    "echo_verified": self.verified and (not self.wireless or self.wireless.synced),
                    "last": dict(self.last),
                    "wireless": self.wireless.state() if self.wireless else None,
                    "physical_hardware": self.mode == "serial"}

    def connect(self, mode, port="", *, wireless_host="", wireless_port=4097):
        if mode not in ("software", "serial", "simulated"):
            raise ValueError("当前提供软件标记、USB串口和模拟TriggerBox。其他传输通过触发适配器扩展。")
        with self.lock:
            self.disconnect()
            if wireless_host and mode != 'serial':
                raise ValueError('无线事件接收目前配合USB串口打标使用。')
            self.mode = mode
            if mode == "serial":
                from ..vendor.trigger_box import TriggerBoxConnection
                box = TriggerBoxConnection()
                box.connect(validate_port(port))
                if not box.device_name or not any(x in box.device_name.lower() for x in ("trigger", "titing", "nst3tb")):
                    box.disconnect()
                    raise ValueError("串口返回的设备名不符合已知TriggerBox身份。")
                self.box, self.name, self.port = box, box.device_name, validate_port(port)
                if wireless_host:
                    from .wireless_trigger import WirelessTriggerInput
                    self.wireless = WirelessTriggerInput(self.get_device(), self.on_packet, self.on_status)
                    try:
                        self.wireless.connect(wireless_host, wireless_port)
                    except BaseException:
                        self.disconnect()
                        raise
            elif mode == "simulated":
                device = self.get_device()
                if device is None or not device.synthetic:
                    raise ValueError("模拟TriggerBox只可用于人工模拟设备。")
                self.box, self.name = device, "模拟TriggerBox（无真实硬件）"
            self.mode = mode
            self.verified = False
        return self.state()

    def disconnect(self):
        with self.lock:
            if self.wireless:
                self.wireless.close()
                self.wireless = None
            if self.box and self.mode == "serial":
                self.box.disconnect()
            self.box = None
            self.mode, self.name, self.port = "software", "", ""
            self.verified = False

    def send(self, code, *, require_echo=False, wait_for_echo=True):
        with self.lock:
            if self.mode != "software" and self.box is None:
                raise ValueError("请先连接TriggerBox。")
            if self.wireless and not self.wireless.synced:
                self.verified = False
                raise ValueError('TriggerBox无线时钟未通过同步，请重新连接并自检。')
            self.verified = False
            before = time.perf_counter_ns()
            row = {"event_id": int(code), "trigger_mode": self.mode,
                   "send_wall_ns": time.time_ns(), "send_mono_ns": before,
                   "acknowledged": False, "echo_verified": False}
            self.last = row
            if not 0 <= int(code) <= 255:
                raise ValueError('事件码须在0至255之间。')
            if self.mode == "serial":
                ack = self.box.send_event(int(code))
                row.update(acknowledged=True, ack_wall_ns=time.time_ns(), ack_payload_hex=(ack or b'').hex())
            elif self.mode == "simulated":
                self.box.emit_marker(int(code))
                row.update(acknowledged=True, ack_wall_ns=time.time_ns())
            if self.mode != "software":
                if wait_for_echo:
                    self.confirm(row, require_echo=require_echo)
                else:
                    row['verification_pending'] = True
            self.last = row
            return row

    def confirm(self, row, *, timeout=2.0, require_echo=False, before=None):
        """Confirm on a separate worker during recording; self-test remains strict."""
        after = row['send_mono_ns']
        kwargs = {'source':'triggerbox_acp'} if self.wireless else {}
        if before is not None: kwargs['before'] = before
        echo = self.wait_echo(row['event_id'], after, max(0., timeout), **kwargs)
        # A reply outside the original two-second deadline cannot verify a mark.
        valid = (echo and echo['mono_time_ns'] <= after+2_000_000_000
                 and (not self.wireless or self.wireless.synced))
        row['verification_pending'] = False
        if valid:
            row.update(echo_verified=True, observed_device_ms=echo['device_time_ms'],
                       observed_wall_ns=echo['wall_time_ns'],
                       verification='hardware_wireless_echo' if self.wireless else 'simulated_echo' if self.mode == 'simulated' else 'hardware_eeg_echo',
                       observed_event_detail=echo.get('detail'))
        else:
            row['echo_verified'] = False
        # Do not let a previous mark's asynchronous confirmation overwrite the
        # state of a newer send. This lock never blocks the EEG consumer.
        with self.lock:
            if self.last.get('send_mono_ns') == after:
                self.verified = bool(valid)
                self.last.update(row)
        if not valid and require_echo:
            route = '已同步的TriggerBox无线事件流' if self.wireless else '脑电流'
            hint = '' if self.wireless or self.mode != 'serial' else ' 使用WiFi无线同步时，请启用无线事件接收并连接TriggerBox数据IP。'
            raise ValueError(f'TriggerBox已应答，但2秒内未从{route}确认同一事件码；不会自动改用软件标签。{hint}')
        return row

    def test(self):
        if self.mode == "software":
            raise ValueError("软件标记没有硬件回传自检；请选择USB或模拟TriggerBox。")
        result = self.send(240 + time.time_ns()%10, require_echo=True)
        self.verified = True
        return result
