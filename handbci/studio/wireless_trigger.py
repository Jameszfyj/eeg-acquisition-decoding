"""Neuracle wireless event input, separate from USB/DCP command output.

Protocol evidence: supplied NeuroNexus TriggerBoxController (ACP Hello + event
subscription), ResetTimeStampUtlity (SendSyncCode to amplifier then box, query
amplifier), and DeviceMonitor.BroadcastTrigger (map the shared counter to EEG
uptime). Never align by host arrival time or use a DCP ACK as an input event.
"""
from __future__ import annotations

import secrets
import struct
import threading
import time

from .devices import Packet, event_code
from .vendor import acp


class WirelessTriggerInput:
    # The USB sender still requires its independent echo within two seconds.
    # Delays above the original 200 ms allowance require a matching EEG packet
    # in the hardware-clock buffer, and never move the event to arrival time.
    BUFFERED_EVENT_LIMIT_MS = 1500

    def __init__(self, device, on_packet, on_status):
        self.device, self.on_packet, self.on_status = device, on_packet, on_status
        self.link = None
        self.info = {}
        self.sync_code = None
        self.synced = False
        self.error = ''
        self.eeg_offset = None
        self.stop = threading.Event()
        self.monitor = None

    def state(self):
        return {'connected': bool(self.link and self.link.state == 'connected' and not self.error),
                'clock_synced': self.synced, 'sync_code': self.sync_code,
                'info': dict(self.info), 'error': self.error,
                'event_source': 'triggerbox_acp', 'timestamp_basis': 'shared_hardware_counter_to_eeg_uptime'}

    def _status(self, kind, detail):
        if kind in ('down', 'stalled', 'crc_error', 'tail_error', 'header_error', 'frame_error'):
            self.synced = False
            self.error = detail.get('error', kind)
            self.on_status('down', {'source': 'triggerbox_acp', 'error': 'TriggerBox无线接收异常：'+self.error})

    @staticmethod
    def _sync_request(link, code=None):
        cmd = acp.CMD_ASKSYNCCODE if code is None else acp.CMD_SENDSYNCCODE
        payload = b'' if code is None else struct.pack('<i', code)
        response = link.request(cmd, lambda seq: acp.command(acp.TOKEN_CMD, acp.MOD_DEVICE, cmd, seq, payload))
        if response is None or response.result != 1 or len(response.payload) != 4:
            raise ValueError('未收到完整的设备同步码应答。')
        return struct.unpack('<i', response.payload)[0], response

    def connect(self, host, port=4097):
        d = self.device
        if not d or d.id != 'neusen_acp' or not d.running or d.impedance_active:
            raise ValueError('无线同步需要先连接真实Neusen脑电设备，并结束阻抗检查。')
        if not host or not 1 <= int(port) <= 65535 or int(port) in (4321, 8712):
            raise ValueError('填写TriggerBox的数据IP及ACP端口4097；此处不使用打标端口4321。')
        if host == d.info.get('host') and int(port) == d.info.get('port'):
            raise ValueError('TriggerBox数据地址不能与脑电设备地址相同。')
        self.link = acp.Link(host, int(port), self._frame, self._status,
                             silent_ms=60_000, down_ms=120_000)
        try:
            self.link.open()
            info = self.link.do_hello()
            if not any(x in info.type_code.lower() for x in ('nst3tb', 'trigger', 'titing')):
                raise ValueError('无线数据端返回的设备身份不是已识别的TriggerBox。')
            self.info = {**info.as_dict(), 'host': host, 'port': int(port)}
            # The manufacturer does not Start/SetModPara for event-only input.
            before = time.perf_counter_ns()
            code = secrets.randbelow(2**31-1)+1
            for link in (d.link, self.link):
                result, _ = self._sync_request(link, code)
                if result != code:
                    raise ValueError('设备未确认本次同步码，不能进行事件对齐。')
            for link in (d.link, self.link):
                result, response = self._sync_request(link)
                if result != code or response.sync_ts_ms is None or response.sync_ts_ms >= 2000:
                    raise ValueError('同步码或复位后的硬件时间未通过核验。')
            deadline = time.monotonic()+2
            while time.monotonic() < deadline:
                pair = d.clock_pair
                if pair and pair[2] >= before and pair[1] is not None and pair[1] < 2000:
                    self.eeg_offset = (pair[0]-pair[1]) & 0xFFFFFFFF
                    self.sync_code, self.synced = code, True
                    break
                time.sleep(.01)
            if not self.synced:
                raise ValueError('脑电流未出现本次同步后的硬件时间，不能进行事件对齐。')
            self.stop.clear()
            self.monitor = threading.Thread(target=self._monitor, name='triggerbox-sync-monitor', daemon=True)
            self.monitor.start()
            return self.state()
        except BaseException:
            self.close()
            raise

    def _monitor(self):
        while not self.stop.wait(1):
            try:
                code, _ = self._sync_request(self.link)
                if code != self.sync_code:
                    raise ValueError('TriggerBox同步码已改变，请重新连接并自检。')
            except Exception as exc:
                if not self.stop.is_set(): self._status('down', {'error': str(exc)})
                return

    def _frame(self, frame):
        if frame.module != acp.MOD_EVENT:
            return
        # TriggerOut is the actual output event reported by this independent
        # TriggerBox input stream. Do not widen the amplifier's TriggerIn parser.
        payload = frame.payload
        code = event_code(payload[:1]+b'\x01'+payload[2:]) if len(payload) >= 3 and payload[1] in (1, 2) else None
        snapshot = getattr(self.device, 'clock_snapshot', None)
        pair, history = snapshot() if callable(snapshot) else (self.device.clock_pair, ())
        error = ''
        mapped = None
        distance = None
        arrival_delay = None
        matched_packet = None
        basis = 'current_device_clock'
        if not self.synced:
            error = '无线事件时钟尚未同步'
        elif not self.device.running or self.device.impedance_active or not pair or frame.mono_ns-pair[2] > 500_000_000:
            error = '脑电流已停止或超过0.5秒未更新'
        elif frame.sync_ts_ms is None or pair[1] is None or ((pair[0]-pair[1]) & 0xFFFFFFFF) != self.eeg_offset:
            error = '脑电时钟同步关系已变化'
        else:
            mapped = (frame.sync_ts_ms+self.eeg_offset) & 0xFFFFFFFF
            distance = ((mapped-pair[0]+2**31) & 0xFFFFFFFF)-2**31
            if abs(distance) > 200:
                # A late network delivery is distinct from clock drift. Accept
                # a bounded delay only when its exact shared hardware time is
                # covered by an actual, already received EEG packet.
                if -self.BUFFERED_EVENT_LIMIT_MS <= distance < -200:
                    for uptime, shared, received_ns, span_ms in reversed(history):
                        within = ((mapped-uptime+2**31) & 0xFFFFFFFF)-2**31
                        delay_ms = (frame.mono_ns-received_ns)/1e6
                        if (shared is not None and ((uptime-shared) & 0xFFFFFFFF) == self.eeg_offset
                                and 0 <= within < span_ms
                                and ((shared+within+self.eeg_offset) & 0xFFFFFFFF) == mapped
                                and 0 <= delay_ms <= self.BUFFERED_EVENT_LIMIT_MS):
                            matched_packet, arrival_delay = uptime, delay_ms
                            basis = 'buffered_device_clock'
                            break
                if matched_packet is None:
                    error = ('无线事件超过近期脑电硬件时间校验范围：'
                             f'相对最新脑电{distance:+.0f}毫秒，未找到有效缓存匹配')
        # An individual late/unmatched frame is not a broken connection or a
        # reset synchronization code. Reject that label without poisoning the
        # synchronized stream; the next valid frame can still be confirmed.
        aligned = not error and code is not None
        detail = {'source': 'triggerbox_acp', 'event_type': payload[1] if len(payload)>1 else None,
                  'trigger_uptime_ms': frame.dev_ts_ms, 'sync_time_ms': frame.sync_ts_ms,
                  'sync_code': self.sync_code, 'clock_verified': aligned, 'alignment_error': error,
                  'eeg_offset_ms': self.eeg_offset,
                  'event_vs_latest_eeg_ms': distance,
                  'arrival_delay_ms': arrival_delay,
                  'matched_eeg_packet_ms': matched_packet,
                  'alignment_basis': basis if aligned else None}
        self.on_packet(Packet('event' if aligned else 'other', frame.wall_ns, frame.mono_ns,
                              mapped if aligned else frame.dev_ts_ms, module=frame.module,
                              packet_id=frame.packet_id, payload=payload, code=code if aligned else None,
                              detail=detail, raw_device_ms=frame.dev_ts_ms))
        if not aligned:
            self.on_status('wireless_event_unaligned', {**detail, 'payload_hex': payload.hex(),
                                                       'error': error or '无线事件格式未识别，未作为标签使用'})

    def close(self):
        self.synced = False
        self.stop.set()
        if self.link:
            self.link.close()
        if self.monitor and self.monitor is not threading.current_thread():
            self.monitor.join(4)
        self.link = None
