"""Short ACP connection/stream check using the decoder's own device adapter."""
from __future__ import annotations

import queue
import ipaddress
import threading
import time

from .four_class_online import channel_order
from .receiver_access import ReceiverLease
from .studio.devices import NeusenAdapter


def discovery_hosts(hosts):
    if not isinstance(hosts, (list, tuple)):
        raise ValueError('设备搜索历史格式无效。')
    result = set()
    for value in hosts[:16]:
        if not isinstance(value, str):
            continue
        try:
            address = ipaddress.ip_address(value)
        except (ValueError, TypeError):
            continue
        if (address.version == 4 and address.is_private and not address.is_loopback
                and not address.is_multicast and not address.is_unspecified):
            result.add(str(address))
    return sorted(result, key=lambda ip: int(ipaddress.ip_address(ip)))


def discover_eeg(known_hosts=()):
    """UDP announcements only; never opens the advertised control/data socket."""
    devices = {}
    hosts = discovery_hosts(known_hosts)
    replies = NeusenAdapter.discover(known_hosts=hosts) if hosts else NeusenAdapter.discover()
    for item in replies:
        kind = str(item.get('type', '')).upper()
        if not kind.startswith(('NSW', 'NEUSEN')):
            continue  # TriggerBox and unknown responders are not EEG receivers.
        try:
            address = ipaddress.ip_address(item['ip'])
        except (ValueError, KeyError):
            continue
        if address.version != 4:
            continue
        ip = str(address)
        devices[ip] = dict(ip=ip, port=4097, type=item['type'], serial=str(item.get('serial', '')))
    return sorted(devices.values(), key=lambda d: int(ipaddress.ip_address(d['ip'])))


def probe_acp(device, cancel_event: threading.Event):
    packets, faults = queue.Queue(4096), queue.Queue()

    def packet(p):
        if p.kind == 'eeg':
            try:
                packets.put_nowait(p)
            except queue.Full:
                faults.put('接收队列溢出，连接测试失败。')

    def status(kind, detail):
        if kind in ('down', 'stalled', 'configuration_error', 'payload_mismatch', 'crc_error', 'tail_error'):
            faults.put(f'设备数据异常：{kind}，{detail}')

    adapter = NeusenAdapter(packet, status)
    options = dict(host=device['host'], port=device['port'], channels=len(device['channels']),
                   fs=device['sample_rate'], points=5, gain=12)
    with ReceiverLease(device['host'], device['port']):
        try:
            if cancel_event.is_set():
                raise RuntimeError('连接测试已取消。')
            adapter.connect(options)
            if not adapter.labels_verified:
                raise ValueError('设备通道名称未经核对，不能确认模型匹配。')
            if len(adapter.labels) != len(device['channels']) or set(adapter.labels) != set(device['channels']):
                raise ValueError('设备通道名称与模型完整通道表不符。')
            order = [adapter.labels.index(ch) for ch in device['channels']]
            if (adapter.unit != device['unit'] or adapter.settings['fs'] != device['sample_rate']
                    or len(adapter.labels) != len(device['channels'])):
                raise ValueError('设备通道数、采样率或输入单位与所选模型不匹配。')
            fs = float(device['sample_rate'])
            total, first, previous = 0, None, None
            deadline = time.monotonic()+5
            while total < int(2*fs):
                if cancel_event.is_set():
                    raise RuntimeError('连接测试已取消。')
                if not faults.empty():
                    raise RuntimeError(faults.get_nowait())
                if time.monotonic() >= deadline:
                    raise RuntimeError('未在5秒内收到足够的连续脑电数据，请检查设备和网络。')
                try:
                    p = packets.get(timeout=.1)
                except queue.Empty:
                    continue
                n = p.counts.shape[1]
                if p.counts.shape[0] != len(device['channels']):
                    raise ValueError('实际脑电数据通道数与所选模型不匹配。')
                if previous is not None:
                    expected = previous.device_ms+previous.counts.shape[1]*1000/fs
                    if abs(p.device_ms-expected) > .501 or p.packet_id != ((previous.packet_id+1)&255):
                        raise RuntimeError('测试数据的设备时间或包序号出现缺口，不能确认连续连接正常。')
                first = p if first is None else first
                previous, total = p, total+n
            if not faults.empty():
                raise RuntimeError(faults.get_nowait())
            duration = (previous.mono_ns-first.mono_ns)/1e9+previous.counts.shape[1]/fs
            if duration <= 0:
                raise RuntimeError('接收时间无效，连接测试失败。')
            rate = total/duration
            if not .8*fs <= rate <= 1.2*fs:
                raise RuntimeError(f'实际接收速率约{rate:.1f} Hz，与模型要求{fs:g} Hz不符。')
            description = adapter.describe()
            result = dict(connection_verified=True, matched_model=True, samples_received=total,
                received_rate_hz=round(rate, 1), sample_rate=fs, channels=len(adapter.labels),
                unit=adapter.unit, device=description, reorder_indices=order, gap_count=0,
                tested_seconds=round(total/fs, 3))
        finally:
            adapter.disconnect()
    result['device'].update(connected=False, running=False)
    result.update(closed_after_check=True, connected=False,
        message='脑电连接测试通过，已收到连续数据并匹配所选模型。测试连接已释放，可直接开始解码。')
    return result
