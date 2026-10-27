"""TriggerBox discovery and an isolated EEG/marker check; never drives the hand."""
from __future__ import annotations

import ipaddress
import threading
import time

from .config import write_json
from .eeg_connection import discovery_hosts
from .prompted_control import OnlineMarkers
from .receiver_access import ReceiverLease
from .studio.devices import NeusenAdapter
from .triggerbox import serial_ports


def discover_trigger_devices(known_hosts=()):
    """Use the studio's UDP discovery and identity filter, without opening ports."""
    ports = serial_ports()
    if not ports['available']:
        raise ValueError(ports['message'])
    devices = {}
    hosts = discovery_hosts(known_hosts)
    replies = NeusenAdapter.discover(known_hosts=hosts) if hosts else NeusenAdapter.discover()
    for item in replies:
        kind = str(item.get('type', ''))
        if not any(token in kind.lower() for token in ('nst3tb', 'trigger', 'titing')):
            continue
        try:
            address = ipaddress.ip_address(item['ip'])
        except (ValueError, KeyError):
            continue
        if (address.version != 4 or not address.is_private or address.is_loopback
                or address.is_multicast or address.is_unspecified):
            continue
        ip = str(address)
        devices[ip] = dict(ip=ip, port=4097, type=kind, serial=str(item.get('serial', '')))
    usable_ports = [p for p in ports['ports']
                    if not p['description'].lower().startswith('bluetooth') and '蓝牙' not in p['description']]
    result = sorted(devices.values(), key=lambda d: int(ipaddress.ip_address(d['ip'])))
    return dict(ports=usable_ports, devices=result,
                message=f'找到{len(usable_ports)}个串口、{len(result)}台无线TriggerBox；串口身份须通过连接检验。')


def probe_trigger_connection(folder, device, settings, cancel):
    """Connect EEG, verify serial identity, synchronize (optional) and test echo.

    Writes diagnostics only. No model inference, hand connection or trial starts.
    Both sockets and serial transport are released before returning success.
    """
    if settings.get('marking') != 'serial':
        raise ValueError('硬件连接检验仅用于真实USB TriggerBox。')
    guard = threading.RLock()
    ready = threading.Event()
    markers, previous = None, None
    faults, samples, last_arrival = [], 0, None
    fs = device['sample_rate']

    def packet(p):
        nonlocal previous, samples, last_arrival
        with guard:
            if p.kind == 'eeg':
                if p.counts.shape[0] != len(device['channels']):
                    faults.append('实际脑电通道数不符。')
                if previous is not None:
                    expected = previous.device_ms+previous.counts.shape[1]*1000/fs
                    if abs(p.device_ms-expected) > .501 or p.packet_id != ((previous.packet_id+1)&255):
                        faults.append('脑电时间或包序号不连续。')
                previous = p
                samples += p.counts.shape[1]
                last_arrival = time.monotonic()
                ready.set()
            elif markers is not None:
                markers.observe(p)

    def status(kind, detail):
        if kind in ('down', 'stalled', 'configuration_error', 'payload_mismatch', 'crc_error', 'tail_error'):
            with guard:
                faults.append(f'{kind}: {detail}')

    adapter = NeusenAdapter(packet, status)
    folder.mkdir(parents=True, exist_ok=False)
    result, error = None, ''
    try:
        with ReceiverLease(device['host'], device['port']):
            try:
                if cancel.is_set():
                    raise RuntimeError('TriggerBox连接检验已取消。')
                adapter.connect(dict(host=device['host'], port=device['port'], channels=len(device['channels']),
                                     fs=fs, points=5, gain=12))
                if (not adapter.labels_verified or adapter.unit != device['unit']
                        or adapter.settings['fs'] != fs or len(adapter.labels) != len(device['channels'])):
                    raise ValueError('脑电设备通道、单位或采样率未通过检查。')
                deadline = time.monotonic()+3
                while not ready.wait(.05):
                    if cancel.is_set():
                        raise RuntimeError('TriggerBox连接检验已取消。')
                    if time.monotonic() >= deadline:
                        raise RuntimeError('未收到脑电数据，不能核验事件对齐。')
                markers = OnlineMarkers(folder, adapter, settings, packet, status)
                markers.connect()
                with guard:
                    if cancel.is_set():
                        raise RuntimeError('TriggerBox连接检验已取消。')
                    if faults:
                        raise RuntimeError('连接检验期间数据异常：'+faults[0])
                    if last_arrival is None or time.monotonic()-last_arrival > .5:
                        raise RuntimeError('脑电数据已停止，不能确认事件链路正常。')
                    result = dict(ok=True, identity_verified=True, echo_verified=True,
                                  trigger=markers.trigger.state(), test_event=copy_last(markers),
                                  eeg_samples=samples, physical_time_verified=False,
                                  physical_action=False)
            finally:
                try:
                    if markers is not None:
                        markers.close()
                finally:
                    adapter.disconnect()
        result.update(closed_after_check=True, connected=False,
                      message='TriggerBox设备身份及事件回传检验通过；测试连接已释放，正式开始时重新连接并自检。')
        return result
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        write_json(folder/'connection_check.json', dict(result=result, error=error, complete=not error))


def copy_last(markers):
    import copy
    return copy.deepcopy(markers.trigger.last)
