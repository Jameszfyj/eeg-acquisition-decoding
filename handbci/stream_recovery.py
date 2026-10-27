"""Quarantine isolated corrupt EEG frames; never reconstruct missing samples."""
from collections import deque
import struct
import time


def packet_discontinuity(previous, packet, fs):
    """Compare verified EEG frames, independent of TCP arrival timing."""
    span = previous.counts.shape[1]*1000/fs
    expected = previous.device_ms+span
    expected_id = (previous.packet_id+1) & 255
    if abs(packet.device_ms-expected) <= .501 and packet.packet_id == expected_id:
        return None
    return dict(previous_device_ms=previous.device_ms, device_ms=packet.device_ms,
                expected_device_ms=expected, device_gap_ms=packet.device_ms-expected,
                previous_packet_id=previous.packet_id, packet_id=packet.packet_id,
                packet_delta=(packet.packet_id-previous.packet_id) & 255,
                packet_span_ms=span, previous_points=previous.counts.shape[1],
                points=packet.counts.shape[1], sample_rate=fs)


class ForwardGapRecovery:
    """Allow bounded, corroborated losses; never hide/reset a device clock."""
    max_gap_ms = 100
    max_events_per_minute = 10
    max_missing_ms_per_minute = 200

    def __init__(self):
        self.recent = deque()
        self.count = 0

    def accept(self, detail, now=None):
        now = time.monotonic() if now is None else now
        gap, span = detail['device_gap_ms'], detail['packet_span_ms']
        missing_frames = round(gap/span)
        if (gap <= 0 or gap > self.max_gap_ms or missing_frames < 1
                or abs(gap-missing_frames*span) > .501
                or detail['points'] != detail['previous_points']
                or detail['packet_delta'] != (missing_frames+1) % 256):
            raise RuntimeError('脑电时间与包序号不一致、发生回退或缺口超过100毫秒，已停止；原始记录保留。')
        while self.recent and now-self.recent[0][0] > 60:
            self.recent.popleft()
        if (len(self.recent) >= self.max_events_per_minute
                or sum(ms for _, ms in self.recent)+gap > self.max_missing_ms_per_minute):
            raise RuntimeError('60秒内脑电缺口超过10次或累计超过200毫秒，连接持续不稳定，已停止并保存。')
        self.recent.append((now, gap))
        self.count += 1
        return dict(detail, kind='stream_recovery', reason='packet_gap',
                    recovery_number=self.count, missing_frames=missing_frames,
                    missing_samples=round(gap*detail['sample_rate']/1000),
                    repaired_samples=0, filters_reset=True, decision_votes_cleared=True)


class StreamRecovery:
    def __init__(self, channels, points):
        self.payload_length = channels*points*3
        self.recent = deque()
        self.pending = []
        self.count = 0
        self.wait_started = None

    def request(self, detail, now=None):
        now = time.monotonic() if now is None else now
        if not isinstance(detail,dict):
            raise RuntimeError('CRC异常缺少可核验的原始包，已停止。')
        frame = bytes.fromhex(detail.get('candidate_frame_hex',''))
        if (len(frame)<17 or frame[:3]!=b'\xf0\x0f\x11' or frame[5:7]!=b'\x03\x02'
                or detail.get('payload_length')!=self.payload_length):
            raise RuntimeError('异常包不能确认为当前脑电数据，无法安全恢复，已停止。')
        while self.recent and now-self.recent[0] > 60:self.recent.popleft()
        self.recent.append(now)
        if len(self.recent)>=3:
            raise RuntimeError('60秒内已发生3次脑电CRC异常，连接不稳定，已停止；未继续自动恢复。')
        self.pending.append(dict(device_ms=struct.unpack_from('<I',frame,13)[0],packet_id=frame[8]))
        if self.wait_started is None:self.wait_started=now

    def expired(self, now=None):
        now = time.monotonic() if now is None else now
        if self.wait_started is not None and now-self.wait_started > 2:
            raise RuntimeError('CRC异常后2秒内未找到可确认的新脑电边界，已停止。')

    def accepts(self, packet, previous, fs):
        if not self.pending:return False
        self.expired()
        cutoff = max(item['device_ms'] for item in self.pending)
        if packet.device_ms < cutoff:return False
        if packet.device_ms-cutoff > 1000:
            raise RuntimeError('CRC异常后的设备时间跨度过大，无法安全恢复，已停止。')
        if previous is not None and packet.device_ms < previous.device_ms+previous.counts.shape[1]*1000/fs-.501:
            raise RuntimeError('CRC异常后设备时间倒退，已停止。')
        return True

    def complete(self):
        self.count += 1
        event = dict(kind='stream_recovery',recovery_number=self.count,
                     corrupt_frames=list(self.pending),reason='CRC',
                     repaired_samples=0,filters_reset=True,decision_votes_cleared=True)
        self.pending.clear();self.wait_started=None
        return event
