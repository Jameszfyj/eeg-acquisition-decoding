"""Record lost/corrupt EEG frames without reconstructing missing samples."""
from collections import deque
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
    """Loss frequency/size is quality evidence, not a session stop condition."""

    def __init__(self):
        self.recent = deque()
        self.count = 0

    def accept(self, detail, now=None):
        now = time.monotonic() if now is None else now
        gap, span = detail['device_gap_ms'], detail['packet_span_ms']
        missing_frames = round(gap/span)
        if gap < -.501:
            raise RuntimeError('脑电设备时间回退或重叠，已停止；原始记录保留。')
        if detail['points'] != detail['previous_points']:
            raise RuntimeError('脑电每包点数发生变化，已停止；原始记录保留。')
        while self.recent and now-self.recent[0][0] > 60:
            self.recent.popleft()
        self.recent.append((now, gap))
        self.count += 1
        return dict(detail, kind='stream_recovery', reason='packet_gap',
                    recovery_number=self.count, missing_frames=max(0, missing_frames),
                    missing_samples=max(0, round(gap*detail['sample_rate']/1000)),
                    clock_id_consistent=(abs(gap-missing_frames*span) <= .501
                        and detail['packet_delta'] == (missing_frames+1) % 256),
                    repaired_samples=0, filters_reset=True, decision_votes_cleared=False)


class StreamRecovery:
    def __init__(self, channels, points):
        self.payload_length = channels*points*3
        self.recent = deque()
        self.pending = []
        self.count = 0
        self.wait_started = None

    def request(self, detail, now=None, after_mono_ns=None, after_packet_serial=None):
        now = time.monotonic() if now is None else now
        # Corrupt headers (including their timestamps) cannot define a boundary.
        # Use parser callback order (clock fallback for callers without a serial);
        # only a subsequently verified frame ends quarantine. The parser has
        # already rejected the corrupt bytes.
        while self.recent and now-self.recent[0] > 60:self.recent.popleft()
        self.recent.append(now)
        self.pending.append(dict(after_mono_ns=after_mono_ns or time.monotonic_ns(),
                                 after_packet_serial=after_packet_serial))
        if self.wait_started is None:self.wait_started=now

    def expired(self, now=None):
        now = time.monotonic() if now is None else now
        return self.wait_started is not None and now-self.wait_started > 2

    def accepts(self, packet, previous, fs, arrival_mono_ns=None, packet_serial=None):
        if not self.pending:return False
        serials = [item['after_packet_serial'] for item in self.pending if item['after_packet_serial'] is not None]
        if packet_serial is not None and serials:
            if packet_serial <= max(serials):return False
        elif arrival_mono_ns is not None and arrival_mono_ns <= max(item['after_mono_ns'] for item in self.pending):
            return False
        if previous is not None and packet.device_ms < previous.device_ms+previous.counts.shape[1]*1000/fs-.501:
            raise RuntimeError('CRC异常后设备时间倒退，已停止。')
        return True

    def complete(self):
        self.count += 1
        event = dict(kind='stream_recovery',recovery_number=self.count,
                     corrupt_frames=list(self.pending),reason='CRC',
                     repaired_samples=0,filters_reset=True,decision_votes_cleared=False)
        self.pending.clear();self.wait_started=None
        return event
