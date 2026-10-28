"""Measure ACP parser arrivals independently of storage, calibration and inference."""
from collections import deque
import threading
import time
from .device_timeline import DeviceClockEstimate


class ReceiveTiming:
    def __init__(self, fs, window=3., bad_seconds=2., max_backlog_seconds=1.):
        self.fs, self.window = float(fs), window
        self.bad_seconds, self.max_backlog = bad_seconds, max_backlog_seconds
        self.lock = threading.Lock()
        self.arrivals = deque()
        self.total = 0
        self.bad_since = None
        self.device_clock = DeviceClockEstimate(fs)

    def received(self, count, stamp_ns, device_ms=None):
        if device_ms is not None:
            self.device_clock.observe(device_ms, count, stamp_ns)
        stamp = stamp_ns/1e9
        with self.lock:
            self.arrivals.append((stamp, count))
            self.total += count
            while self.arrivals and stamp-self.arrivals[0][0] > self.window:
                self.arrivals.popleft()

    def snapshot(self, processed, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            entries = list(self.arrivals)
            total = self.total
        live = [(t,n) for t,n in entries if t >= now-self.window]
        span = live[-1][0]-live[0][0] if live else 0.
        duration = now-live[0][0]+live[0][1]/self.fs if live else 0.
        rate = sum(n for _,n in live)/duration if duration > 0 else None
        return dict(received_rate_hz=rate, received_points_per_second=rate, observation_seconds=span,
                    received_samples=total, processed_samples=processed,
                    queued_samples=max(0,total-processed),
                    backlog_seconds=max(0,total-processed)/self.fs,
                    last_arrival_age_seconds=now-entries[-1][0] if entries else None,
                    clock='ACP parser callback, independent of processing',
                    device_timeline=self.device_clock.snapshot(int(now*1e9)),
                    physical_sample_clock_verified=False)

    def restart_observation(self):
        """Discard transport-burst rate estimates, retaining totals and last arrival."""
        with self.lock:
            if self.arrivals:
                self.arrivals = deque([self.arrivals[-1]])
            self.bad_since = None

    def check(self, snapshot, now=None, *, allow_packet_loss=False):
        now = time.monotonic() if now is None else now
        if snapshot['backlog_seconds'] > self.max_backlog and not allow_packet_loss:
            raise RuntimeError(f"脑电处理积压约{snapshot['backlog_seconds']:.2f}秒，已停止以避免校准和动作使用滞后数据；不是采样率不匹配。")
        rate = snapshot['received_rate_hz']
        if allow_packet_loss:
            # ACP sample rate is checked against the device configuration. Loss
            # and transport bursts change arrival rate, not the sampling contract.
            self.bad_since = None
            return
        bad = (rate is not None and snapshot['observation_seconds'] >= 1.5
               and not .8*self.fs <= rate <= 1.2*self.fs)
        if not bad:
            self.bad_since = None
        elif self.bad_since is None:
            self.bad_since = now
        elif now-self.bad_since >= self.bad_seconds:
            raise RuntimeError(f"接收回调速率持续{self.bad_seconds:g}秒异常，约{rate:.1f} Hz，模型要求{self.fs:g} Hz；已停止，未补点或改采样率。")
