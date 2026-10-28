"""Separate the device sample clock from transport arrival time."""
import threading

PIPELINE_REVISION = 'device-timeline-20261028.1'


class DeviceClockEstimate:
    """Host-to-device estimate using the lowest observed transport delay.

    Arrival bursts must not move a software cue back onto stale samples. This
    remains a host estimate, not a hardware-verified physical onset timestamp.
    """
    def __init__(self, fs):
        self.fs = float(fs)
        self.lock = threading.Lock()
        self.offset_ms = None
        self.origin_ms = None
        self.latest_end_ms = None
        self.transport_delay_ms = 0.

    def observe(self, device_ms, points, arrival_ns):
        end = float(device_ms)+(points-1)*1000/self.fs
        offset = arrival_ns/1e6-end
        with self.lock:
            if self.origin_ms is None:
                self.origin_ms = float(device_ms)
            self.offset_ms = offset if self.offset_ms is None else min(self.offset_ms, offset)
            self.latest_end_ms = end
            self.transport_delay_ms = max(0., offset-self.offset_ms)

    def estimate(self, host_ns):
        with self.lock:
            if self.offset_ms is None:
                return None
            return host_ns/1e6-self.offset_ms

    def snapshot(self, host_ns):
        with self.lock:
            return dict(clock='device_sample_time', sample_rate_hz=self.fs,
                        device_origin_ms=self.origin_ms, latest_device_time_ms=self.latest_end_ms,
                        estimated_device_now_ms=host_ns/1e6-self.offset_ms if self.offset_ms is not None else None,
                        transport_delay_ms=self.transport_delay_ms,
                        host_alignment='minimum_observed_arrival_offset', physical_onset_verified=False)
