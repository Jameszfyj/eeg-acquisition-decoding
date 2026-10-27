"""Independent display filtering; never supplies samples to the decoder or recorder."""
from __future__ import annotations

from collections import deque
import math

import numpy as np
from scipy import signal


class WaveformPreview:
    channels = ('C3', 'Cz', 'C4')

    def __init__(self, input_channels=(), sample_rate=None, unit='uV'):
        self.fs = float(sample_rate) if sample_rate else None
        self.unit = unit
        names = [str(name).strip().casefold() for name in input_channels]
        self.indices = [names.index(ch.casefold()) if ch.casefold() in names else None
                        for ch in self.channels]
        self.available = [ch for ch, ix in zip(self.channels, self.indices) if ix is not None]
        self.step = max(1, math.ceil(self.fs/200)) if self.fs else 1
        self.points = deque(maxlen=1200)
        self.last_index = None
        self.zi = None
        self.sos = None
        self.high = min(40., self.fs*.2) if self.fs else None
        self.notch = bool(self.fs and self.fs > 100)
        if self.fs and self.high > 1 and self.available:
            self.sos = signal.butter(4, [1., self.high], btype='bandpass', fs=self.fs, output='sos')
            if self.notch:
                b, a = signal.iirnotch(50., 30., fs=self.fs)
                self.sos = np.concatenate([signal.tf2sos(b, a), self.sos])

    def append(self, raw, source_indices):
        if self.sos is None or not len(source_indices):
            return
        # Extract a copy before the model can transform its own input in place.
        values = np.asarray(raw)[[ix for ix in self.indices if ix is not None]].astype(float, copy=True)
        if not np.isfinite(values).all():
            self.zi = None
            self.last_index = None
            self.points.clear()
            return
        if self.last_index is not None and int(source_indices[0]) != self.last_index+1:
            self.zi = None
            self.points.clear()
        if self.zi is None:
            self.zi = signal.sosfilt_zi(self.sos)[:, None, :]*values[None, :, :1]
        filtered, self.zi = signal.sosfilt(self.sos, values, axis=1, zi=self.zi)
        keep = np.flatnonzero(np.asarray(source_indices) % self.step == 0)
        for j in keep:
            row, available_index = [int(source_indices[j])/self.fs], 0
            for ix in self.indices:
                row.append(float(filtered[available_index, j]) if ix is not None else None)
                if ix is not None:
                    available_index += 1
            self.points.append(row)
        self.last_index = int(source_indices[-1])
        cutoff = self.last_index/self.fs-5
        while self.points and self.points[0][0] < cutoff:
            self.points.popleft()

    def snapshot(self):
        return dict(channels=list(self.channels), available=list(self.available), unit=self.unit,
                    seconds=5, points=[list(row) for row in self.points],
                    display_sample_rate=self.fs/self.step if self.fs else None,
                    bandpass_hz=[1., self.high] if self.sos is not None else None,
                    notch_hz=50 if self.notch and self.sos is not None else None,
                    display_only=True)
