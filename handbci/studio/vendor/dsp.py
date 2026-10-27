"""Display-side signal conditioning.

The recorder keeps raw counts. This module only prepares the *display* copy:
a causal, per-channel SOS chain (high-pass, low-pass, notch) with persistent
state, so a 1 Hz high-pass removes the electrode drift that otherwise dwarfs
20 uV EEG on screen.

scipy is used when present; without it the chain degrades to a first-order
DC blocker per channel.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from scipy.signal import butter, iirnotch, sosfilt, tf2sos
    HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    HAVE_SCIPY = False


class DisplayFilter:
    def __init__(self, fs: int, channels: int, hp: float = 1.0,
                 lp: float = 40.0, notch: float = 50.0):
        self.fs = max(1, int(fs))
        self.channels = max(1, int(channels))
        self.hp = float(hp or 0.0)
        self.lp = float(lp or 0.0)
        self.notch = float(notch or 0.0)
        self._sos: Optional[np.ndarray] = None
        self._zi: Optional[np.ndarray] = None
        self._dc: Optional[np.ndarray] = None
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        sections: List[np.ndarray] = []
        nyq = self.fs / 2.0
        if HAVE_SCIPY:
            try:
                if 0 < self.hp < nyq * 0.99:
                    sections.append(butter(2, self.hp, btype='highpass', fs=self.fs, output='sos'))
                if 0 < self.lp < nyq * 0.99:
                    sections.append(butter(4, self.lp, btype='lowpass', fs=self.fs, output='sos'))
                if 0 < self.notch < nyq * 0.99:
                    b, a = iirnotch(self.notch, 30.0, fs=self.fs)
                    sections.append(tf2sos(b, a))
            except ValueError:
                sections = []
        self._sos = np.vstack(sections) if sections else None
        self._zi = None
        self._dc = None
        if self._sos is not None:
            n_sections = self._sos.shape[0]
            self._zi = np.zeros((n_sections, self.channels, 2), np.float32)

    def reset(self) -> None:
        if self._sos is not None:
            self._zi = np.zeros_like(self._zi)
        self._dc = None

    # ------------------------------------------------------------------ run
    def process(self, x: np.ndarray) -> np.ndarray:
        """x: (channels, n) counts -> filtered float32 counts (DC removed)."""
        y = np.asarray(x, dtype=np.float32)
        if self._sos is None:
            # fallback: one-pole DC blocker, ~0.5 Hz
            if self._dc is None:
                self._dc = y[:, :1].copy()
            a = 0.995
            out = np.empty_like(y)
            prev_in = self._dc
            prev_out = np.zeros_like(prev_in)
            for i in range(y.shape[1]):
                cur = y[:, i:i + 1]
                prev_out = a * (prev_out + cur - prev_in)
                prev_in = cur
                out[:, i:i + 1] = prev_out
            self._dc = prev_in
            return out
        if y.shape[0] != self.channels:
            self.channels = y.shape[0]
            n_sections = self._sos.shape[0]
            self._zi = np.zeros((n_sections, self.channels, 2), np.float32)
        y, self._zi = sosfilt(self._sos, y, axis=1, zi=self._zi)
        return y.astype(np.float32)

    def as_dict(self) -> dict:
        return {'hp': self.hp, 'lp': self.lp, 'notch': self.notch,
                'scipy': HAVE_SCIPY}
