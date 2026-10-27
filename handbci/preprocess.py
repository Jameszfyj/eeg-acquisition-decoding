"""离线和在线共用的有状态FIR；模型分频仅在已经到达的窗口内执行。"""
from __future__ import annotations

import numpy as np
from scipy import signal


class StreamPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.fs = float(config["device"]["sample_rate"])
        prep = config["preprocess"]
        self.output_fs = float(prep["output_rate"])
        self.factor = int(round(self.fs / self.output_fs))
        self.n_channels = len(config["device"]["channels"])
        taps = int(prep["numtaps"])
        self.lowpass = signal.firwin(taps, prep["lowpass_hz"], fs=self.fs, window=("kaiser", 5.65))
        self.highpass = signal.firwin(taps, prep["highpass_hz"], fs=self.output_fs,
                                     pass_zero="highpass", window=("kaiser", 5.65))
        self.delay_seconds = (taps - 1) / (2 * self.fs) + (taps - 1) / (2 * self.output_fs)
        self.history_source_samples = (len(self.lowpass)-1) + (len(self.highpass)-1)*self.factor
        self.initial_trim = int(round(prep["trim_seconds"] * self.output_fs))
        self.reset()

    def window_bounds(self, retained, received_end_index):
        """Raw source support, including causal FIR history, of a model window."""
        return dict(source_start_index=int(retained[0])-self.history_source_samples,
                    source_end_index=int(retained[-1]), received_end_index=int(received_end_index),
                    sample_rate=self.fs, input_seconds=len(retained)/self.output_fs,
                    filter_history_seconds=self.history_source_samples/self.fs)

    def reset(self, first_source_index: int = 0) -> None:
        self.expected_index = int(first_source_index)
        self.low_state = np.zeros((self.n_channels, len(self.lowpass) - 1))
        self.high_state = np.zeros((self.n_channels, len(self.highpass) - 1))
        self.trim_remaining = self.initial_trim

    def process(self, data: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(data, dtype=np.float64)
        idx = np.asarray(indices, dtype=np.int64)
        if x.ndim != 2 or x.shape != (self.n_channels, len(idx)):
            raise ValueError("输入必须为[通道,时间]，索引数量须与采样点一致。")
        if not len(idx):
            return np.empty((self.n_channels, 0), np.float32), idx
        if not np.all(np.isfinite(x)):
            raise ValueError("脑电含NaN或无穷值。")
        if idx[0] != self.expected_index or np.any(np.diff(idx) != 1):
            raise ValueError("数据不连续，须显式重置滤波和解码缓存后再处理。")
        low, self.low_state = signal.lfilter(self.lowpass, [1.0], x, axis=-1, zi=self.low_state)
        keep = idx % self.factor == 0
        decimated, kept = low[:, keep], idx[keep]
        self.expected_index = int(idx[-1]) + 1
        if not len(kept):
            return np.empty((self.n_channels, 0), np.float32), kept
        out, self.high_state = signal.lfilter(self.highpass, [1.0], decimated,
                                             axis=-1, zi=self.high_state)
        drop = min(self.trim_remaining, len(kept))
        self.trim_remaining -= drop
        return out[:, drop:].astype(np.float32), kept[drop:]


def preprocess_record(config: dict, eeg: np.ndarray, indices: np.ndarray,
                      chunk_samples: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    stream = StreamPreprocessor(config)
    stream.reset(int(indices[0]))
    values, retained = [], []
    for start in range(0, eeg.shape[1], chunk_samples):
        y, ix = stream.process(eeg[:, start:start + chunk_samples], indices[start:start + chunk_samples])
        if len(ix):
            values.append(y)
            retained.append(ix)
    if not retained:
        raise ValueError("记录过短，剔除滤波启动段后没有数据。")
    return np.concatenate(values, axis=1), np.concatenate(retained)


def filter_bank(windows: np.ndarray, fs: float) -> np.ndarray:
    """参考IFNetV2：4–16/16–40 Hz双频带，返回[N,2,C,T]。"""
    x = np.asarray(windows, dtype=np.float32)
    if x.ndim != 3 or fs <= 80:
        raise ValueError("IFNet双频带输入必须为[N,C,T]，采样率必须大于80 Hz。")
    result = []
    for band in ((4.0, 16.0), (16.0, 40.0)):
        sos = signal.butter(4, band, btype="bandpass", fs=fs, output="sos")
        result.append(signal.sosfiltfilt(sos, x, axis=-1).astype(np.float32))
    return np.stack(result, axis=1)


class TrainStandardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)

    @classmethod
    def fit(cls, train_windows: np.ndarray):
        return cls(train_windows.mean(axis=(0, 2)),
                   np.maximum(train_windows.std(axis=(0, 2)), 1e-6))

    def transform(self, windows: np.ndarray) -> np.ndarray:
        return ((windows - self.mean[None, :, None]) / self.std[None, :, None]).astype(np.float32)
