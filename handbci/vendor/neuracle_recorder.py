#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal real-time EEG recorder for Neuracle TCP streams.

Main interface
--------------
EEGRecorder.connect()
EEGRecorder.connect_synthetic()
EEGRecorder.disconnect()
EEGRecorder.read_latest(frame)
EEGRecorder.start_preprocess(...)
EEGRecorder.read_latest_preprocessed(frame)

Important alignment rule
------------------------
Each preprocessed sample stores the absolute raw-sample index from which it was
generated. Therefore, read_latest_preprocessed() returns an offset defined as:

    offset_seconds =
        (latest_raw_sample_index - latest_preprocessed_source_index) / raw_srate

The offset includes:
1. Any current lag of the preprocessing thread.
2. The 0 to downsample_factor - 1 sample alignment lag caused by decimation.

It does not add a theoretical IIR group-delay estimate. Causal IIR filters are
stateful and their output is timestamped at the current input sample.
"""

from __future__ import annotations

import os
import queue
import re
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

import h5py
import numpy as np
from scipy.signal import butter, sosfilt


def decode_trigger_values(raw) -> np.ndarray:
    """Decode Neuracle trigger samples to integer event ids.

    Neuracle H5/TCP trigger channels can carry the uint32 bit pattern of a
    float32 event id, e.g. 0x3F800000 for event 1. Plain integer trigger
    channels are also accepted.
    """
    values = np.asarray(raw)
    if values.size == 0:
        return np.asarray(values, dtype=np.uint32)

    numeric = np.asarray(values, dtype=np.float64)
    nonzero = numeric[np.nonzero(numeric)]
    if nonzero.size == 0:
        return np.zeros(values.shape, dtype=np.uint32)

    if float(np.nanmax(np.abs(nonzero))) > 65535.0:
        words = np.rint(numeric).astype(np.uint32, copy=False)
        decoded = words.view(np.float32).astype(np.float64)
        return np.rint(decoded).astype(np.uint32)

    return np.rint(numeric).astype(np.uint32)


def parse_bdf_local_start_datetime(bdf_file: str | Path) -> datetime:
    path = Path(bdf_file)
    with path.open("rb") as handle:
        header = handle.read(256)
    recording = header[88:168].decode("ascii", errors="ignore").strip()
    match = re.search(r"ExamE(\d{8})(\d{6})(\d+)", recording)
    if match:
        date_text, time_text, fraction = match.groups()
        microsecond = int((fraction[:6]).ljust(6, "0"))
        return datetime.strptime(date_text + time_text, "%Y%m%d%H%M%S").replace(microsecond=microsecond)
    date_text = header[168:176].decode("ascii", errors="ignore").strip()
    time_text = header[176:184].decode("ascii", errors="ignore").strip()
    return datetime.strptime(f"{date_text} {time_text}", "%d.%m.%y %H.%M.%S")


@dataclass(frozen=True)
class RawBlockMeta:
    """One block of raw data as it was PUBLISHED into the ring buffer.

    This is the single place where "when did this data arrive on this machine" is
    decided: `acq_wall_ns` / `acq_mono_ns` are taken once, when the producer hands
    the block over, and every consumer copies that same value instead of asking
    the clock again - so a per-frame timestamp channel, a chunk/loop row and a
    latency figure all refer to the *same* instant.

    `epoch` identifies the acquisition source: it starts at 0 and is incremented
    for every connect / synthetic start / replay start / source switch, because
    each of those numbers its own samples from 0 (so `source_index` alone is not
    a global identity - `(epoch, source_index)` is).
    """

    seq: int                 # block counter inside its epoch
    epoch: int
    kind: str                # "device" | "synthetic" | "replay"
    source_first: int
    source_last: int
    n_samples: int
    n_channels: int
    acq_wall_ns: int         # time.time_ns()   - local wall clock, human readable
    acq_mono_ns: int         # time.perf_counter_ns() - monotone, for intervals
    seek: bool = False       # the block follows a replay SEEK (content jump)


class RingBuffer:
    """Thread-safe channel-first ring buffer with absolute source indices."""

    def __init__(self, n_channels: int, capacity: int, dtype=np.float64):
        if n_channels <= 0:
            raise ValueError("n_channels must be positive.")
        if capacity <= 0:
            raise ValueError("capacity must be positive.")

        self.n_channels = int(n_channels)
        self.capacity = int(capacity)
        self.data = np.zeros((self.n_channels, self.capacity), dtype=dtype)
        self.source_indices = np.full(self.capacity, -1, dtype=np.int64)

        self.write_position = 0
        self.size = 0
        self.total_written = 0
        # Bumped by every `clear()`. A caller that remembers the generation it
        # last read learns IMMEDIATELY that the buffer was reset (a source switch,
        # a reconnect) instead of waiting for the new sample counter to pass the
        # old one - which, after a counter restart, is minutes of silence.
        self.generation = 0

        self.lock = threading.RLock()
        self.updated = threading.Condition(self.lock)

    @property
    def latest_source_index(self) -> Optional[int]:
        with self.lock:
            if self.size == 0:
                return None
            position = (self.write_position - 1) % self.capacity
            return int(self.source_indices[position])

    def append(self, data: np.ndarray, source_indices: np.ndarray) -> None:
        data = np.asarray(data)
        source_indices = np.asarray(source_indices, dtype=np.int64)

        if data.ndim != 2:
            raise ValueError("data must have shape (n_channels, n_samples).")
        if data.shape[0] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {data.shape[0]}."
            )
        if source_indices.ndim != 1 or source_indices.shape[0] != data.shape[1]:
            raise ValueError(
                "source_indices must have shape (n_samples,) matching data."
            )

        n_samples = data.shape[1]
        if n_samples == 0:
            return

        # If a chunk is larger than the whole buffer, retain only its tail.
        if n_samples >= self.capacity:
            data = data[:, -self.capacity :]
            source_indices = source_indices[-self.capacity :]
            n_samples = self.capacity

        with self.updated:
            first = min(n_samples, self.capacity - self.write_position)
            second = n_samples - first

            self.data[:, self.write_position : self.write_position + first] = (
                data[:, :first]
            )
            self.source_indices[
                self.write_position : self.write_position + first
            ] = source_indices[:first]

            if second > 0:
                self.data[:, :second] = data[:, first:]
                self.source_indices[:second] = source_indices[first:]

            self.write_position = (self.write_position + n_samples) % self.capacity
            self.size = min(self.capacity, self.size + n_samples)
            self.total_written += n_samples
            self.updated.notify_all()

    def read_latest(self, frame: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return up to the latest `frame` samples and their source indices."""
        frame = int(frame)
        if frame <= 0:
            raise ValueError("frame must be positive.")

        with self.lock:
            n = min(frame, self.size)
            if n == 0:
                return (
                    np.empty((self.n_channels, 0), dtype=self.data.dtype),
                    np.empty(0, dtype=np.int64),
                )

            start = (self.write_position - n) % self.capacity
            if start < self.write_position:
                data = self.data[:, start : self.write_position].copy()
                indices = self.source_indices[start : self.write_position].copy()
            else:
                data = np.concatenate(
                    (self.data[:, start:], self.data[:, : self.write_position]),
                    axis=1,
                )
                indices = np.concatenate(
                    (
                        self.source_indices[start:],
                        self.source_indices[: self.write_position],
                    )
                ).copy()

            return data, indices

    def read_after_source_index(
        self,
        last_source_index: int,
        max_samples: Optional[int] = None,
        generation: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Return samples whose source index is greater than last_source_index.

        Returns
        -------
        data:
            Shape (n_channels, n_samples).
        indices:
            Absolute source indices.
        overrun:
            True when requested samples were already overwritten.

        Performance note
        ----------------
        The workspace holds up to `buffer_seconds` of RAW EEG (120 s x 40 ch at
        1000 Hz = 38 MB) and the realtime pipeline calls this for every chunk
        (~50 ms).  Selecting with a mask over the WHOLE buffer made one catch-up
        read cost ~55 ms and the full-buffer path ~330 ms - all of it holding
        `self.lock`, which the acquisition thread needs for `append`, so the
        recorder starved, the device FIFO overran and the stream stalled for
        seconds (measured; see `_buf_bench.py`).  Because the source indices are
        monotonic, the new range is a contiguous run at the write head - so the
        read is O(new samples): at most two slice copies.
        """
        with self.lock:
            if generation is not None and int(generation) != int(self.generation):
                # The buffer was cleared since the caller's last read: everything
                # it tracked is gone. Reported as an overrun so the caller resets
                # and resynchronises from the current head at once.
                return (
                    np.empty((self.n_channels, 0), dtype=self.data.dtype),
                    np.empty(0, dtype=np.int64),
                    True,
                )
            if self.size == 0:
                return (
                    np.empty((self.n_channels, 0), dtype=self.data.dtype),
                    np.empty(0, dtype=np.int64),
                    False,
                )

            newest_index = int(self.source_indices[(self.write_position - 1) % self.capacity])
            oldest_index = int(self.source_indices[(self.write_position - self.size) % self.capacity])
            overrun = last_source_index < oldest_index - 1

            # first source index the caller still needs (a negative request - the
            # overrun reset - asks for everything the buffer holds)
            want_from = oldest_index if int(last_source_index) < 0 else int(last_source_index) + 1
            if want_from < oldest_index:
                want_from = oldest_index
            available = newest_index - want_from + 1
            if available <= 0:
                return (
                    np.empty((self.n_channels, 0), dtype=self.data.dtype),
                    np.empty(0, dtype=np.int64),
                    overrun,
                )
            # source indices can jump (a producer that skips), so never claim more
            # samples than the buffer actually holds
            available = min(int(available), int(self.size))
            count = available
            if max_samples is not None:
                count = min(count, max(0, int(max_samples)))
            if count <= 0:
                return (
                    np.empty((self.n_channels, 0), dtype=self.data.dtype),
                    np.empty(0, dtype=np.int64),
                    overrun,
                )

            # the selected run ends at the write head: the OLDEST `count` of the
            # available samples (which is what the mask + [:max_samples] meant)
            start = (self.write_position - int(available)) % self.capacity
            if start + count <= self.capacity:
                data = self.data[:, start : start + count].copy()
                indices = self.source_indices[start : start + count].copy()
            else:
                first = self.capacity - start
                data = np.concatenate(
                    (self.data[:, start:], self.data[:, : count - first]), axis=1
                )
                indices = np.concatenate(
                    (self.source_indices[start:], self.source_indices[: count - first])
                )
            # A producer that JUMPS (a device counter reset, or a hole in the
            # stream) leaves the ring holding two runs, and the slice above takes
            # the OLDEST samples - so it can start inside the run the caller has
            # already consumed. Delivering that stale head would make the caller
            # re-process samples it already saw AND hide the jump (its own gap
            # detection compares the first returned index with the last one it
            # had). The head is trimmed here, and when nothing is left the read is
            # reported as an overrun so the caller resets and resynchronises.
            # One integer comparison keeps the hot path free of masking.
            if indices.size and int(indices[0]) <= int(last_source_index):
                keep = indices > int(last_source_index)
                if not bool(keep.any()):
                    return (
                        np.empty((self.n_channels, 0), dtype=self.data.dtype),
                        np.empty(0, dtype=np.int64),
                        True,
                    )
                first_keep = int(np.argmax(keep))
                data = data[:, first_keep:]
                indices = indices[first_keep:]
            return data, indices, overrun

    def wait_for_new_data(self, timeout: float = 0.1) -> None:
        with self.updated:
            self.updated.wait(timeout=timeout)

    def clear(self) -> None:
        with self.updated:
            self.data.fill(0)
            self.source_indices.fill(-1)
            self.write_position = 0
            self.size = 0
            self.total_written = 0
            self.generation += 1
            self.updated.notify_all()


class _BaseAcquisitionThread(threading.Thread):
    def __init__(
        self,
        raw_buffer: RingBuffer,
        n_channels: int,
        srate: float,
        stop_event: threading.Event,
        record_queue_provider: Optional[Callable[[], Optional[queue.Queue]]] = None,
        block_observer: Optional[Callable[[RawBlockMeta, np.ndarray, np.ndarray], None]] = None,
        epoch: int = 0,
        kind: str = "device",
    ):
        super().__init__(daemon=True)
        self.raw_buffer = raw_buffer
        self.n_channels = int(n_channels)
        self.srate = float(srate)
        self.stop_event = stop_event
        self.record_queue_provider = record_queue_provider
        # The block observer is the ONLY addition to this hot path: it receives the
        # block's meta (with the single arrival stamp) plus the SAME arrays that
        # went into the ring buffer. It must not mutate them and must not raise -
        # `EEGRecorder._observe_raw_block` guards both.
        self.block_observer = block_observer
        self.epoch = int(epoch)
        self.kind = str(kind)
        self.next_source_index = 0
        self.block_seq = 0
        self.error: Optional[BaseException] = None

    def _publish_chunk(
        self,
        data: np.ndarray,
        source_indices: np.ndarray,
        *,
        seek: bool = False,
    ) -> None:
        # ONE stamp per block, taken before anything else can move the clock: it is
        # copied into the per-frame time channel and into the chunk/loop row, so
        # "when did this arrive" and "when did I predict on it" share a time base.
        stamp_wall_ns = time.time_ns()
        stamp_mono_ns = time.perf_counter_ns()

        self.raw_buffer.append(data, source_indices)

        meta = RawBlockMeta(
            seq=self.block_seq,
            epoch=self.epoch,
            kind=self.kind,
            source_first=int(source_indices[0]) if source_indices.size else -1,
            source_last=int(source_indices[-1]) if source_indices.size else -1,
            n_samples=int(source_indices.size),
            n_channels=int(data.shape[0]),
            acq_wall_ns=int(stamp_wall_ns),
            acq_mono_ns=int(stamp_mono_ns),
            seek=bool(seek),
        )
        self.block_seq += 1
        if self.block_observer is not None:
            try:
                self.block_observer(meta, data, source_indices)
            except Exception:                      # noqa: BLE001 - never kill acquisition
                pass

        record_queue = (
            self.record_queue_provider()
            if self.record_queue_provider is not None
            else None
        )
        if record_queue is not None:
            try:
                record_queue.put_nowait(
                    (data[:].copy(), source_indices[:].copy())
                )
            except queue.Full as exc:
                raise RuntimeError(
                    "Recording queue is full; disk writing cannot keep up."
                ) from exc


class _NeuracleThread(_BaseAcquisitionThread):
    """Receive and parse Neuracle TCP frames."""

    def __init__(
        self,
        raw_buffer: RingBuffer,
        n_channels: int,
        srate: float,
        stop_event: threading.Event,
        sock: socket.socket,
        recv_bytes: int,
        record_queue_provider: Optional[Callable[[], Optional[queue.Queue]]] = None,
        block_observer: Optional[Callable[[RawBlockMeta, np.ndarray, np.ndarray], None]] = None,
        epoch: int = 0,
        kind: str = "device",
        host: Optional[str] = None,
        port: Optional[int] = None,
        reconnect_seconds: float = 0.5,
        connect_timeout: float = 1.0,
        epoch_provider: Optional[Callable[[str], int]] = None,
        status_notifier: Optional[Callable[[str, str], None]] = None,
    ):
        super().__init__(
            raw_buffer,
            n_channels,
            srate,
            stop_event,
            record_queue_provider=record_queue_provider,
            block_observer=block_observer,
            epoch=epoch,
            kind=kind,
        )
        self.sock = sock
        self.recv_bytes = int(recv_bytes)
        self.byte_remainder = b""

        # A CLOSED PORT IS NOT THE END OF THE SESSION. The operator may power-cycle
        # the amplifier, restart the acquisition software, or the link may blink -
        # so when the connection ends this thread stays ON THE SAME host:port and
        # keeps trying, and the session, its record and the dashboard all survive
        # the outage and resume the moment data arrives again. With no host/port
        # (a hand-built thread, a test double) there is nothing to wait at, and the
        # old contract stands: the link ended, and this thread ends with it.
        self.host = None if host is None else str(host)
        self.port = None if port is None else int(port)
        self.reconnect_seconds = max(0.05, float(reconnect_seconds))
        self.connect_timeout = max(0.05, float(connect_timeout))
        self.epoch_provider = epoch_provider
        self.status_notifier = status_notifier
        self.reconnecting = False
        self.reconnects = 0
        self.wait_attempts = 0
        self.link_lost_wall_ns: Optional[int] = None
        self.last_link_error = ""

        # Neuracle frame: n_channels - 1 float32 values + 1 uint32 trigger.
        fields = [(f"eeg_{i}", "<f4") for i in range(self.n_channels - 1)]
        fields.append(("trigger", "<u4"))
        self.frame_dtype = np.dtype(fields)
        self.frame_bytes = self.frame_dtype.itemsize

    @property
    def waiting_for_device(self) -> bool:
        """True while the port is closed and this thread is waiting at it."""
        return bool(self.reconnecting)

    @property
    def waiting_seconds(self) -> float:
        if not self.reconnecting or self.link_lost_wall_ns is None:
            return 0.0
        return max(0.0, (time.time_ns() - int(self.link_lost_wall_ns)) / 1e9)

    def close_socket(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass

    def _parse(self, raw: bytes) -> np.ndarray:
        raw = self.byte_remainder + raw
        n_complete_bytes = len(raw) - len(raw) % self.frame_bytes
        complete = raw[:n_complete_bytes]
        self.byte_remainder = raw[n_complete_bytes:]

        if not complete:
            return np.empty((self.n_channels, 0), dtype=np.float64)

        records = np.frombuffer(complete, dtype=self.frame_dtype)
        n_samples = records.shape[0]
        output = np.empty((self.n_channels, n_samples), dtype=np.float64)

        for channel in range(self.n_channels - 1):
            output[channel] = records[f"eeg_{channel}"]

        output[-1] = records["trigger"].astype(np.float64)
        return output

    def _wait_for_port(self, reason: str) -> bool:
        """Stay at the port until the device answers again.

        Returns True once a new connection is up (the caller keeps receiving),
        False when this thread must end: the session is stopping, or there is no
        address to wait at. Nothing here touches the buffer, the record or the
        worker - an outage is simply a period in which no block is published, and
        the pipeline reports it as a gap exactly as it does for any other stall.
        """
        if self.host is None or self.port is None:
            self.error = ConnectionError("The Neuracle TCP connection was closed.")
            return False
        self.close_socket()
        self.reconnecting = True
        self.link_lost_wall_ns = time.time_ns()
        self._notify("disconnect", f"{reason}; waiting at {self.host}:{self.port}")
        attempt = 0
        while not self.stop_event.is_set():
            attempt += 1
            self.wait_attempts = attempt
            try:
                sock = socket.create_connection(
                    (self.host, self.port), timeout=self.connect_timeout)
            except OSError as exc:
                self.last_link_error = repr(exc)
                # Wait, but stay responsive to disconnect()/stop.
                self.stop_event.wait(self.reconnect_seconds)
                continue
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(0.5)
            self.sock = sock
            self.byte_remainder = b""
            # The new producer numbers its samples from 0 again, so this is a NEW
            # source: it needs a new epoch and no leftover index bookkeeping.
            self.next_source_index = 0
            self.reconnecting = False
            self.wait_attempts = 0
            self.reconnects += 1
            if self.epoch_provider is not None:
                self.epoch = int(self.epoch_provider(f"back after {attempt} attempt(s)"))
            self._notify(
                "reconnect",
                f"device {self.host}:{self.port} is back after {attempt} attempt(s), "
                f"epoch {self.epoch}",
            )
            return True
        return False

    def _notify(self, code: str, text: str) -> None:
        if self.status_notifier is None:
            return
        try:
            self.status_notifier(code, text)
        except Exception:                                        # noqa: BLE001
            pass

    def run(self) -> None:
        try:
            self.sock.settimeout(0.5)
            while not self.stop_event.is_set():
                try:
                    raw = self.sock.recv(self.recv_bytes)
                except socket.timeout:
                    continue
                except OSError as exc:
                    # Reset/aborted link: the same "the port went away" case.
                    self.last_link_error = repr(exc)
                    raw = b""

                if not raw:
                    if not self._wait_for_port("the device closed the connection"):
                        break
                    continue

                data = self._parse(raw)
                n_samples = data.shape[1]
                if n_samples == 0:
                    continue

                indices = np.arange(
                    self.next_source_index,
                    self.next_source_index + n_samples,
                    dtype=np.int64,
                )
                self._publish_chunk(data, indices)
                self.next_source_index += n_samples

        except BaseException as exc:
            if not self.stop_event.is_set():
                self.error = exc
        finally:
            self.close_socket()


class _SyntheticThread(_BaseAcquisitionThread):
    """Generate independent narrow-band noise for each EEG channel."""

    def __init__(
        self,
        raw_buffer: RingBuffer,
        n_channels: int,
        n_eeg: int,
        srate: float,
        stop_event: threading.Event,
        chunk_seconds: float,
        band_hz: Tuple[float, float],
        seed: Optional[int],
        record_queue_provider: Optional[Callable[[], Optional[queue.Queue]]] = None,
        block_observer: Optional[Callable[[RawBlockMeta, np.ndarray, np.ndarray], None]] = None,
        epoch: int = 0,
        kind: str = "synthetic",
    ):
        super().__init__(
            raw_buffer,
            n_channels,
            srate,
            stop_event,
            record_queue_provider=record_queue_provider,
            block_observer=block_observer,
            epoch=epoch,
            kind=kind,
        )

        if not 0 < n_eeg <= n_channels:
            raise ValueError("n_eeg must satisfy 0 < n_eeg <= n_channels.")
        if not 0 < band_hz[0] < band_hz[1] < srate / 2:
            raise ValueError("band_hz must lie strictly inside (0, Nyquist).")

        self.n_eeg = int(n_eeg)
        self.chunk_samples = max(1, int(round(chunk_seconds * srate)))
        self.rng = np.random.default_rng(seed)

        self.sos = butter(
            4,
            band_hz,
            btype="bandpass",
            fs=srate,
            output="sos",
        )
        self.zi = np.zeros(
            (self.sos.shape[0], self.n_eeg, 2),
            dtype=np.float64,
        )

        # Give channels different stable amplitudes.
        self.channel_scale = self.rng.uniform(5.0, 25.0, size=(self.n_eeg, 1))

    def run(self) -> None:
        next_deadline = time.perf_counter()

        try:
            while not self.stop_event.is_set():
                white = self.rng.standard_normal(
                    (self.n_eeg, self.chunk_samples)
                )
                eeg, self.zi = sosfilt(
                    self.sos,
                    white,
                    axis=-1,
                    zi=self.zi,
                )
                eeg = eeg * self.channel_scale

                data = np.zeros(
                    (self.n_channels, self.chunk_samples),
                    dtype=np.float64,
                )
                data[: self.n_eeg] = eeg

                indices = np.arange(
                    self.next_source_index,
                    self.next_source_index + self.chunk_samples,
                    dtype=np.int64,
                )
                self._publish_chunk(data, indices)
                self.next_source_index += self.chunk_samples

                next_deadline += self.chunk_samples / self.srate
                sleep_time = next_deadline - time.perf_counter()
                if sleep_time > 0:
                    self.stop_event.wait(sleep_time)
                else:
                    # Avoid accumulating timing error after a temporary delay.
                    next_deadline = time.perf_counter()

        except BaseException as exc:
            if not self.stop_event.is_set():
                self.error = exc


class _HDF5WriterThread(threading.Thread):
    """Append raw EEG, trigger, and source indices to an HDF5 file."""

    _STOP = object()

    def __init__(
        self,
        file_path: str,
        data_queue: queue.Queue,
        n_eeg: int,
        srate: float,
        write_chunk_seconds: float,
        stop_event: threading.Event,
    ):
        super().__init__(daemon=True)
        if write_chunk_seconds <= 0:
            raise ValueError("write_chunk_seconds must be positive.")

        self.file_path = str(file_path)
        self.data_queue = data_queue
        self.n_eeg = int(n_eeg)
        self.srate = float(srate)
        self.write_chunk_samples = max(
            1,
            int(round(write_chunk_seconds * srate)),
        )
        self.stop_event = stop_event
        self.error: Optional[BaseException] = None
        self.total_written = 0

    def request_stop(self) -> None:
        self.stop_event.set()
        try:
            self.data_queue.put_nowait(self._STOP)
        except queue.Full:
            # The consumer will observe stop_event after draining queued chunks.
            pass

    @staticmethod
    def _append_dataset(dataset, values: np.ndarray) -> None:
        old_size = dataset.shape[-1]
        new_size = old_size + values.shape[-1]
        dataset.resize(new_size, axis=dataset.ndim - 1)
        dataset[..., old_size:new_size] = values

    def _write_block(
        self,
        h5_file: h5py.File,
        eeg_dataset,
        trigger_dataset,
        index_dataset,
        data_parts: list[np.ndarray],
        index_parts: list[np.ndarray],
        n_samples: int,
    ) -> int:
        if n_samples <= 0:
            return 0

        data = np.concatenate(data_parts, axis=1)
        indices = np.concatenate(index_parts)

        self._append_dataset(
            eeg_dataset,
            data[: self.n_eeg].astype(np.float32, copy=False),
        )
        self._append_dataset(
            trigger_dataset,
            data[-1].astype(np.uint32, copy=False),
        )
        self._append_dataset(index_dataset, indices.astype(np.int64, copy=False))

        h5_file.attrs["n_samples"] = int(eeg_dataset.shape[1])
        h5_file.flush()

        # Request an OS-level flush as well. This reduces, but cannot eliminate,
        # loss caused by a full process or machine crash.
        try:
            handle = h5_file.id.get_vfd_handle()
            if isinstance(handle, tuple):
                handle = handle[0]
            if isinstance(handle, int):
                os.fsync(handle)
        except (AttributeError, OSError, TypeError, ValueError):
            pass

        self.total_written += n_samples
        data_parts.clear()
        index_parts.clear()
        return 0

    def run(self) -> None:
        data_parts: list[np.ndarray] = []
        index_parts: list[np.ndarray] = []
        accumulated = 0

        try:
            parent = os.path.dirname(os.path.abspath(self.file_path))
            os.makedirs(parent, exist_ok=True)

            with h5py.File(self.file_path, "w", libver="latest") as h5_file:
                h5_file.attrs["format"] = "Neuracle EEG recording"
                h5_file.attrs["srate"] = self.srate
                h5_file.attrs["n_eeg"] = self.n_eeg
                h5_file.attrs["trigger_channel_position"] = "last"
                h5_file.attrs["start_time_unix"] = time.time()
                h5_file.attrs["complete"] = False
                h5_file.attrs["n_samples"] = 0

                eeg_dataset = h5_file.create_dataset(
                    "eeg",
                    shape=(self.n_eeg, 0),
                    maxshape=(self.n_eeg, None),
                    chunks=(self.n_eeg, self.write_chunk_samples),
                    dtype=np.float32,
                    compression=None,
                )
                trigger_dataset = h5_file.create_dataset(
                    "trigger",
                    shape=(0,),
                    maxshape=(None,),
                    chunks=(self.write_chunk_samples,),
                    dtype=np.uint32,
                    compression=None,
                )
                index_dataset = h5_file.create_dataset(
                    "source_indices",
                    shape=(0,),
                    maxshape=(None,),
                    chunks=(self.write_chunk_samples,),
                    dtype=np.int64,
                    compression=None,
                )

                while True:
                    try:
                        item = self.data_queue.get(timeout=0.2)
                    except queue.Empty:
                        if self.stop_event.is_set() and self.data_queue.empty():
                            break
                        continue

                    try:
                        if item is self._STOP:
                            if self.stop_event.is_set() and self.data_queue.empty():
                                break
                            continue

                        data, indices = item
                        data_parts.append(data)
                        index_parts.append(indices)
                        accumulated += data.shape[1]

                        if accumulated >= self.write_chunk_samples:
                            accumulated = self._write_block(
                                h5_file,
                                eeg_dataset,
                                trigger_dataset,
                                index_dataset,
                                data_parts,
                                index_parts,
                                accumulated,
                            )
                    finally:
                        self.data_queue.task_done()

                accumulated = self._write_block(
                    h5_file,
                    eeg_dataset,
                    trigger_dataset,
                    index_dataset,
                    data_parts,
                    index_parts,
                    accumulated,
                )
                h5_file.attrs["complete"] = True
                h5_file.attrs["end_time_unix"] = time.time()
                h5_file.flush()

        except BaseException as exc:
            self.error = exc


@dataclass(frozen=True)
class ReplayStatus:
    current_sample: int
    total_samples: int
    srate: float
    playing: bool

    @property
    def current_seconds(self) -> float:
        return float(self.current_sample) / float(self.srate)

    @property
    def total_seconds(self) -> float:
        return float(self.total_samples) / float(self.srate)


class _ReplayThread(_BaseAcquisitionThread):
    """Replay a recorded HDF5 file as a live-like acquisition source."""

    def __init__(
        self,
        raw_buffer: RingBuffer,
        n_channels: int,
        n_eeg: int,
        srate: float,
        stop_event: threading.Event,
        h5_path: str | Path,
        chunk_seconds: float,
        record_queue_provider: Optional[Callable[[], Optional[queue.Queue]]] = None,
        seek_burst_samples: int = 0,
        block_observer: Optional[Callable[[RawBlockMeta, np.ndarray, np.ndarray], None]] = None,
        epoch: int = 0,
        kind: str = "replay",
    ):
        super().__init__(
            raw_buffer,
            n_channels,
            srate,
            stop_event,
            record_queue_provider=record_queue_provider,
            block_observer=block_observer,
            epoch=epoch,
            kind=kind,
        )
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive.")
        self.n_eeg = int(n_eeg)
        self.h5_path = Path(h5_path)
        self.chunk_seconds = float(chunk_seconds)
        self.chunk_samples = max(1, int(round(self.chunk_seconds * self.srate)))
        self.seek_burst_samples = max(0, int(seek_burst_samples))
        self._control = threading.Condition()
        self._current_sample = 0
        self._total_samples = 0
        self._playing = True
        self._burst_remaining = 0

    def status(self) -> ReplayStatus:
        with self._control:
            return ReplayStatus(
                current_sample=int(self._current_sample),
                total_samples=int(self._total_samples),
                srate=float(self.srate),
                playing=bool(self._playing),
            )

    def set_playing(self, playing: bool) -> None:
        with self._control:
            self._playing = bool(playing)
            self._control.notify_all()

    def toggle_playing(self) -> bool:
        with self._control:
            self._playing = not self._playing
            self._control.notify_all()
            return self._playing

    def seek_sample(self, sample: int, burst_samples: Optional[int] = None) -> None:
        with self._control:
            sample = max(0, min(int(sample), int(self._total_samples)))
            self._current_sample = sample
            # The replay source NEVER touches the ring caches: like the live
            # device it only APPENDS chunks with monotonically increasing
            # source indices. A seek only repositions the file read; the raw /
            # preprocessed buffers keep their content and consumers keep their
            # incremental reads (offsets, preprocessed queues, sliding windows)
            # intact. The optional burst below just publishes the first samples
            # from the new position faster so sliding windows are rebuilt
            # quickly; it is still a plain incremental append.
            if burst_samples is None:
                burst_samples = self.seek_burst_samples
            self._burst_remaining = max(0, int(burst_samples))
            # Pacing restarts from now so a burst is not throttled by the old
            # deadline.
            self._control.notify_all()

    def run(self) -> None:
        try:
            with h5py.File(self.h5_path, "r") as h5_file:
                eeg_dataset = h5_file["eeg"]
                trigger_dataset = h5_file.get("trigger")
                total_samples = int(eeg_dataset.shape[1])
                if int(eeg_dataset.shape[0]) != self.n_eeg:
                    raise ValueError(
                        f"Replay file EEG channel count mismatch: expected {self.n_eeg}, got {int(eeg_dataset.shape[0])}."
                    )
                if trigger_dataset is not None and int(trigger_dataset.shape[0]) != total_samples:
                    raise ValueError("Replay file trigger length does not match EEG length.")
                with self._control:
                    self._total_samples = total_samples

                next_deadline = time.perf_counter()
                while not self.stop_event.is_set():
                    with self._control:
                        while not self.stop_event.is_set() and not self._playing:
                            self._control.wait(timeout=0.1)
                        if self.stop_event.is_set():
                            break
                        start = int(self._current_sample)
                        if start >= self._total_samples:
                            self._playing = False
                            self._control.wait(timeout=0.1)
                            next_deadline = time.perf_counter()
                            continue
                        stop = min(start + self.chunk_samples, self._total_samples)
                        self._current_sample = stop
                        bursting = self._burst_remaining > 0

                    eeg = np.asarray(eeg_dataset[:, start:stop], dtype=np.float64)
                    trigger = (
                        np.asarray(trigger_dataset[start:stop], dtype=np.float64)
                        if trigger_dataset is not None
                        else np.zeros(stop - start, dtype=np.float64)
                    )
                    chunk = np.zeros((self.n_channels, stop - start), dtype=np.float64)
                    chunk[: self.n_eeg] = eeg
                    chunk[-1] = trigger
                    indices = np.arange(
                        self.next_source_index,
                        self.next_source_index + (stop - start),
                        dtype=np.int64,
                    )
                    self._publish_chunk(chunk, indices)
                    self.next_source_index += stop - start

                    if bursting:
                        # Seek burst: publish the next chunks immediately (no
                        # pacing) so consumers can refill their decode window
                        # with the NEW position within one read instead of the
                        # stale window sliding out over ~n_p/srate seconds.
                        self._burst_remaining -= stop - start
                        next_deadline = time.perf_counter()
                        continue

                    next_deadline += float(stop - start) / float(self.srate)
                    remaining = next_deadline - time.perf_counter()
                    while remaining > 0 and not self.stop_event.is_set():
                        time.sleep(min(remaining, 0.02))
                        remaining = next_deadline - time.perf_counter()
        except BaseException as exc:
            if not self.stop_event.is_set():
                self.error = exc


@dataclass(frozen=True)
class PreprocessConfig:
    lowpass_hz: float
    downsample_factor: int
    highpass_hz: Optional[float]
    iir_order: int
    output_srate: float


class _PreprocessThread(threading.Thread):
    """Incrementally process raw EEG while preserving causal IIR states."""

    def __init__(
        self,
        raw_buffer: RingBuffer,
        output_buffer: RingBuffer,
        n_eeg: int,
        raw_srate: float,
        config: PreprocessConfig,
        stop_event: threading.Event,
    ):
        super().__init__(daemon=True)
        self.raw_buffer = raw_buffer
        self.output_buffer = output_buffer
        self.n_eeg = int(n_eeg)
        self.raw_srate = float(raw_srate)
        self.config = config
        self.stop_event = stop_event
        self.error: Optional[BaseException] = None

        self.lowpass_sos = butter(
            config.iir_order,
            config.lowpass_hz,
            btype="lowpass",
            fs=raw_srate,
            output="sos",
        )
        self.lowpass_zi = np.zeros(
            (self.lowpass_sos.shape[0], self.n_eeg, 2),
            dtype=np.float64,
        )

        if config.highpass_hz is None:
            self.highpass_sos = None
            self.highpass_zi = None
        else:
            self.highpass_sos = butter(
                config.iir_order,
                config.highpass_hz,
                btype="highpass",
                fs=config.output_srate,
                output="sos",
            )
            self.highpass_zi = np.zeros(
                (self.highpass_sos.shape[0], self.n_eeg, 2),
                dtype=np.float64,
            )

        self.last_raw_source_index = -1

    def _reset_filter_states(self) -> None:
        self.lowpass_zi.fill(0.0)
        if self.highpass_zi is not None:
            self.highpass_zi.fill(0.0)

    def run(self) -> None:
        factor = self.config.downsample_factor

        try:
            while not self.stop_event.is_set():
                raw, raw_indices, overrun = self.raw_buffer.read_after_source_index(
                    self.last_raw_source_index
                )

                if raw.shape[1] == 0:
                    self.raw_buffer.wait_for_new_data(timeout=0.1)
                    continue

                if overrun:
                    # Continuity is broken because raw samples were overwritten.
                    # Reset states rather than silently filtering across a gap.
                    self._reset_filter_states()

                eeg = raw[: self.n_eeg]
                lowpassed, self.lowpass_zi = sosfilt(
                    self.lowpass_sos,
                    eeg,
                    axis=-1,
                    zi=self.lowpass_zi,
                )

                # Use absolute raw indices so the decimation phase remains stable
                # across arbitrary TCP chunk boundaries.
                keep = raw_indices % factor == 0
                downsampled = lowpassed[:, keep]
                output_indices = raw_indices[keep]

                if downsampled.shape[1] > 0:
                    if self.highpass_sos is not None:
                        downsampled, self.highpass_zi = sosfilt(
                            self.highpass_sos,
                            downsampled,
                            axis=-1,
                            zi=self.highpass_zi,
                        )

                    self.output_buffer.append(downsampled, output_indices)

                self.last_raw_source_index = int(raw_indices[-1])

        except BaseException as exc:
            if not self.stop_event.is_set():
                self.error = exc


class EEGRecorder:
    """
    Minimal Neuracle EEG recorder with optional real-time preprocessing.

    Parameters
    ----------
    n_channels:
        Total streamed channels, including any trigger/non-EEG channels.
        For the common Neuracle protocol, the final channel is uint32 trigger.
    n_eeg:
        Number of EEG channels from the beginning of the stream.
    srate:
        Raw sampling rate in Hz.
    buffer_seconds:
        Raw ring-buffer duration.
    host, port:
        Neuracle TCP server address.
    """

    def __init__(
        self,
        n_channels: int = 60,
        n_eeg: int = 59,
        srate: float = 1000.0,
        buffer_seconds: float = 360.0,
        host: str = "127.0.0.1",
        port: int = 8712,
    ):
        if not 0 < n_eeg <= n_channels:
            raise ValueError("n_eeg must satisfy 0 < n_eeg <= n_channels.")
        if srate <= 0:
            raise ValueError("srate must be positive.")
        if buffer_seconds <= 0:
            raise ValueError("buffer_seconds must be positive.")

        self.n_channels = int(n_channels)
        self.n_eeg = int(n_eeg)
        self.srate = float(srate)
        self.buffer_seconds = float(buffer_seconds)
        self.host = host
        self.port = int(port)

        self.raw_buffer = RingBuffer(
            self.n_channels,
            max(1, int(round(self.buffer_seconds * self.srate))),
        )

        self._acquisition_stop = threading.Event()
        self._acquisition_thread: Optional[_BaseAcquisitionThread] = None

        self._preprocess_stop = threading.Event()
        self._preprocess_thread: Optional[_PreprocessThread] = None
        self.preprocessed_buffer: Optional[RingBuffer] = None
        self.preprocess_config: Optional[PreprocessConfig] = None

        self._record_queue: Optional[queue.Queue] = None
        self._record_stop = threading.Event()
        self._record_thread: Optional[_HDF5WriterThread] = None

        # ---------------------------------------------------------------- time
        # One arrival stamp per published block, plus a small session log. These
        # are READ-ONLY extras for recorders of the session file: the armed
        # `data.h5` path (record queue + HDF5 writer) is untouched by them.
        self._epoch = 0
        self._epoch_used = False
        self._block_seq = 0
        self._raw_block_meta: Deque[RawBlockMeta] = deque(maxlen=8192)
        self._raw_block_sinks: Dict[int, Callable[[RawBlockMeta, np.ndarray, np.ndarray], None]] = {}
        self._sink_token = 0
        self._status_events: Deque[Tuple[int, int, str, str]] = deque(maxlen=4096)
        self._status_lock = threading.RLock()

    # ------------------------------------------------------------- block times
    def _observe_raw_block(
        self,
        meta: RawBlockMeta,
        data: np.ndarray,
        source_indices: np.ndarray,
    ) -> None:
        """Producer-side hook: keep the block's arrival stamp and fan it out.

        Runs on the acquisition thread, so it does nothing but a deque append and
        a dispatch. Sinks receive the SAME arrays that were published (no copy
        here): a sink that keeps data must copy it itself, and a sink that raises
        is disconnected from the caller's perspective - the exception is swallowed
        so acquisition can never be killed by a writer.
        """
        self._raw_block_meta.append(meta)
        if not self._raw_block_sinks:
            return
        for sink in list(self._raw_block_sinks.values()):
            try:
                sink(meta, data, source_indices)
            except Exception:                                    # noqa: BLE001
                pass

    def add_raw_block_sink(
        self,
        sink: Callable[[RawBlockMeta, np.ndarray, np.ndarray], None],
    ) -> int:
        """Register a consumer of every published raw block; returns its token."""
        with self._status_lock:
            self._sink_token += 1
            self._raw_block_sinks[self._sink_token] = sink
            return self._sink_token

    def remove_raw_block_sink(self, token: Optional[int]) -> None:
        if token is None:
            return
        with self._status_lock:
            self._raw_block_sinks.pop(int(token), None)

    def raw_block_meta_since(self, after_seq: int) -> List[RawBlockMeta]:
        """The blocks published after a producer-side sequence number.

        `seq` is per epoch and monotone inside it; a new epoch restarts it, so
        callers compare `(epoch, seq)`. The deque is bounded, so a caller that
        fell far behind simply gets the oldest blocks still held.
        """
        with self._status_lock:
            blocks = list(self._raw_block_meta)
        return [block for block in blocks if block.seq > int(after_seq)]

    def latest_block_meta(self) -> Optional[RawBlockMeta]:
        with self._status_lock:
            return self._raw_block_meta[-1] if self._raw_block_meta else None

    @property
    def epoch(self) -> int:
        """The acquisition-source epoch: +1 for every connect / source switch."""
        return int(self._epoch)

    # -------------------------------------------------------------- session log
    def _note_status(self, code: str, text: str) -> None:
        with self._status_lock:
            self._status_events.append(
                (time.time_ns(), time.perf_counter_ns(), str(code), str(text))
            )

    def status_events(self) -> List[Tuple[int, int, str, str]]:
        """The session-level events (connect, switch, disconnect) with stamps."""
        with self._status_lock:
            return list(self._status_events)

    def status_events_since(self, after_mono_ns: int = -1) -> List[Tuple[int, int, str, str]]:
        """The status events newer than a cursor from a previous call.

        `after_mono_ns` is the `mono_ns` of the last event the caller already has
        (-1 = everything), which is monotone and therefore a safe cursor for a
        BOUNDED deque - an index would silently re-deliver or skip after the deque
        rolls. Scans from the newest end, so a poll that finds nothing is O(1).
        """
        with self._status_lock:
            events: List[Tuple[int, int, str, str]] = []
            for event in reversed(self._status_events):
                if int(event[1]) <= int(after_mono_ns):
                    break
                events.append(event)
            events.reverse()
            return events

    @property
    def connected(self) -> bool:
        return (
            self._acquisition_thread is not None
            and self._acquisition_thread.is_alive()
        )

    @property
    def recording(self) -> bool:
        return self._record_thread is not None and self._record_thread.is_alive()

    @property
    def preprocess_running(self) -> bool:
        return (
            self._preprocess_thread is not None
            and self._preprocess_thread.is_alive()
        )

    def _check_acquisition_error(self) -> None:
        thread = self._acquisition_thread
        if thread is not None and thread.error is not None:
            raise RuntimeError("Acquisition thread failed.") from thread.error

    def _check_preprocess_error(self) -> None:
        thread = self._preprocess_thread
        if thread is not None and thread.error is not None:
            raise RuntimeError("Preprocessing thread failed.") from thread.error

    def _check_recording_error(self) -> None:
        thread = self._record_thread
        if thread is not None and thread.error is not None:
            raise RuntimeError("HDF5 writer thread failed.") from thread.error

    def _get_record_queue(self) -> Optional[queue.Queue]:
        return self._record_queue

    def _clear_runtime_buffers(self) -> None:
        self.raw_buffer.clear()
        if self.preprocessed_buffer is not None:
            self.preprocessed_buffer.clear()

    def _new_epoch(self) -> int:
        """Open a new acquisition epoch (0 for the very first source).

        EVERY producer numbers its samples from 0, and a reconnect/dropped socket
        builds a new producer - so a fresh source needs a new epoch id AND cleared
        buffers, or two different samples would share one (epoch, index) identity
        and the realtime readers would pair samples that never were neighbours.
        """
        with self._status_lock:
            if self._epoch_used:
                self._epoch += 1
                self._clear_runtime_buffers()
                self._note_status("switch", f"epoch {self._epoch}")
            self._epoch_used = True
            return int(self._epoch)

    def _epoch_for_reconnected_source(self, reason: str) -> int:
        """The epoch a reconnected socket publishes under.

        A socket that answers again is a NEW producer: it numbers its samples
        from 0, so sharing an epoch with the old one would give two different
        samples one identity - and the realtime readers would wait for the new
        counter to pass the old one, i.e. minutes of silence. Same rule (and the
        same buffer clear) as a manual connect, which is why it reuses
        `_new_epoch`.
        """
        epoch = self._new_epoch()
        self._note_status(
            "reconnect", f"device {self.host}:{self.port} back ({reason}) epoch {epoch}")
        return epoch

    @property
    def waiting_for_device(self) -> bool:
        """True while the port is closed and acquisition is waiting at it.

        The session is NOT over in that state: the acquisition thread is alive
        and will resume the same session as soon as the device answers.
        """
        return bool(getattr(self._acquisition_thread, "waiting_for_device", False))

    @property
    def waiting_seconds(self) -> float:
        getter = getattr(self._acquisition_thread, "waiting_seconds", None)
        return float(getter()) if callable(getter) else 0.0

    @property
    def reconnect_count(self) -> int:
        return int(getattr(self._acquisition_thread, "reconnects", 0))

    def _retire_acquisition(self) -> None:
        """Close the previous source before a NEW connect.

        A dropped TCP link leaves the thread object behind while `connected` is
        already False, and the next `connect()` builds a fresh thread that numbers
        its samples from 0 - in a ring buffer that may still hold the old source's
        (much larger) indices. That is the mix-up `_replace_acquisition` already
        guards against for a hot swap; a plain reconnect needs the same treatment:
        join the dead thread, open a NEW EPOCH, and clear the buffers.
        """
        thread = self._acquisition_thread
        if thread is not None and thread.is_alive():
            raise RuntimeError("Recorder is already connected.")
        if thread is None:
            return
        if isinstance(thread, _NeuracleThread):
            thread.close_socket()
        self._acquisition_thread = None
        self._note_status("disconnect", "previous source ended")

    def connect(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: float = 3.0,
    ) -> None:
        """Connect to Neuracle and immediately start background acquisition."""
        if self.connected:
            raise RuntimeError("Recorder is already connected.")
        self._retire_acquisition()
        self._new_epoch()

        host = self.host if host is None else host
        port = self.port if port is None else int(port)

        sock = socket.create_connection((host, port), timeout=timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        self._acquisition_stop.clear()
        recv_bytes = max(
            4096,
            int(round(0.1 * self.srate * self.n_channels * 4)),
        )
        thread = _NeuracleThread(
            raw_buffer=self.raw_buffer,
            n_channels=self.n_channels,
            srate=self.srate,
            stop_event=self._acquisition_stop,
            sock=sock,
            recv_bytes=recv_bytes,
            record_queue_provider=self._get_record_queue,
            block_observer=self._observe_raw_block,
            epoch=self._epoch,
            kind="device",
            # A closed port is waited out AT THE PORT (see _NeuracleThread): the
            # session keeps running, and the socket that finally answers is a NEW
            # producer that numbers its samples from 0 - so it gets a new epoch.
            host=host,
            port=port,
            epoch_provider=self._epoch_for_reconnected_source,
            status_notifier=self._note_status,
        )
        self._acquisition_thread = thread
        thread.start()
        self._note_status("connect", f"device {host}:{port} epoch {self._epoch}")

    def connect_synthetic(
        self,
        band_hz: Tuple[float, float] = (8.0, 12.0),
        chunk_seconds: float = 0.04,
        seed: Optional[int] = None,
    ) -> None:
        """Start a real-time synthetic narrow-band-noise source."""
        if self.connected:
            raise RuntimeError("Recorder is already connected.")
        self._retire_acquisition()
        self._new_epoch()
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive.")

        self._acquisition_stop.clear()
        thread = _SyntheticThread(
            raw_buffer=self.raw_buffer,
            n_channels=self.n_channels,
            n_eeg=self.n_eeg,
            srate=self.srate,
            stop_event=self._acquisition_stop,
            chunk_seconds=chunk_seconds,
            band_hz=band_hz,
            seed=seed,
            record_queue_provider=self._get_record_queue,
            block_observer=self._observe_raw_block,
            epoch=self._epoch,
            kind="synthetic",
        )
        self._acquisition_thread = thread
        thread.start()
        self._note_status("connect", f"synthetic epoch {self._epoch}")

    def connect_replay(
        self,
        h5_path: str | Path,
        chunk_seconds: float = 0.04,
    ) -> None:
        """Replay a recorded HDF5 file as a live acquisition source."""
        if self.connected:
            raise RuntimeError("Recorder is already connected.")
        self._retire_acquisition()
        self._new_epoch()
        self._acquisition_stop.clear()
        thread = _ReplayThread(
            raw_buffer=self.raw_buffer,
            n_channels=self.n_channels,
            n_eeg=self.n_eeg,
            srate=self.srate,
            stop_event=self._acquisition_stop,
            h5_path=h5_path,
            chunk_seconds=chunk_seconds,
            record_queue_provider=self._get_record_queue,
            block_observer=self._observe_raw_block,
            epoch=self._epoch,
            kind="replay",
        )
        self._acquisition_thread = thread
        thread.start()
        self._note_status("connect", f"replay {Path(h5_path).name} epoch {self._epoch}")

    @property
    def replaying(self) -> bool:
        return isinstance(self._acquisition_thread, _ReplayThread) and self.connected

    def replay_status(self) -> Optional[ReplayStatus]:
        thread = self._acquisition_thread
        if not isinstance(thread, _ReplayThread):
            return None
        self._check_acquisition_error()
        return thread.status()

    def replay_set_playing(self, playing: bool) -> None:
        thread = self._acquisition_thread
        if not isinstance(thread, _ReplayThread):
            raise RuntimeError("Replay is not active.")
        thread.set_playing(playing)

    def replay_toggle_playing(self) -> bool:
        thread = self._acquisition_thread
        if not isinstance(thread, _ReplayThread):
            raise RuntimeError("Replay is not active.")
        return thread.toggle_playing()

    def replay_seek_seconds(self, seconds: float, burst_seconds: float = 0.0) -> None:
        thread = self._acquisition_thread
        if not isinstance(thread, _ReplayThread):
            raise RuntimeError("Replay is not active.")
        sample = int(round(max(0.0, float(seconds)) * float(self.srate)))
        burst = int(round(float(burst_seconds) * float(self.srate)))
        thread.seek_sample(sample, burst_samples=burst)

    def _replace_acquisition(self, thread: _BaseAcquisitionThread) -> None:
        """Stop the current acquisition source and start ``thread`` on the SAME
        raw buffer, keeping the preprocessing pipeline (and the recorder
        instance itself) alive. Used by merge replay to hot-swap between the
        live EEG source and a replay without breaking running paradigms.

        Unlike a replay SEEK (which only repositions the same source and never
        touches the caches), a source SWITCH is a device discontinuity: both
        sources number their chunks from virtual index 0, so the buffers are
        cleared to avoid mixing two sources' samples with overlapping indices.
        """
        self._acquisition_stop.set()
        old = self._acquisition_thread
        if isinstance(old, _NeuracleThread):
            old.close_socket()
        if old is not None and old.is_alive():
            old.join(timeout=2.0)
        self._acquisition_thread = None
        self._clear_runtime_buffers()
        self._acquisition_stop.clear()
        # A source switch is a new EPOCH: both sources number their samples from
        # 0, so (epoch, source_index) - not source_index alone - identifies a
        # sample. Consumers that track time read the epoch from the block meta.
        thread.epoch = self._new_epoch()
        self._acquisition_thread = thread
        thread.start()
        self._note_status("switch", f"{thread.kind} epoch {self._epoch}")

    def switch_to_replay(
        self,
        h5_path: str | Path,
        chunk_seconds: float = 0.04,
    ) -> None:
        """Hot-swap the LIVE acquisition source for a replay of a recorded H5
        file on this recorder (merge replay). The preprocess pipeline and all
        consumers keep running; the replay thread paces itself with this
        recorder's srate, so the file must have been recorded at ``self.srate``.
        """
        if self.recording:
            raise RuntimeError("Cannot switch to replay while recording is active.")
        thread = _ReplayThread(
            raw_buffer=self.raw_buffer,
            n_channels=self.n_channels,
            n_eeg=self.n_eeg,
            srate=self.srate,
            stop_event=self._acquisition_stop,
            h5_path=h5_path,
            chunk_seconds=chunk_seconds,
            record_queue_provider=self._get_record_queue,
            block_observer=self._observe_raw_block,
            epoch=self._epoch,
            kind="replay",
        )
        self._replace_acquisition(thread)

    def switch_to_live(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        synthetic: bool = False,
        band_hz: Tuple[float, float] = (8.0, 12.0),
        chunk_seconds: float = 0.04,
        seed: Optional[int] = None,
    ) -> None:
        """Hot-swap the replay source back to the LIVE EEG (or synthetic)
        source on this recorder (merge replay). The TCP socket is created
        BEFORE the swap so a failed live connect leaves the replay running."""
        if synthetic:
            thread = _SyntheticThread(
                raw_buffer=self.raw_buffer,
                n_channels=self.n_channels,
                n_eeg=self.n_eeg,
                srate=self.srate,
                stop_event=self._acquisition_stop,
                chunk_seconds=chunk_seconds,
                band_hz=band_hz,
                seed=seed,
                record_queue_provider=self._get_record_queue,
                block_observer=self._observe_raw_block,
                epoch=self._epoch,
                kind="synthetic",
            )
        else:
            host = self.host if host is None else host
            port = self.port if port is None else int(port)
            sock = socket.create_connection((host, port), timeout=3.0)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            recv_bytes = max(
                4096,
                int(round(0.1 * self.srate * self.n_channels * 4)),
            )
            thread = _NeuracleThread(
                raw_buffer=self.raw_buffer,
                n_channels=self.n_channels,
                srate=self.srate,
                stop_event=self._acquisition_stop,
                block_observer=self._observe_raw_block,
                kind="device",
                sock=sock,
                recv_bytes=recv_bytes,
                record_queue_provider=self._get_record_queue,
            )
        self._replace_acquisition(thread)


    def start_recording(
        self,
        file_path: str,
        write_chunk_seconds: float = 5.0,
        queue_seconds: float = 30.0,
    ) -> None:
        """
        Start raw HDF5 recording before connect() or connect_synthetic().

        Only the first `n_eeg` EEG channels and the final trigger channel are
        stored. HDF5 datasets are extendable and chunked, but not compressed.
        """
        if self.connected and not self.replaying:
            raise RuntimeError(
                "Call start_recording() before connect() so no initial data is missed."
            )
        # During REPLAY the source is already connected, but the "no initial
        # data" rationale does not apply: replay chunks are virtual and can be
        # re-sought, so arming a recording mid-replay is allowed (the replay
        # stream is written to the new H5 from the arm point on).
        if self.recording:
            raise RuntimeError("Recording is already running.")
        if queue_seconds <= 0:
            raise ValueError("queue_seconds must be positive.")

        queue_chunks = max(
            1,
            int(round(queue_seconds / 0.04)),
        )
        self._record_queue = queue.Queue(maxsize=queue_chunks)
        self._record_stop.clear()
        thread = _HDF5WriterThread(
            file_path=file_path,
            data_queue=self._record_queue,
            n_eeg=self.n_eeg,
            srate=self.srate,
            write_chunk_seconds=write_chunk_seconds,
            stop_event=self._record_stop,
        )
        self._record_thread = thread
        thread.start()

    def stop_recording(self) -> None:
        """Drain queued data, write the final partial block, and close the file."""
        thread = self._record_thread
        if thread is None:
            return

        thread.request_stop()
        thread.join(timeout=10.0)
        if thread.is_alive():
            raise RuntimeError("HDF5 writer thread did not stop cleanly.")

        self._check_recording_error()
        self._record_thread = None
        self._record_queue = None

    def disconnect(self) -> None:
        """Stop preprocessing, acquisition, and the TCP socket."""
        self._note_status("disconnect", "disconnect() called")
        self._stop_preprocess()

        self._acquisition_stop.set()
        thread = self._acquisition_thread

        if isinstance(thread, _NeuracleThread):
            thread.close_socket()

        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

        self._acquisition_thread = None
        self.stop_recording()

    def read_latest(self, frame: int) -> np.ndarray:
        """
        Read up to the latest `frame` raw samples.

        Returns
        -------
        ndarray
            Shape (n_channels, available_samples).
        """
        self._check_acquisition_error()
        self._check_recording_error()
        data, _ = self.raw_buffer.read_latest(frame)
        return data

    def start_preprocess(
        self,
        lowpass_hz: float = 45.0,
        downsample_factor: int = 10,
        highpass_hz: Optional[float] = 1.0,
        iir_order: int = 4,
        buffer_seconds: Optional[float] = None,
    ) -> None:
        """
        Start or replace the causal preprocessing pipeline.

        Pipeline
        --------
        Raw EEG
          -> causal Butterworth low-pass at raw sampling rate
          -> integer downsampling
          -> optional causal Butterworth high-pass at output sampling rate

        Only the first `n_eeg` channels are processed and stored.
        """
        if lowpass_hz <= 0 or lowpass_hz >= self.srate / 2:
            raise ValueError("lowpass_hz must lie inside (0, raw Nyquist).")

        downsample_factor = int(downsample_factor)
        if downsample_factor <= 0:
            raise ValueError("downsample_factor must be positive.")

        iir_order = int(iir_order)
        if iir_order <= 0:
            raise ValueError("iir_order must be positive.")

        output_srate = self.srate / downsample_factor

        if lowpass_hz >= output_srate / 2:
            raise ValueError(
                "lowpass_hz must be below the downsampled Nyquist frequency. "
                "For 1000 -> 100 Hz, use lowpass_hz < 50 Hz."
            )

        if highpass_hz is not None:
            highpass_hz = float(highpass_hz)
            if highpass_hz <= 0 or highpass_hz >= output_srate / 2:
                raise ValueError(
                    "highpass_hz must lie inside (0, output Nyquist), "
                    "or be None."
                )

        self._stop_preprocess()

        if buffer_seconds is None:
            buffer_seconds = self.buffer_seconds
        if buffer_seconds <= 0:
            raise ValueError("buffer_seconds must be positive.")

        config = PreprocessConfig(
            lowpass_hz=float(lowpass_hz),
            downsample_factor=downsample_factor,
            highpass_hz=highpass_hz,
            iir_order=iir_order,
            output_srate=output_srate,
        )

        output_capacity = max(1, int(round(buffer_seconds * output_srate)))
        self.preprocessed_buffer = RingBuffer(
            self.n_eeg,
            output_capacity,
        )
        self.preprocess_config = config

        self._preprocess_stop.clear()
        thread = _PreprocessThread(
            raw_buffer=self.raw_buffer,
            output_buffer=self.preprocessed_buffer,
            n_eeg=self.n_eeg,
            raw_srate=self.srate,
            config=config,
            stop_event=self._preprocess_stop,
        )
        self._preprocess_thread = thread
        thread.start()

    def _stop_preprocess(self) -> None:
        self._preprocess_stop.set()
        thread = self._preprocess_thread

        if thread is not None and thread.is_alive():
            # Wake a worker waiting on the raw buffer.
            self.raw_buffer.wait_for_new_data(timeout=0.0)
            thread.join(timeout=2.0)

        self._preprocess_thread = None

    def read_latest_preprocessed(
        self,
        frame: int,
    ) -> Tuple[np.ndarray, float]:
        """
        Read latest preprocessed EEG and its lag relative to current raw data.

        Returns
        -------
        data:
            Shape (n_eeg, available_samples).
        offset_seconds:
            Time difference between:
            - the current latest raw sample, and
            - the raw-source timestamp of the latest returned preprocessed sample.

            Returns NaN if either buffer is still empty.
        """
        if self.preprocessed_buffer is None or self.preprocess_config is None:
            raise RuntimeError("Call start_preprocess() first.")

        self._check_acquisition_error()
        self._check_preprocess_error()

        data, source_indices = self.preprocessed_buffer.read_latest(frame)

        raw_latest = self.raw_buffer.latest_source_index
        if raw_latest is None or source_indices.size == 0:
            offset_seconds = float("nan")
        else:
            offset_seconds = (
                int(raw_latest) - int(source_indices[-1])
            ) / self.srate

        return data, float(offset_seconds)

    def __enter__(self) -> "EEGRecorder":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.disconnect()
