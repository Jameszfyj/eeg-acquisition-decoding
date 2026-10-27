# -*- coding: utf-8 -*-
"""Synthetic recordings for the studio tests (no real project data needed)."""
from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from mi_studio_v2 import contracts as C

NAMES = ["左握拳", "右握拳", "左旋臂", "右旋臂", "双脚", "舌头"]


def base_config(n_loops: int, task_s: float = 4.0) -> dict:
    return {
        "paradigm_id": "studio_test",
        "device": {"source": "neuracle_acp", "unit": "uV", "sample_rate": 1000,
                   "stream_channels": 33, "channels": list(C.DEVICE_CHANNELS),
                   "host": None, "port": None},
        "preprocess": {"highpass_hz": 4.0, "lowpass_hz": 100.0, "numtaps": 201,
                       "trim_seconds": 1.0, "input_rate": 1000, "output_rate": 250},
        "window": {"seconds": 2.0, "stride_seconds": 0.5, "task_offset_seconds": 1.0},
        "classes": [{"id": i, "name": n, "task_text": n, "action": "custom"}
                    for i, n in enumerate(NAMES)],
        "collection": {"loops": n_loops, "initial_rest_seconds": 3.0,
                       "prepare_seconds": 2.4, "task_seconds": task_s,
                       "rest_seconds": 3.0, "block_rest_seconds": 6.0,
                       "tail_seconds": 1.0},
    }


def _write_offline(folder, n_trials=2, task_s=4.0, labels=None, incomplete=False,
                   cue_offset=5000, gap_at=None, seed=0):
    labels = labels or [0, 1]
    fs = 1000
    cursor = cue_offset
    trials = []
    n_total = cursor + (len(labels) + 1) * (int(task_s * fs) + 3 * fs) + 2 * fs
    eeg = np.random.default_rng(seed).normal(0, 3, (32, n_total)).astype(np.float32)
    trigger = np.zeros(n_total, np.uint32)
    events = []
    prepare_s = float(base_config(len(labels), task_s)["collection"]["prepare_seconds"])
    for i, label in enumerate(labels, 1):
        start = cursor + i * (int(task_s * fs) + 3 * fs)
        end = start + int(task_s * fs)
        prepare = int(round(start - prepare_s * fs))
        trigger[prepare] = 20 + label
        trigger[start] = label + 1
        trigger[end] = 9
        events.append({"kind": "prepare", "trial_id": i, "label": int(label),
                       "event_id": 20 + int(label), "sample_index": prepare,
                       "echo_verified": True})
        events.append({"kind": "cue", "trial_id": i, "label": int(label),
                       "sample_index": int(start)})
        events.append({"kind": "task_end", "trial_id": i, "sample_index": int(end)})
        trials.append((start, end))
    indices = np.arange(n_total, dtype=np.int64)
    if gap_at is not None:
        a, b = gap_at
        indices[a:b] = indices[a:b] + 4000
    metadata = {"format": "handbci.raw.v1", "session_id": folder.name,
                "complete": not incomplete, "source": "neuracle_acp",
                "config": base_config(len(labels), task_s), "participant_id": "P01",
                "protocol": {"tasks": [{"text": n} for n in NAMES],
                             "condition": "test"},
                "recording_context": {"condition": "test"}}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "session.json").write_text(json.dumps(metadata, ensure_ascii=False),
                                         encoding="utf-8")
    with h5py.File(folder / "data.h5", "w") as h:
        h.attrs["handbci_format"] = "handbci.raw.v1"
        h.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
        h.attrs["complete"] = str(not incomplete)
        h.attrs["srate"] = "1000"
        h.attrs["n_eeg"] = 32
        h.create_dataset("eeg", data=eeg)
        h.create_dataset("source_indices", data=indices)
        h.create_dataset("trigger", data=trigger)
        h.create_dataset("events_json",
                         data=[json.dumps(e, ensure_ascii=False) for e in events],
                         dtype=h5py.string_dtype("utf-8"))
        h.create_dataset("device_time_ms", data=indices.astype(np.float64))
    return folder


def _write_cargame(folder, n_trials=2, task_s=4.0, codes=None, gap_at=None, seed=0):
    codes = codes or [18]
    fs = 1000
    cursor = 5000
    rows = []
    n_total = cursor + n_trials * (int(task_s * fs) + 2 * fs) + 2 * fs
    eeg = np.random.default_rng(seed).normal(0, 3, (32, n_total)).astype(np.float32)
    trigger = np.zeros(n_total, np.uint32)
    for i in range(n_trials):
        code = codes[i % len(codes)]
        start = cursor + i * (int(task_s * fs) + 2 * fs)
        end = start + int(task_s * fs)
        trigger[start] = code
        trigger[end] = 9
        rows.append((start, end))
    indices = np.arange(n_total, dtype=np.int64)
    if gap_at is not None:
        a, b = gap_at
        indices[a:b] = indices[a:b] + 4000
    folder.mkdir(parents=True, exist_ok=True)
    with h5py.File(folder / "data.h5", "w") as h:
        h.attrs["format"] = "Neuracle EEG recording"
        h.attrs["srate"] = "1000.0"
        h.attrs["n_eeg"] = 32
        h.attrs["complete"] = "True"
        h.create_dataset("eeg", data=eeg)
        h.create_dataset("source_indices", data=indices)
        h.create_dataset("trigger", data=trigger)
    (folder / "log.txt").write_text(
        "2026-01-01 00:00:00.000|start record\n"
        "2026-01-01 00:00:01.000|100;I;Lq,Lx,Rq,Rx;F;C;20;4;2;2;T;F\n",
        encoding="utf-8")
    return folder


def _write_online(folder, n_trials=2, task_s=4.0, with_source_index=True,
                  external_trials=True, seed=0):
    fs = 1000
    n_total = 5000 + n_trials * (int(task_s * fs) + 3 * fs) + 2 * fs
    eeg = np.random.default_rng(seed).normal(0, 3, (32, n_total)).astype(np.float32)
    indices = np.arange(n_total, dtype=np.int64) + (1200 if with_source_index else 0)
    folder.mkdir(parents=True, exist_ok=True)
    with h5py.File(folder / "raw_acp.h5", "w") as h:
        h.attrs["sample_rate"] = "1000"
        h.attrs["unit"] = "uV"
        h.create_dataset("eeg", data=eeg)
        if with_source_index:
            h.create_dataset("source_index", data=indices)
        h.create_dataset("device_ms", data=indices.astype(np.float64))
        h.create_dataset("trigger", data=np.zeros(n_total, np.uint32))
    (folder / "device.json").write_text(
        json.dumps({"labels": list(C.DEVICE_CHANNELS), "settings": {"fs": 1000}},
                   ensure_ascii=False), encoding="utf-8")
    trials = []
    for i in range(n_trials):
        start = 5000 + i * (int(task_s * fs) + 3 * fs)
        end = start + int(task_s * fs)
        trials.append({"trial_id": i + 1, "label": i % 6,
                       "start_sample": int(start + (1200 if with_source_index else 0)),
                       "end_sample": int(end + (1200 if with_source_index else 0)),
                       "start_seconds": start / fs, "end_seconds": end / fs,
                       "task_text": NAMES[i % 6], "complete": True})
    if external_trials:
        (folder / "trials.json").write_text(json.dumps(trials, ensure_ascii=False),
                                             encoding="utf-8")
    else:
        with h5py.File(folder / "raw_acp.h5", "a") as h:
            h.create_dataset("trials_json",
                             data=[json.dumps(t, ensure_ascii=False) for t in trials],
                             dtype=h5py.string_dtype("utf-8"))
            h.create_dataset("events_json", data=[],
                             dtype=h5py.string_dtype("utf-8"))
    return folder


@pytest.fixture
def offline_session(tmp_path):
    return _write_offline(tmp_path / "20260101_000000_offline_P01")


@pytest.fixture
def cargame_session(tmp_path):
    return _write_cargame(tmp_path / "20260101_000000_cargame")


@pytest.fixture
def online_session(tmp_path):
    return _write_online(tmp_path / "20260101_000000_online_P01")


@pytest.fixture
def make_offline(tmp_path):
    def factory(name="sess", **kwargs):
        return _write_offline(tmp_path / name, **kwargs)
    return factory


@pytest.fixture
def make_cargame(tmp_path):
    def factory(name="cg", **kwargs):
        return _write_cargame(tmp_path / name, **kwargs)
    return factory


@pytest.fixture
def make_online(tmp_path):
    def factory(name="on", **kwargs):
        return _write_online(tmp_path / name, **kwargs)
    return factory
