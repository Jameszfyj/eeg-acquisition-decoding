# -*- coding: utf-8 -*-
"""Session adapters: one directory == one Session, in three wire formats.

The adapters only *normalise*; they never decide which raw samples become a
model column (that is :mod:`mi_studio.preprocess`) and never invent a class
mapping (the operator must review every source label id).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import contracts as C

KIND_CARGAME = "cargame_record"
KIND_OFFLINE = "offline_session"
KIND_ONLINE = "online_session"
KIND_ONLINE_CALIB = "online_calibration_session"

KIND_LABELS = {
    KIND_CARGAME: "cargame 离线记录",
    KIND_OFFLINE: "离线训练数据",
    KIND_ONLINE: "在线训练数据",
    KIND_ONLINE_CALIB: "在线校准数据",
}


@dataclass
class Session:
    session_id: str
    kind: str
    path: str
    date: str
    participant: str
    fs: float
    channels: list[str]
    eeg: np.ndarray                 # (n_channels, n_samples) float32 µV
    indices: np.ndarray             # (n_samples,) raw source index
    device_ms: np.ndarray           # (n_samples,) device clock in ms
    segments: list[tuple[int, int]]
    trials: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    complete: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    role: str = ""
    content_id: str = ""

    # ------------------------------------------------------------------ basic
    @property
    def n_channels(self) -> int:
        return int(self.eeg.shape[0])

    @property
    def n_samples(self) -> int:
        return int(len(self.indices))

    @property
    def duration_s(self) -> float:
        if not self.n_samples:
            return 0.0
        return float((self.device_ms[-1] - self.device_ms[0]) / 1000.0) + 1.0 / self.fs

    @property
    def kind_label(self) -> str:
        return KIND_LABELS.get(self.kind, self.kind)

    def fingerprint(self) -> str:
        """Content-backed recording identity (independent of the import path)."""
        if self.content_id:
            return self.content_id
        h = hashlib.sha256()
        h.update(f"{self.kind}|{self.session_id}|{self.n_channels}|{self.n_samples}".encode())
        h.update(np.asarray(self.indices[:64], np.int64).tobytes())
        h.update(np.asarray(self.indices[-64:], np.int64).tobytes())
        for t in self.trials:
            h.update(f"{t['trial_id']}:{t['label']}:{t['start']}:{t['end']}:"
                     f"{t.get('anchor')}".encode())
        return h.hexdigest()[:20]

    def channel_index(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for i, name in enumerate(self.channels):
            out.setdefault(str(name).strip(), i)
        return out

    # ------------------------------------------------------------------ labels
    def label_inventory(self) -> list[dict[str, Any]]:
        rows: dict[int, dict[str, Any]] = {}
        for trial in self.trials:
            label = int(trial["label"])
            row = rows.setdefault(label, {
                "id": label,
                "hint": trial.get("hint") or None,
                "hint_source": trial.get("hint_source") or None,
                "declared": trial.get("declared") or None,
                "count": 0,
            })
            row["count"] += 1
            if not row["hint"] and trial.get("hint"):
                row["hint"] = trial["hint"]
                row["hint_source"] = trial.get("hint_source")
            if not row["declared"] and trial.get("declared"):
                row["declared"] = trial["declared"]
        return [rows[k] for k in sorted(rows)]

    def scan_dict(self) -> dict[str, Any]:
        meta = {
            "paradigm_id": self.metadata.get("paradigm_id"),
            "condition": (self.metadata.get("recording_context") or {}).get("condition")
            if isinstance(self.metadata.get("recording_context"), dict) else None,
            "protocol_tasks": _protocol_task_texts(self.metadata),
            "n_gaps": max(0, len(self.segments) - 1),
            "n_samples": self.n_samples,
            "n_channels": self.n_channels,
            "duration_s": round(self.duration_s, 3),
            "complete": bool(self.complete),
            "role": self.role,
        }
        return {
            "session_id": self.session_id,
            "kind": self.kind,
            "kind_label": self.kind_label,
            "path": self.path,
            "date": self.date,
            "participant": self.participant,
            "fs": self.fs,
            "channels": list(self.channels),
            "n_channels": self.n_channels,
            "n_trials": len(self.trials),
            "duration_s": round(self.duration_s, 3),
            "complete": bool(self.complete),
            "fingerprint": self.fingerprint(),
            "n_segments": len(self.segments),
            "labels": self.label_inventory(),
            "warnings": list(self.warnings),
            "meta": meta,
        }


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _content_id(kind: str, session_id: str, eeg: np.ndarray | None, indices: np.ndarray,
                trigger: np.ndarray | None, metadata: dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(f"{kind}|{session_id}|{indices.shape[0]}|{eeg.shape[0] if eeg is not None else 0}"
             .encode())
    h.update(np.asarray(indices[:64], np.int64).tobytes())
    h.update(np.asarray(indices[-64:], np.int64).tobytes())
    if trigger is not None and trigger.size:
        nz = np.flatnonzero(trigger)
        h.update(nz[:64].astype(np.int64).tobytes())
        h.update(trigger[nz[:64]].astype(np.int64).tobytes())
        h.update(nz[-64:].astype(np.int64).tobytes())
        h.update(trigger[nz[-64:]].astype(np.int64).tobytes())
    if eeg is not None and eeg.size:
        h.update(np.ascontiguousarray(eeg[:, :16]).tobytes())
        h.update(np.ascontiguousarray(eeg[:, -16:]).tobytes())
    core = {k: metadata.get(k) for k in ("session_id", "paradigm_id", "participant_id",
                                         "protocol", "config") if metadata.get(k) is not None}
    h.update(json.dumps(core, ensure_ascii=False, sort_keys=True, default=str).encode())
    return h.hexdigest()[:20]


def _protocol_task_texts(metadata: dict[str, Any]) -> list[str]:
    config = metadata.get("config") or {}
    collection = config.get("collection") or {}
    tasks = collection.get("tasks") or metadata.get("protocol", {}).get("tasks") or []
    if isinstance(tasks, list) and tasks and isinstance(tasks[0], dict):
        return [str(t.get("text", "")).strip() for t in tasks]
    if isinstance(tasks, list):
        return [str(t) for t in tasks]
    return []


def _class_names(metadata: dict[str, Any]) -> dict[int, str]:
    config = metadata.get("config") or {}
    out: dict[int, str] = {}
    for item in config.get("classes") or []:
        try:
            out[int(item["id"])] = str(item.get("name", "")).strip()
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _segments(indices: np.ndarray, device_ms: np.ndarray | None, fs: float) -> list[tuple[int, int]]:
    """Split storage columns into continuous runs of the source index/clock."""
    n = len(indices)
    if not n:
        return []
    cut = np.zeros(n, bool)
    if n > 1:
        cut[1:] |= np.diff(indices.astype(np.int64)) != 1
    if device_ms is not None and len(device_ms) == n and n > 1:
        tol_ms = 1.5 * (1000.0 / float(fs))
        cut[1:] |= np.abs(np.diff(np.asarray(device_ms, np.float64))) > tol_ms
    starts = [0] + list(np.flatnonzero(cut))
    stops = starts[1:] + [n]
    return [(int(a), int(b)) for a, b in zip(starts, stops) if b > a]


def _decode_trigger_values(raw) -> np.ndarray:
    """Neuracle trigger channel -> integer ids (copied from cargame)."""
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


def _pulse_rows(trigger: np.ndarray) -> list[tuple[int, int]]:
    """Non-zero trigger samples, merging runs of the same id."""
    rows: list[tuple[int, int]] = []
    prev_code, prev_row = None, -10
    for row in np.flatnonzero(trigger):
        code = int(trigger[row])
        if code == 0:
            continue
        if code == prev_code and row == prev_row + 1:
            prev_row = int(row)
            continue
        rows.append((int(row), code))
        prev_code, prev_row = code, int(row)
    return rows


def _parse_cargame_header(path: Path) -> dict[str, Any]:
    header, best = "", 0
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    for line in lines:
        if "|" not in line:
            continue
        payload = line.split("|", 1)[1].strip()
        n_fields = payload.count(";")
        if n_fields > best:
            best, header = n_fields, payload
    parts = [p for p in header.split(";") if p != ""]
    info: dict[str, Any] = {"raw": header, "codes": [], "loops": 20,
                            "task_seconds": 6.0, "prepare_seconds": 2.0,
                            "rest_seconds": 2.0}
    for i, part in enumerate(parts):
        if part and all(ch in "LRTFDqxd," for ch in part) and ("L" in part or "R" in part):
            info["codes"] = [ch.strip() for ch in part.split(",") if ch.strip()]
        if part.isdigit():
            n = int(part)
            rest = parts[i + 1] if i + 1 < len(parts) else ""
            if rest.isdigit() and n <= 100:
                info["loops"] = n
                info["task_seconds"] = float(parts[i + 1])
                if i + 2 < len(parts) and parts[i + 2].isdigit():
                    info["prepare_seconds"] = float(parts[i + 2])
                if i + 3 < len(parts) and parts[i + 3].isdigit():
                    info["rest_seconds"] = float(parts[i + 3])
                break
    return info


def _in_one_segment(segments: list[tuple[int, int]], row_a: int, row_b: int) -> bool:
    for a, b in segments:
        if a <= row_a < b:
            return a <= row_b < b
    return False


def _session_date(text: str, fallback_path: Path) -> str:
    digits = "".join(ch if ch.isdigit() else " " for ch in text).split()
    for token in digits:
        if len(token) >= 8:
            return token[:8]
    for part in reversed(fallback_path.parts):
        for token in part.replace("_", " ").replace("-", " ").split():
            if len(token) >= 8 and token[:8].isdigit():
                return token[:8]
    return ""


# ---------------------------------------------------------------------------
# cargame offline record
# ---------------------------------------------------------------------------
def load_cargame(path: str | Path, with_eeg: bool = True) -> Session:
    import h5py

    folder = Path(path)
    h5_path = folder / "data.h5" if folder.is_dir() else folder
    folder = h5_path.parent
    log_path = folder / "log.txt"
    if not h5_path.is_file():
        raise FileNotFoundError(f"缺少 {h5_path}")
    with h5py.File(h5_path, "r") as h:
        fmt = str(h.attrs.get("format", ""))
        if "Neuracle" not in fmt and "eeg" not in h:
            raise C.ContractError(f"{h5_path} 不是 cargame Neuracle 记录")
        n_eeg = int(h["eeg"].shape[0])
        n_columns = int(h["eeg"].shape[1])
        eeg = np.asarray(h["eeg"], dtype=np.float32) if with_eeg else np.empty((n_eeg, 0), np.float32)
        trigger = _decode_trigger_values(h["trigger"][:]) if "trigger" in h else np.zeros(n_columns, np.uint32)
        indices = (np.asarray(h["source_indices"][:], dtype=np.int64)
                   if "source_indices" in h else np.arange(n_columns, dtype=np.int64))
        attrs = {k: str(v) for k, v in h.attrs.items()}
    fs = float(attrs.get("srate", 1000.0) or 1000.0)
    warnings: list[str] = []
    if n_eeg != len(C.CARGAME_CHANNELS):
        warnings.append(f"通道数 {n_eeg} ≠ {len(C.CARGAME_CHANNELS)}，将按名称匹配")
    channels = list(C.CARGAME_CHANNELS[:n_eeg])
    if len(channels) < n_eeg:
        channels += [f"CH{i}" for i in range(len(channels), n_eeg)]
    device_ms = np.asarray(indices, np.float64) * (1000.0 / fs)
    segments = _segments(indices, None, fs)
    header = _parse_cargame_header(log_path) if log_path.is_file() else {"task_seconds": 6.0}
    task_s = float(header.get("task_seconds") or 6.0)

    events = _pulse_rows(trigger)
    cues = [(row, code) for row, code in events if code in C.CARGAME_CUE_CODES]
    ends = [row for row, code in events if code == C.CARGAME_TASK_END_CODE]
    ends_arr = np.asarray(ends, dtype=np.int64)
    trials: list[dict[str, Any]] = []
    skipped = {"no_end": 0, "bad_duration": 0, "unknown_code": 0}
    for row, code in cues:
        j = int(np.searchsorted(ends_arr, row, side="right"))
        if j >= len(ends_arr):
            skipped["no_end"] += 1
            continue
        end_row = int(ends_arr[j])
        # Trigger rows are STORAGE rows; windows/timestamps are in SOURCE
        # indices.  Translate through the recording's index map and refuse a
        # pair that straddles a missing-source gap.
        start_idx = int(indices[row])
        end_idx = int(indices[end_row])
        if end_idx <= start_idx:
            skipped["bad_duration"] += 1
            continue
        duration = (end_idx - start_idx) / fs
        if abs(duration - task_s) > 0.3:
            skipped["bad_duration"] += 1
            continue
        if not _in_one_segment(segments, row, end_row):
            skipped["bad_duration"] += 1
            warnings.append(f"cue {code}@{row} 与 task_end 跨连续段，丢弃")
            continue
        hint = C.CARGAME_CODE_HINTS.get(code)
        trials.append(dict(trial_id=f"{code}:{row}", label=int(code),
                           start=start_idx, end=end_idx,
                           anchor=None, anchor_kind=None,
                           prepare_seconds=float(header.get("prepare_seconds") or 2.0),
                           row_start=int(row), row_end=int(end_row), hint=hint,
                           hint_source="cargame TASK_SPECS" if hint else None,
                           declared=None, duration_s=round(duration, 3)))
    if skipped["no_end"] or skipped["bad_duration"]:
        warnings.append(f"丢弃试次：缺 task_end {skipped['no_end']}，时长不符 {skipped['bad_duration']}")
    if not trials:
        raise C.ContractError(f"{folder.name}: 没有可用的 cue→task_end 试次")
    date = _session_date(folder.name, folder)
    meta = {"log_header": header, "device": attrs.get("format", ""), "source": "cargame"}
    return Session(
        session_id=folder.name, kind=KIND_CARGAME, path=str(folder.resolve()),
        date=date, participant=folder.parent.name, fs=fs, channels=channels,
        eeg=np.ascontiguousarray(eeg), indices=np.asarray(indices, np.int64),
        device_ms=device_ms, segments=segments, trials=trials, warnings=warnings,
        complete=bool(str(attrs.get("complete", "True")).lower() in ("1", "true")),
        metadata=meta, role="cargame_offline",
        content_id=_content_id(KIND_CARGAME, folder.name, eeg if with_eeg else None,
                               np.asarray(indices, np.int64), trigger, meta),
    )


# ---------------------------------------------------------------------------
# handbci offline training session (data.h5 + session.json)
# ---------------------------------------------------------------------------
def load_offline(path: str | Path, with_eeg: bool = True) -> Session:
    import h5py

    folder = Path(path)
    h5_path = folder / "data.h5" if folder.is_dir() else folder
    folder = h5_path.parent
    with h5py.File(h5_path, "r") as h:
        if h.attrs.get("handbci_format") != "handbci.raw.v1":
            raise C.ContractError(f"{h5_path} 不是 handbci.raw.v1 记录")
        metadata = json.loads(h.attrs["metadata_json"])
        n_eeg = int(h["eeg"].shape[0])
        eeg = np.asarray(h["eeg"], dtype=np.float32) if with_eeg else np.empty((n_eeg, 0), np.float32)
        indices = np.asarray(h["source_indices"], dtype=np.int64)
        trigger = np.asarray(h["trigger"], dtype=np.uint32) if "trigger" in h else np.zeros(len(indices), np.uint32)
        events = [json.loads(e) for e in h["events_json"].asstr()[:]]
        device_ms = (np.asarray(h["device_time_ms"], np.float64)
                     if "device_time_ms" in h else None)
        complete = bool(h.attrs.get("complete", False))
    fs = float(metadata.get("config", {}).get("device", {}).get("sample_rate", 1000.0))
    if device_ms is None:
        device_ms = np.asarray(indices, np.float64) * (1000.0 / fs)
    segments = _segments(indices, device_ms, fs)
    warnings: list[str] = []
    if not complete or not metadata.get("complete", False):
        warnings.append("记录未完整结束（complete=false）")

    task_texts = _protocol_task_texts(metadata)
    declared_names = _class_names(metadata)
    collection = (metadata.get("config") or {}).get("collection") or {}
    try:
        session_prepare_s = float(collection.get("prepare_seconds"))
    except (TypeError, ValueError):
        session_prepare_s = None
    # prepare（目标提示）是 v2 的锚点：按 trial_id 建索引，缺失才回退
    prepares = {int(e.get("trial_id", -1)): e for e in events
                if str(e.get("kind", "")) == "prepare" and e.get("sample_index") is not None}
    active: dict[str, Any] | None = None
    trials: list[dict[str, Any]] = []
    bad = 0
    n_anchor_prepare = 0
    for event in events:
        kind = str(event.get("kind", ""))
        if kind == "cue":
            if active is not None:
                bad += 1
                active = None
            active = event
        elif kind == "task_end":
            if active is None or active.get("trial_id") != event.get("trial_id"):
                bad += 1
                active = None
                continue
            start = int(active.get("sample_index", -1))
            end = int(event.get("sample_index", -1))
            label = int(active.get("label", -1))
            if start < 0 or end <= start:
                bad += 1
                active = None
                continue
            hint = task_texts[label] if 0 <= label < len(task_texts) else None
            declared = declared_names.get(label)
            if hint and declared and hint != declared:
                warnings.append(f"标签 {label}：提示「{hint}」与类别表「{declared}」不一致")
            trial_id = int(active.get("trial_id", len(trials) + 1))
            prep = prepares.get(trial_id)
            anchor = int(prep["sample_index"]) if prep is not None else None
            anchor_kind = "prepare" if prep is not None else None
            if prep is not None:
                n_anchor_prepare += 1
            trials.append(dict(trial_id=trial_id,
                               label=label, start=start, end=end,
                               anchor=anchor, anchor_kind=anchor_kind,
                               prepare_seconds=session_prepare_s,
                               hint=hint or None,
                               hint_source="protocol.tasks" if hint else None,
                               declared=declared or None,
                               duration_s=round((end - start) / fs, 3)))
            active = None
    if active is not None:
        bad += 1
    if bad:
        warnings.append(f"丢弃 {bad} 个开/关不配对的试次")
    if trials and n_anchor_prepare < len(trials):
        warnings.append(f"{len(trials) - n_anchor_prepare}/{len(trials)} 个试次没有 prepare 事件，"
                        f"锚点回退到 prepare_seconds")
    if not trials:
        raise C.ContractError(f"{folder.name}: 没有 cue→task_end 试次")

    channels = [str(c) for c in metadata.get("config", {}).get("device", {}).get("channels", [])]
    if len(channels) != eeg.shape[0]:
        raise C.ContractError(f"{folder.name}: 通道表 {len(channels)} 与数据 {eeg.shape[0]} 不一致")
    date = _session_date(str(metadata.get("session_id", folder.name)), folder)
    return Session(
        session_id=str(metadata.get("session_id", folder.name)), kind=KIND_OFFLINE,
        path=str(folder.resolve()), date=date,
        participant=str(metadata.get("participant_id", "")), fs=fs,
        channels=channels, eeg=np.ascontiguousarray(eeg), indices=indices,
        device_ms=np.asarray(device_ms, np.float64), segments=segments, trials=trials,
        warnings=warnings, complete=bool(complete and metadata.get("complete", False)),
        metadata=metadata, role=str(metadata.get("recording_context", {}).get("condition", "")),
    )


# ---------------------------------------------------------------------------
# online decode session (raw_acp.h5 + trials_json/device.json)
# ---------------------------------------------------------------------------
def _online_channels(folder: Path, attrs: dict[str, str]) -> list[str]:
    device = folder / "device.json"
    if device.is_file():
        try:
            data = json.loads(device.read_text(encoding="utf-8"))
            labels = data.get("labels") or data.get("channels")
            if labels:
                return [str(x) for x in labels]
        except (OSError, ValueError):
            pass
    inp = folder / "input.json"
    if inp.is_file():
        try:
            data = json.loads(inp.read_text(encoding="utf-8"))
            labels = data.get("device_channels")
            if labels:
                return [str(x) for x in labels]
        except (OSError, ValueError):
            pass
    raw = attrs.get("channels")
    if raw:
        try:
            return [str(x) for x in json.loads(raw)]
        except ValueError:
            pass
    return list(C.DEVICE_CHANNELS)


def _online_eeg(folder: Path, with_eeg: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    import h5py

    h5_path = folder / "raw_acp.h5"
    if not h5_path.is_file():
        raise FileNotFoundError(f"缺少 {h5_path}")
    with h5py.File(h5_path, "r") as h:
        attrs = {k: str(v) for k, v in h.attrs.items()}
        n_eeg = int(h["eeg"].shape[0])
        n_cols = int(h["eeg"].shape[1])
        eeg = (np.asarray(h["eeg"], dtype=np.float32) if with_eeg
               else np.empty((n_eeg, 0), np.float32))
        if "source_index" in h:
            indices = np.asarray(h["source_index"], dtype=np.int64)
            attrs["_coords"] = "source"
        elif "source_indices" in h:
            indices = np.asarray(h["source_indices"], dtype=np.int64)
            attrs["_coords"] = "source"
        else:
            indices = np.arange(n_cols, dtype=np.int64)
            attrs["_coords"] = "row"
            attrs["_missing_source_index"] = "1"
        device_ms = (np.asarray(h["device_ms"], np.float64)
                     if "device_ms" in h else np.asarray(indices, np.float64))
        trigger = (np.asarray(h["trigger"], dtype=np.uint32)
                   if "trigger" in h else np.zeros(len(indices), np.uint32))
        trials = ([json.loads(x) for x in h["trials_json"].asstr()[:]]
                  if "trials_json" in h else [])
        events = ([json.loads(x) for x in h["events_json"].asstr()[:]]
                  if "events_json" in h else [])
    # external trial table is an accepted online layout too
    external = folder / "trials.json"
    if not trials and external.is_file():
        try:
            data = json.loads(external.read_text(encoding="utf-8"))
            trials = data if isinstance(data, list) else data.get("trials") or []
        except (OSError, ValueError):
            trials = []
    return eeg, indices, device_ms, {**attrs, "_trials_json": trials,
                                     "_events_json": events, "_trigger": trigger}


def load_online_decode(path: str | Path, with_eeg: bool = True) -> Session:
    folder = Path(path)
    if not folder.is_dir():
        folder = folder.parent
    eeg, indices, device_ms, attrs = _online_eeg(folder, with_eeg)
    trials_json = attrs.pop("_trials_json", [])
    events_json = attrs.pop("_events_json", [])
    trigger = attrs.pop("_trigger", None)
    coords = str(attrs.pop("_coords", "source"))
    missing_index = attrs.pop("_missing_source_index", "")
    channels = _online_channels(folder, attrs)
    fs = float(attrs.get("sample_rate", 1000.0) or 1000.0)
    segments = _segments(indices, device_ms, fs)
    warnings: list[str] = []
    if missing_index:
        warnings.append("raw_acp.h5 没有 source_index：试次坐标按存储行处理")
    if coords == "row":
        warnings.append("试次坐标是存储行，不是源索引；已按行号解释")

    def as_source(value, seconds):
        """Translate a declared trial bound into a source index."""
        v = int(value)
        if coords == "row":
            return int(indices[v]) if 0 <= v < len(indices) else v
        if seconds is not None:
            seconds = float(seconds)
            expected = int(round(seconds * fs))
            if v < int(indices[0]) or v > int(indices[-1]):
                return int(indices[0]) + expected
        return v

    trials: list[dict[str, Any]] = []
    if not trials_json and events_json:
        active = None
        for event in events_json:
            kind = str(event.get("kind", ""))
            if kind == "cue":
                active = event
            elif kind == "task_end" and active is not None:
                try:
                    label = int(active.get("label"))
                    start = as_source(active.get("sample_index"), None)
                    end = as_source(event.get("sample_index"), None)
                except (TypeError, ValueError):
                    active = None
                    continue
                hint = str(active.get("text") or active.get("task_text") or "").strip() or None
                trials.append(dict(trial_id=int(active.get("trial_id", len(trials) + 1)),
                                   label=label, start=start, end=end, hint=hint,
                                   anchor=None, anchor_kind=None, prepare_seconds=None,
                                   hint_source="events.text" if hint else None,
                                   declared=None, duration_s=round((end - start) / fs, 3)))
                active = None
    for t in trials_json:
        try:
            start_raw = t.get("start_sample", t.get("start_index"))
            end_raw = t.get("end_sample", t.get("end_index"))
            start = as_source(start_raw, t.get("start_seconds"))
            end = as_source(end_raw, t.get("end_seconds"))
            label = int(t.get("label"))
        except (TypeError, ValueError):
            continue
        if not t.get("complete", True) or end <= start:
            continue
        row_a = int(np.searchsorted(indices, start, side="left"))
        row_b = int(np.searchsorted(indices, end, side="left"))
        if row_a >= len(indices) or row_b >= len(indices) or not _in_one_segment(segments, row_a, row_b):
            warnings.append(f"试次 {t.get('trial_id')} 跨连续段或越界，丢弃")
            continue
        hint = str(t.get("task_text", "")).strip() or None
        preview = t.get("preview_sample")
        anchor = as_source(preview, t.get("preview_seconds")) if preview is not None else None
        if anchor is not None and not (start - anchor) > 0:
            anchor = None  # 预提示必须早于 cue，否则这条元数据不可信
        try:
            lead = float(t.get("cue_display_lead_seconds"))
        except (TypeError, ValueError):
            lead = None
        trials.append(dict(trial_id=int(t.get("trial_id", len(trials) + 1)), label=label,
                           start=start, end=end, hint=hint,
                           anchor=anchor,
                           anchor_kind="preview_sample" if anchor is not None else None,
                           prepare_seconds=lead if lead and lead > 0 else None,
                           hint_source="trials.task_text" if hint else None,
                           declared=None, round=t.get("round"),
                           duration_s=round((end - start) / fs, 3)))
    model_selection = folder / "model_selection.json"
    role = "online_decode"
    if model_selection.is_file():
        try:
            role = str(json.loads(model_selection.read_text(encoding="utf-8"))
                       .get("session_purpose", role))
        except (OSError, ValueError):
            pass
    if not trials:
        raise C.ContractError(f"{folder.name}: trials_json 没有完整试次")
    date = _session_date(folder.name, folder)
    return Session(
        session_id=folder.name, kind=KIND_ONLINE, path=str(folder.resolve()), date=date,
        participant=folder.name.split("_")[-1] if "_" in folder.name else "", fs=fs,
        channels=channels, eeg=np.ascontiguousarray(eeg), indices=indices,
        device_ms=np.asarray(device_ms, np.float64), segments=segments, trials=trials,
        warnings=warnings, complete=True, metadata={"attrs": attrs}, role=role,
    )


# ---------------------------------------------------------------------------
# online calibration session (raw_acp.h5 + calibration.json)
# ---------------------------------------------------------------------------
def load_online_calibration(path: str | Path, with_eeg: bool = True) -> Session:
    folder = Path(path)
    if not folder.is_dir():
        folder = folder.parent
    calib_path = folder / "calibration.json"
    if not calib_path.is_file():
        raise FileNotFoundError(f"缺少 {calib_path}")
    calibration = json.loads(calib_path.read_text(encoding="utf-8"))
    eeg, indices, device_ms, attrs = _online_eeg(folder, with_eeg)
    attrs.pop("_trials_json", None)
    attrs.pop("_events_json", None)
    trigger = attrs.pop("_trigger", None)
    attrs.pop("_coords", None)
    attrs.pop("_missing_source_index", None)
    channels = _online_channels(folder, attrs)
    fs = float(attrs.get("sample_rate", 1000.0) or 1000.0)
    segments = _segments(indices, device_ms, fs)

    class_names: dict[int, str] = {}
    selection = folder / "model_selection.json"
    if selection.is_file():
        try:
            data = json.loads(selection.read_text(encoding="utf-8"))
            for item in data.get("classes") or []:
                class_names[int(item["id"])] = str(item.get("name", "")).strip()
        except (OSError, ValueError, KeyError, TypeError):
            pass

    events = calibration.get("events") or []
    active: dict[str, Any] | None = None
    trials: list[dict[str, Any]] = []
    for event in events:
        kind = str(event.get("kind", ""))
        if kind == "imagery":
            active = event
        elif kind == "task_end" and active is not None:
            start = int(active.get("source_index", -1))
            end = int(event.get("source_index", -1))
            label = int(active.get("task_id", -1))
            if start >= 0 and end > start and label >= 0:
                row_a = int(np.searchsorted(indices, start, side="left"))
                row_b = int(np.searchsorted(indices, end, side="left"))
                if row_a >= len(indices) or row_b >= len(indices) or not _in_one_segment(segments, row_a, row_b):
                    active = None
                    continue
                hint = class_names.get(label) or None
                trials.append(dict(trial_id=int(active.get("trial", len(trials) + 1)),
                                   label=label, start=start, end=end, hint=hint,
                                   anchor=None, anchor_kind=None, prepare_seconds=None,
                                   hint_source="model_selection.classes" if hint else None,
                                   declared=hint, round=active.get("round"),
                                   duration_s=round((end - start) / fs, 3)))
            active = None
    if not trials:
        raise C.ContractError(f"{folder.name}: calibration.json 没有 imagery→task_end 试次")
    date = _session_date(folder.name, folder)
    complete = bool(calibration.get("complete", True))
    return Session(
        session_id=folder.name, kind=KIND_ONLINE_CALIB, path=str(folder.resolve()),
        date=date, participant=folder.name.split("_")[-1] if "_" in folder.name else "",
        fs=fs, channels=channels, eeg=np.ascontiguousarray(eeg), indices=indices,
        device_ms=np.asarray(device_ms, np.float64), segments=segments, trials=trials,
        warnings=[] if complete else ["校准记录未完整结束"],
        complete=complete, metadata={"calibration": {k: v for k, v in calibration.items()
                                                     if k != "events"}},
        role="online_calibration",
        content_id=_content_id(KIND_ONLINE_CALIB, folder.name, eeg if with_eeg else None,
                               indices, trigger, {"session_id": folder.name}),
    )


# ---------------------------------------------------------------------------
# detection / entry point
# ---------------------------------------------------------------------------
def detect_kind(path: str | Path) -> str:
    import h5py

    folder = Path(path)
    if folder.is_file():
        folder = folder.parent
    if (folder / "raw_acp.h5").is_file():
        # A decode recording carries a trials_json dataset; a pure calibration
        # recording carries calibration events instead.  Prefer trials_json so a
        # decode session that happens to embed a calibration block stays a decode
        # session.
        with h5py.File(folder / "raw_acp.h5", "r") as h:
            has_trials = "trials_json" in h and len(h["trials_json"]) > 0
        has_trials = has_trials or (folder / "trials.json").is_file()
        if not has_trials and (folder / "calibration.json").is_file():
            try:
                data = json.loads((folder / "calibration.json").read_text(encoding="utf-8"))
                if data.get("events"):
                    return KIND_ONLINE_CALIB
            except (OSError, ValueError):
                pass
        return KIND_ONLINE
    h5_path = folder / "data.h5"
    if not h5_path.is_file():
        raise FileNotFoundError(f"{folder} 下找不到 data.h5 / raw_acp.h5")
    with h5py.File(h5_path, "r") as h:
        if h.attrs.get("handbci_format") == "handbci.raw.v1":
            return KIND_OFFLINE
        if "eeg" in h and ("trigger" in h or "source_indices" in h):
            return KIND_CARGAME
    raise C.ContractError(f"{folder} 的记录格式无法识别")


LOADERS = {
    KIND_CARGAME: load_cargame,
    KIND_OFFLINE: load_offline,
    KIND_ONLINE: load_online_decode,
    KIND_ONLINE_CALIB: load_online_calibration,
}


def load_session(path: str | Path, with_eeg: bool = True) -> Session:
    kind = detect_kind(path)
    return LOADERS[kind](path, with_eeg=with_eeg)


def scan(path: str | Path, max_sessions: int = 200) -> list[dict[str, Any]]:
    """Scan one session directory, or every immediate child that looks like one."""
    folder = Path(path)
    if folder.is_file():
        folder = folder.parent
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} 不存在")
    candidates: list[Path] = []
    try:
        if detect_kind(folder):
            candidates.append(folder)
    except (FileNotFoundError, C.ContractError):
        pass
    if not candidates:
        for child in sorted(folder.iterdir()):
            if not child.is_dir():
                continue
            try:
                detect_kind(child)
            except (FileNotFoundError, C.ContractError):
                continue
            candidates.append(child)
            if len(candidates) >= max_sessions:
                break
    if not candidates:
        raise C.ContractError(f"{folder} 下没有可识别的会话目录")
    rows = []
    for child in candidates:
        try:
            rows.append(load_session(child, with_eeg=False).scan_dict())
        except (C.ContractError, OSError, KeyError, ValueError) as exc:
            rows.append({"path": str(child), "kind": "invalid", "error": f"{type(exc).__name__}: {exc}"})
    return rows
