# -*- coding: utf-8 -*-
"""Causal FIR preprocessing and 2 s window extraction.

The filter class below is a verbatim copy of ``handbci.preprocess.StreamPreprocessor``
(same taps, decimation phase, reset/trim state).  It is vendored so the studio
runs against the same arithmetic on a machine that has no platform checkout;
``tests/test_preprocess_parity.py`` compares it against the real class whenever
the platform repository is importable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import signal

from . import contracts as C
from .adapters import Session, load_session
from .mapping import effective_map, validate_mapping


class StreamPreprocessor:
    """Stateful 4–100 Hz causal FIR -> 250 Hz -> trim (platform semantics)."""

    def __init__(self, config: dict):
        self.config = config
        self.fs = float(config["device"]["sample_rate"])
        prep = config["preprocess"]
        self.output_fs = float(prep["output_rate"])
        self.factor = int(round(self.fs / self.output_fs))
        self.n_channels = len(config["device"]["channels"])
        taps = int(prep["numtaps"])
        self.lowpass = signal.firwin(taps, prep["lowpass_hz"], fs=self.fs,
                                     window=("kaiser", 5.65))
        self.highpass = signal.firwin(taps, prep["highpass_hz"], fs=self.output_fs,
                                      pass_zero="highpass", window=("kaiser", 5.65))
        self.delay_seconds = ((taps - 1) / (2 * self.fs)
                              + (taps - 1) / (2 * self.output_fs))
        self.history_source_samples = ((len(self.lowpass) - 1)
                                       + (len(self.highpass) - 1) * self.factor)
        self.initial_trim = int(round(prep["trim_seconds"] * self.output_fs))
        self.reset()

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
        low, self.low_state = signal.lfilter(self.lowpass, [1.0], x, axis=-1,
                                             zi=self.low_state)
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


def synth_config(session: Session, prof: dict[str, Any]) -> dict[str, Any]:
    return {
        "device": {"channels": list(session.channels),
                   "sample_rate": float(session.fs)},
        "preprocess": {"numtaps": int(prof["numtaps"]),
                       "lowpass_hz": float(prof["lowpass_hz"]),
                       "highpass_hz": float(prof["highpass_hz"]),
                       "input_rate": float(prof["fs_in"]),
                       "output_rate": float(prof["fs_out"]),
                       "trim_seconds": float(prof["trim_seconds"])},
        "window": {"seconds": float(prof["window_seconds"])},
    }


def preprocess_session(session: Session, prof: dict[str, Any]) -> dict[str, Any]:
    """Filter every continuous segment; return model columns and their clock."""
    if not session.segments:
        raise C.ContractError(f"{session.session_id}: 没有连续段")
    cfg = synth_config(session, prof)
    stream = StreamPreprocessor(cfg)
    values: list[np.ndarray] = []
    retained: list[np.ndarray] = []
    seg_ids: list[np.ndarray] = []
    segments: list[dict[str, Any]] = []
    col = 0
    for seg_no, (a, b) in enumerate(session.segments):
        if b - a < 2:
            continue
        stream.reset(int(session.indices[a]))
        y, ix = stream.process(session.eeg[:, a:b], session.indices[a:b])
        if not len(ix):
            continue
        new_id = len(segments)
        values.append(y)
        retained.append(ix)
        seg_ids.append(np.full(len(ix), new_id, np.int64))
        segments.append({"id": new_id, "first": int(session.indices[a]),
                         "last": int(session.indices[b - 1]),
                         "col_start": int(col), "col_stop": int(col + len(ix)),
                         "source_rows": [int(a), int(b)]})
        col += len(ix)
    if not values:
        raise C.ContractError(f"{session.session_id}: 剔除滤波启动段后没有数据")
    values = np.concatenate(values, axis=1)
    retained = np.concatenate(retained)
    seg_ids = np.concatenate(seg_ids)
    delay = float(prof["delay_seconds"]) if prof["timeline"] == "delay_compensated" else 0.0
    # ``retained`` holds RAW sample indices, so the clock divides by the
    # input rate, not by the output rate (a factor-4 trap).
    times = retained.astype(np.float64) / float(prof["fs_in"]) - delay
    return {"values": values, "retained": retained, "seg_id": seg_ids,
            "segments": segments, "times": times, "config": cfg,
            "history": int(stream.history_source_samples),
            "factor": int(stream.factor)}


@dataclass
class WindowBank:
    X: np.ndarray
    y: np.ndarray
    label: np.ndarray
    session: np.ndarray
    trial_key: np.ndarray
    trial_index: np.ndarray
    offset: np.ndarray
    day: np.ndarray
    channels: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_windows(self) -> int:
        return int(len(self.y))

    @property
    def n_trials(self) -> int:
        return int(len(set(self.trial_key.tolist())))

    def summary(self) -> dict[str, Any]:
        classes = {}
        for dim in range(C.OUTPUT_COUNT):
            mask = self.y == dim
            classes[str(dim)] = {
                "windows": int(mask.sum()),
                "trials": int(len(set(self.trial_key[mask].tolist()))) if mask.any() else 0,
            }
        per_session = {}
        for key in sorted(set(self.session.tolist())):
            mask = self.session == key
            per_session[key] = {
                "trials": int(len(set(self.trial_key[mask].tolist()))),
                "windows": int(mask.sum()),
                "classes": {str(d): int(((self.y == d) & mask).sum())
                            for d in range(C.OUTPUT_COUNT)},
            }
        return {"n_windows": self.n_windows, "n_trials": self.n_trials,
                "classes": classes, "sessions": per_session,
                "profile": self.meta.get("profile"),
                "rejected": self.meta.get("rejected", {})}

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out, X=self.X, y=self.y, label=self.label, session=self.session,
            trial_key=self.trial_key, trial_index=self.trial_index,
            offset=self.offset, day=self.day,
            channels=np.asarray(self.channels),
            meta_json=np.asarray(json.dumps(self.meta, ensure_ascii=False)),
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> "WindowBank":
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta_json"]))
        return cls(
            X=np.asarray(data["X"], np.float32), y=np.asarray(data["y"], np.int64),
            label=np.asarray(data["label"], np.int64),
            session=np.asarray(data["session"]).astype(str),
            trial_key=np.asarray(data["trial_key"]).astype(str),
            trial_index=np.asarray(data["trial_index"], np.int64),
            offset=np.asarray(data["offset"], np.float64),
            day=np.asarray(data["day"]).astype(str),
            channels=[str(c) for c in data["channels"]], meta=meta,
        )

    # ---- 目录形态：X 单独一份 X.npy（可 memmap），其余落 meta.npz ----------
    def save_dir(self, path: str | Path) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        if not (isinstance(self.X, np.memmap) and Path(self.X.filename) == (out / "X.npy")):
            np.save(out / "X.npy", np.asarray(self.X))
        np.savez(
            out / "meta.npz", y=self.y, label=self.label, session=self.session,
            trial_key=self.trial_key, trial_index=self.trial_index,
            offset=self.offset, day=self.day,
            channels=np.asarray(self.channels),
            meta_json=np.asarray(json.dumps(self.meta, ensure_ascii=False)),
        )
        return out

    @classmethod
    def load_dir(cls, path: str | Path) -> "WindowBank":
        out = Path(path)
        meta = np.load(out / "meta.npz", allow_pickle=False)
        X = np.load(out / "X.npy", mmap_mode="r")
        return cls(
            X=X, y=np.asarray(meta["y"], np.int64),
            label=np.asarray(meta["label"], np.int64),
            session=np.asarray(meta["session"]).astype(str),
            trial_key=np.asarray(meta["trial_key"]).astype(str),
            trial_index=np.asarray(meta["trial_index"], np.int64),
            offset=np.asarray(meta["offset"], np.float64),
            day=np.asarray(meta["day"]).astype(str),
            channels=[str(c) for c in meta["channels"]],
            meta=json.loads(str(meta["meta_json"])),
        )

    @classmethod
    def open(cls, path: str | Path) -> "WindowBank":
        target = Path(path)
        return cls.load_dir(target) if target.is_dir() else cls.load(target)


def _resolve_anchor(trial: dict[str, Any], fs: float, fallback_s: float,
                    anchor_mode: str) -> tuple[int, str]:
    """v2：一只试次的窗口锚点（源样点号）与它的来源。"""
    anchor = trial.get("anchor")
    if anchor is not None:
        try:
            return int(anchor), str(trial.get("anchor_kind") or "prepare")
        except (TypeError, ValueError):
            pass
    prepare_s = trial.get("prepare_seconds")
    try:
        prepare_s = float(prepare_s) if prepare_s is not None else None
    except (TypeError, ValueError):
        prepare_s = None
    if prepare_s is not None and prepare_s > 0:
        return int(round(float(trial["start"]) - prepare_s * float(fs))), "prepare_seconds"
    if anchor_mode == "prepare":
        return int(round(float(trial["start"]) - fallback_s * float(fs))), "profile_fallback"
    return int(trial["start"]), "cue"


def build_bank(sources: list[dict[str, Any]], profile_id: str, *,
               log: Callable[[str], None] = print,
               require_all_outputs: bool = True,
               out_dir: str | Path | None = None) -> WindowBank:
    """Build a window bank from reviewed sources.

    ``sources`` items: ``{path, mapping: {rows, many_to_one_confirmed}}``.

    ``out_dir`` 给出时按低内存方式落盘：每个会话的窗口先写临时 .npy，最后
    memmap 拼成 ``out_dir/X.npy``，小数组写 ``out_dir/meta.npz``，返回值里的
    ``X`` 是只读 memmap —— 20k 窗（>1 GB）不需要一次性进 RAM。
    """
    prof = C.profile(profile_id)
    anchor_mode = str(prof.get("anchor") or "cue")
    offsets = [float(x) for x in (prof.get("offsets") or [])]
    stride_s = float(prof.get("stride_seconds") or 0.0)
    post_s = float(prof.get("post_seconds") or 0.0)
    pre_s = float(prof.get("pre_seconds") or 0.0)
    fallback_s = float(prof.get("fallback_prepare_seconds", 2.0) or 2.0)
    min_windows = max(1, int(prof.get("min_windows_per_trial", 1) or 1))
    n_time = int(round(float(prof["window_seconds"]) * float(prof["fs_out"])))
    allc = C.MODEL_CHANNELS
    chunk_dir = None
    if out_dir is not None:
        chunk_dir = Path(out_dir) / "_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

    Xs: list[np.ndarray] = []
    chunks: list[tuple[Path, int]] = []
    ys, labels, sessions, keys, offsets_col, days = [], [], [], [], [], []
    supports: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    per_session: list[dict[str, Any]] = []
    trial_keys: list[str] = []
    trial_dims: list[int] = []
    last_outputs: list[dict[str, Any]] = []
    anchor_kinds: dict[str, int] = {}

    def reject(reason: str, n: int = 1) -> None:
        rejected[reason] = rejected.get(reason, 0) + int(n)

    seen_fingerprints: dict[str, str] = {}
    for item in sources:
        path = str(item.get("path") or "")
        if not path:
            raise C.ContractError("数据源缺少 path")
        session = load_session(path, with_eeg=True)
        if not session.complete and not bool(item.get("allow_incomplete")):
            raise C.ContractError(
                f"{session.session_id}: 记录未完整结束；如确需使用，请显式设置 allow_incomplete=true")
        fp = session.fingerprint()
        if fp in seen_fingerprints and seen_fingerprints[fp] != session.path:
            raise C.ContractError(
                f"同一段录音被重复导入：{seen_fingerprints[fp]} 与 {session.path}（内容指纹 {fp}）")
        seen_fingerprints[fp] = session.path
        inv = session.label_inventory()
        outputs = item.get("_outputs")
        if outputs is None:
            raise C.ContractError("内部错误：数据源缺少输出声明")
        last_outputs = outputs
        checked = validate_mapping(
            inv, (item.get("mapping") or {}).get("rows"),
            outputs, require_all_outputs=False,
            many_to_one_confirmed=bool((item.get("mapping") or {}).get("many_to_one_confirmed")),
        )
        cmap = checked["map"]
        for warning in checked["warnings"]:
            log(f"    [mapping] {session.session_id}: {warning}")
        idx = session.channel_index()
        missing = [c for c in allc if c not in idx]
        if missing:
            raise C.ContractError(f"{session.session_id}: 缺少模型通道 {missing}")
        order = [idx[c] for c in allc]
        prep = preprocess_session(session, prof)
        values = np.ascontiguousarray(prep["values"][order])
        retained = prep["retained"]
        times = prep["times"]
        seg_id = prep["seg_id"]
        segments = prep["segments"]
        history = prep["history"]
        factor = prep["factor"]
        fs_out = float(prof["fs_out"])
        fs_in = float(session.fs)
        boundary = prof.get("boundary") or {"mode": "strict_task", "task_end_slack_s": 0.0}
        boundary_mode = str(boundary.get("mode", "strict_task"))
        slack_s = float(boundary.get("task_end_slack_s", 0.0))
        boundary_overshoot = 0
        max_overshoot = 0.0
        local_trials = 0
        local_windows = 0
        class_trials = {i: 0 for i in range(C.OUTPUT_COUNT)}
        sess_X: list[np.ndarray] = []

        def segment_at(pos: int) -> dict[str, Any] | None:
            for seg in segments:
                if int(seg["first"]) <= int(pos) <= int(seg["last"]):
                    return seg
            return None

        for trial in session.trials:
            dim = cmap.get(int(trial["label"]))
            if dim is None:
                reject("unmapped_label")
                continue
            cue = int(trial["start"])
            end = int(trial["end"])
            seg = segment_at(cue)
            if seg is None:
                reject("cue_outside_segment")
                continue
            if not (int(seg["first"]) <= end <= int(seg["last"])):
                reject("trial_crosses_gap")
                continue
            origin, origin_kind = _resolve_anchor(trial, fs_in, fallback_s, anchor_mode)
            anchor_kinds[origin_kind] = anchor_kinds.get(origin_kind, 0) + 1
            origin_request_s = float(origin) / fs_in
            if anchor_mode == "prepare":
                if not origin < cue:
                    reject("anchor_after_cue")
                    continue
            if anchor_mode == "prepare":
                # 锚点吸附到最近的 250 Hz 列（prepare 的 sample_index 有 ±10 ms 级
                # 设备时钟抖动，不吸附会让整只试次的窗都判成 start_not_aligned）
                probe = int(np.searchsorted(times, origin_request_s, side="left"))
                candidates = [c for c in (probe - 1, probe) if 0 <= c < len(times)]
                if not candidates:
                    reject("anchor_outside_segment")
                    continue
                best = min(candidates, key=lambda c: abs(float(times[c]) - origin_request_s))
                snap_s = float(times[best]) - origin_request_s
                if abs(snap_s) > 0.05:  # 50 ms：超过就是锚点落在被裁掉/缺口区，不可信
                    reject("anchor_off_grid")
                    continue
                origin_s = float(times[best])
            else:
                origin_s = origin_request_s
            if anchor_mode == "prepare":
                allowed_end_s = float(end) / fs_in + post_s
                span = allowed_end_s - origin_s - float(prof["window_seconds"])
                if span < -1e-9 or stride_s <= 0:
                    reject("no_window_fits")
                    continue
                n_steps = int(np.floor(span / stride_s + 1e-9))
                wanted = [(origin_s + k * stride_s, k * stride_s) for k in range(n_steps + 1)]
            else:
                allowed_end_s = float(end) / fs_in + slack_s
                wanted = [(float(cue) / fs_in + off, off) for off in offsets]
            picked: list[tuple[int, float, float, float, float]] = []
            trial_ok = True
            for begin_s, off in wanted:
                first = int(np.searchsorted(times, begin_s, side="left"))
                if first < int(seg["col_start"]) or first + n_time > int(seg["col_stop"]):
                    reject("outside_segment_window")
                    if anchor_mode != "prepare":
                        trial_ok = False
                        break
                    continue
                if int(seg_id[first]) != int(seg["id"]) or int(seg_id[first + n_time - 1]) != int(seg["id"]):
                    reject("crosses_gap")
                    if anchor_mode != "prepare":
                        trial_ok = False
                        break
                    continue
                delta = float(times[first]) - begin_s
                if delta < -1e-9 or delta > 1.0 / fs_out + 1e-9:
                    reject("start_not_aligned")
                    if anchor_mode != "prepare":
                        trial_ok = False
                        break
                    continue
                span_raw = int(retained[first + n_time - 1]) - int(retained[first])
                if span_raw != (n_time - 1) * factor:
                    reject("non_contiguous_columns")
                    if anchor_mode != "prepare":
                        trial_ok = False
                        break
                    continue
                raw_start = int(retained[first]) - history
                raw_end = int(retained[first + n_time - 1]) + factor
                if raw_start < int(seg["first"]):
                    reject("no_filter_history")
                    if anchor_mode != "prepare":
                        trial_ok = False
                        break
                    continue
                overshoot = raw_end / fs_in - float(end) / fs_in
                if overshoot > (allowed_end_s - float(end) / fs_in) + 1e-9:
                    if anchor_mode != "prepare" and boundary_mode == "strict_task":
                        reject("beyond_task_end")
                        trial_ok = False
                        break
                    reject("beyond_post_window")
                    continue
                if overshoot > 1e-9:
                    boundary_overshoot += 1
                    max_overshoot = max(max_overshoot, overshoot)
                picked.append((first, float(off), begin_s,
                               raw_start / fs_in, raw_end / fs_in))
            if not trial_ok:
                continue
            if len(picked) < min_windows:
                reject("too_few_windows")
                continue
            key = f"{session.session_id}#{session.fingerprint()[:8]}|{trial['trial_id']}"
            if key in trial_keys:
                reject("duplicate_trial")
                continue
            trial_keys.append(key)
            trial_dims.append(int(dim))
            local_trials += 1
            class_trials[int(dim)] += 1
            for first, off, begin_s, support_start, support_end in picked:
                window = values[:, first:first + n_time]
                if chunk_dir is None:
                    Xs.append(window)
                else:
                    sess_X.append(window)
                ys.append(int(dim))
                labels.append(int(trial["label"]))
                sessions.append(session.session_id)
                keys.append(key)
                offsets_col.append(float(off))
                days.append(session.date)
                local_windows += 1
                supports.append({"trial_key": key, "offset": float(off),
                                 "begin_s": round(float(begin_s), 6),
                                 "support_start_s": round(support_start, 6),
                                 "support_end_s": round(support_end, 6),
                                 "cue_s": round(float(cue) / fs_in, 6),
                                 "task_end_s": round(float(end) / fs_in, 6),
                                 "anchor_s": round(origin_s, 6),
                                 "anchor_request_s": round(origin_request_s, 6),
                                 "anchor_snap_ms": round((origin_s - origin_request_s) * 1000.0, 3),
                                 "anchor_kind": origin_kind,
                                 "cue_minus_anchor_s": round((float(cue) - origin_s * fs_in) / fs_in, 6)})
        if chunk_dir is not None and sess_X:
            chunk = chunk_dir / f"{len(chunks):03d}.npy"
            np.save(chunk, np.ascontiguousarray(np.stack(sess_X)))
            chunks.append((chunk, int(len(sess_X))))
            sess_X.clear()
        log(f"  [{session.kind_label}] {session.session_id}: "
            f"试次 {local_trials}/{len(session.trials)} / 窗口 {local_windows} / "
            f"类别 {[(C.GLOBAL_NAMES.get(o['label_id'], o['name']), class_trials[i]) for i, o in enumerate(outputs)]}")
        per_session.append(dict(session_id=session.session_id, kind=session.kind,
                                path=session.path, fingerprint=session.fingerprint(),
                                trials_used=local_trials, trials_total=len(session.trials),
                                windows=local_windows, class_trials=class_trials,
                                boundary_overshoot_windows=int(boundary_overshoot),
                                boundary_max_overshoot_s=round(float(max_overshoot), 6),
                                mapping=list((item.get("mapping") or {}).get("rows") or [])))
    if chunk_dir is None and not Xs:
        raise C.ContractError(f"没有产出任何窗口；拒绝原因 {rejected}")
    if chunk_dir is not None and not chunks:
        raise C.ContractError(f"没有产出任何窗口；拒绝原因 {rejected}")

    n_total = int(sum(n for _, n in chunks)) if chunk_dir is not None else int(len(ys))
    n_channels = int(values.shape[0])
    if chunk_dir is None:
        X = np.ascontiguousarray(np.stack(Xs).astype(np.float32))
    else:
        target = Path(out_dir) / "X.npy"
        Xm = np.lib.format.open_memmap(target, mode="w+", dtype=np.float32,
                                       shape=(n_total, n_channels, n_time))
        at = 0
        for chunk, n in chunks:
            Xm[at:at + n] = np.load(chunk)
            at += n
            chunk.unlink()
        Xm.flush()
        del Xm
        chunk_dir.rmdir()
        X = np.load(target, mmap_mode="r")
    y = np.asarray(ys, np.int64)
    # trial-level bookkeeping: each unique key gets a contiguous index
    order_keys = {k: i for i, k in enumerate(trial_keys)}
    bank = WindowBank(
        X=X, y=y, label=np.asarray(labels, np.int64),
        session=np.asarray(sessions), trial_key=np.asarray(keys),
        trial_index=np.asarray([order_keys[k] for k in keys], np.int64),
        offset=np.asarray(offsets_col, np.float64), day=np.asarray(days),
        channels=list(allc),
        meta=dict(profile=prof, sources=per_session, rejected=rejected,
                  trial_dims=trial_dims, trial_keys=trial_keys,
                  offsets=offsets, stride_seconds=stride_s, anchor=anchor_mode,
                  pre_seconds=pre_s, post_seconds=post_s,
                  fallback_prepare_seconds=fallback_s,
                  anchor_kinds=anchor_kinds,
                  n_time=n_time, fs=float(prof["fs_out"]),
                  supports=supports,
                  boundary={"mode": str((prof.get("boundary") or {}).get("mode", "strict_task")),
                            "task_end_slack_s": float((prof.get("boundary") or {}).get("task_end_slack_s", 0.0)),
                            "overshoot_windows": int(sum(p.get("boundary_overshoot_windows", 0)
                                                          for p in per_session)),
                            "max_overshoot_s": round(float(max((p.get("boundary_max_overshoot_s", 0.0)
                                                                for p in per_session), default=0.0)), 6)},
                  outputs=list(outputs),
                  outputs_fingerprint=C.outputs_fingerprint(outputs)),
    )
    summary = bank.summary()
    empty = [i for i in range(C.OUTPUT_COUNT) if summary["classes"][str(i)]["trials"] == 0]
    if empty and require_all_outputs:
        names = [last_outputs[i]["name"] for i in empty]
        raise C.ContractError(f"以下输出列在取窗后没有试次：{list(zip(empty, names))}；"
                              f"拒绝原因 {rejected}")
    if out_dir is not None:
        bank.save_dir(out_dir)
    return bank
