"""可自描述HDF5记录、事件、按完整试次切窗和分组划分。"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import uuid

import h5py
import numpy as np

from .config import require_compatible, validate_config, write_json
from .preprocess import StreamPreprocessor, preprocess_record

FORMAT = "handbci.raw.v1"


def new_session(root: str | Path, name: str) -> Path:
    path = Path(root) / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def seal_session(path: str | Path, config: dict, events: list[dict], *, complete: bool,
                 source: str, error: str = "") -> None:
    path = Path(path)
    metadata = {"format": FORMAT, "session_id": path.name + "-" + uuid.uuid4().hex, "config": config,
                "complete": bool(complete), "source": source, "error": error}
    write_json(path / "session.json", metadata)
    with (path / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    with h5py.File(path / "data.h5", "a") as handle:
        handle.attrs["handbci_format"] = FORMAT
        handle.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
        handle.create_dataset("events_json", data=[json.dumps(e, ensure_ascii=False) for e in events],
                              dtype=h5py.string_dtype("utf-8"))


def load_session(path: str | Path) -> dict:
    path = Path(path)
    h5_path = path / "data.h5" if path.is_dir() else path
    with h5py.File(h5_path, "r") as handle:
        if handle.attrs.get("handbci_format") != FORMAT:
            raise ValueError("不是整理版记录格式；旧工程数据需先明确标签和时间对齐后转换。")
        meta = json.loads(handle.attrs["metadata_json"])
        if "quality" in meta and not meta["quality"].get("training_compatible", False):
            raise ValueError("该记录的时钟、标签或单位未满足训练条件，请查看quality。")
        if not meta["complete"] or not bool(handle.attrs.get("complete", False)):
            raise ValueError("拒绝把未完整结束的记录直接用于训练。")
        events = [json.loads(e) for e in handle["events_json"].asstr()[:]]
        eeg = np.asarray(handle["eeg"], dtype=np.float32)
        indices = np.asarray(handle["source_indices"], dtype=np.int64)
        trigger = np.asarray(handle["trigger"], dtype=np.uint32)
        if "device_time_ms" in handle:
            stamps = np.asarray(handle["device_time_ms"], dtype=np.float64)
            fs = float(meta["config"]["device"]["sample_rate"])
            if not np.all(np.isfinite(stamps)) or np.any(np.diff(stamps) <= 0):
                raise ValueError("设备时钟重置或重复，拒绝训练。")
            # Fractional millisecond packet clocks can quantize by half a millisecond.
            pts = meta.get("device", {}).get("settings", {}).get("points", 1)
            tolerance = .501 if (1000*pts/fs) % 1 else .001
            if np.any(np.abs(np.diff(stamps)-1000/fs) > tolerance):
                raise ValueError("设备时间存在缺口；连续source_indices不能证明信号连续。")
    config = validate_config(meta["config"])
    if eeg.shape != (len(config["device"]["channels"]), len(indices)) or len(trigger) != len(indices):
        raise ValueError("EEG、trigger和索引尺寸不一致。")
    if not len(indices) or np.any(np.diff(indices) != 1):
        raise ValueError("记录含重连/丢段/索引重置，须先拆成独立连续会话。")
    if not np.all(np.isfinite(eeg)):
        raise ValueError("原始脑电含无效数值。")
    return {"path": str(h5_path.resolve()), "metadata": meta, "config": config,
            "eeg": eeg, "indices": indices, "trigger": trigger, "events": events}


def trials_from_events(events: list[dict], n_classes: int) -> list[dict]:
    active = None
    trials, seen = [], set()
    for event in events:
        if event["kind"] == "cue":
            if active is not None:
                raise ValueError("试次提示重叠或缺少结束事件。")
            if not 0 <= event["label"] < n_classes or event["trial_id"] in seen:
                raise ValueError("类别越界或试次编号重复。")
            active = event
            seen.add(event["trial_id"])
        elif event["kind"] == "task_end":
            if active is None or active["trial_id"] != event["trial_id"]:
                raise ValueError("任务结束事件无法与提示一一匹配。")
            if event["sample_index"] <= active["sample_index"]:
                raise ValueError("任务结束采样点必须晚于起点。")
            trials.append({"trial_id": active["trial_id"], "label": active["label"],
                           "start": active["sample_index"], "end": event["sample_index"]})
            active = None
    if active is not None:
        raise ValueError("最后一个试次没有结束事件。")
    if not trials:
        raise ValueError("记录中没有完整试次。")
    return trials


@dataclass
class WindowDataset:
    X: np.ndarray
    y: np.ndarray
    rows: list[dict]
    config: dict

    def subset(self, indices):
        ix = np.asarray(indices, dtype=np.int64)
        return WindowDataset(self.X[ix], self.y[ix], [self.rows[i] for i in ix], self.config)

    def __len__(self):
        return len(self.y)


def build_dataset(paths: list[str | Path], expected_config: dict | None = None) -> WindowDataset:
    xs, ys, rows = [], [], []
    cfg = expected_config
    seen_sessions = set()
    for path in paths:
        record = load_session(path)
        config = record["config"]
        if cfg is None:
            cfg = config
        require_compatible(cfg, config)
        identity = record["metadata"]["session_id"]
        if identity in seen_sessions:
            raise ValueError("同一会话重复输入（即使文件被复制也不能算独立会话）。")
        seen_sessions.add(identity)
        processed, retained = preprocess_record(config, record["eeg"], record["indices"])
        prep = StreamPreprocessor(config)
        fs = prep.output_fs
        window = config["window"]
        n_time = int(round(window["seconds"] * fs))
        stride = max(1, int(round(window["stride_seconds"] * fs)))
        # 将处理后采样点按名义FIR群延迟对应回输入时间；记录原始索引便于核对。
        input_seconds = retained / prep.fs - prep.delay_seconds
        trials = trials_from_events(record["events"], len(cfg["classes"]))
        for trial in trials:
            begin = trial["start"] / prep.fs + window["task_offset_seconds"]
            end = trial["end"] / prep.fs
            first = int(np.searchsorted(input_seconds, begin))
            last = int(np.searchsorted(input_seconds, end))
            count_before = len(xs)
            for offset in range(first, last - n_time + 1, stride):
                xs.append(processed[:, offset:offset + n_time])
                ys.append(trial["label"])
                rows.append({"session": identity, "trial_id": trial["trial_id"],
                             "group": f"{identity}:{trial['trial_id']}", "path": record["path"],
                             "source": record["metadata"]["source"],
                             "raw_start": int(retained[offset]), "raw_end": int(retained[offset + n_time - 1]) + prep.factor,
                             "trial_start": trial["start"], "trial_end": trial["end"]})
            if len(xs) == count_before:
                raise ValueError(f"试次{trial['trial_id']}过短，不能生成配置要求的完整窗口。")
    if not xs or cfg is None:
        raise ValueError("没有可训练窗口。")
    return WindowDataset(np.stack(xs).astype(np.float32), np.asarray(ys, np.int64), rows, cfg)


def split_dataset(ds: WindowDataset, mode: str = "session", seed: int = 42):
    if mode == "session":
        sessions = list(dict.fromkeys(row["session"] for row in ds.rows))
        if len(sessions) < 3:
            raise ValueError("会话划分至少需要3份独立记录；可显式使用--split trial作会话内验证。")
        train_ids, valid_id, test_id = set(sessions[:-2]), sessions[-2], sessions[-1]
        indices = [[i for i, r in enumerate(ds.rows) if r["session"] in train_ids],
                   [i for i, r in enumerate(ds.rows) if r["session"] == valid_id],
                   [i for i, r in enumerate(ds.rows) if r["session"] == test_id]]
    elif mode == "trial":
        indices = [[], [], []]
        rng = np.random.default_rng(seed)
        for label in range(len(ds.config["classes"])):
            groups = sorted({r["group"] for i, r in enumerate(ds.rows) if ds.y[i] == label})
            if len(groups) < 3:
                raise ValueError("按试次划分时，每类至少需要3个完整试次。")
            rng.shuffle(groups)
            assignment = {g: (1 if k == 0 else 2 if k == 1 else 0) for k, g in enumerate(groups)}
            for i, row in enumerate(ds.rows):
                if ds.y[i] == label:
                    indices[assignment[row["group"]]].append(i)
    else:
        raise ValueError("split只支持session或trial。")
    parts = tuple(ds.subset(ix) for ix in indices)
    require_disjoint(*parts)
    for part in parts:
        if set(part.y.tolist()) != set(range(len(ds.config["classes"]))):
            raise ValueError("训练/验证/测试的每一部分都必须覆盖所有类别。")
    return parts


def require_disjoint(*datasets: WindowDataset) -> None:
    for i, first in enumerate(datasets):
        for second in datasets[i + 1:]:
            if {r["group"] for r in first.rows} & {r["group"] for r in second.rows}:
                raise ValueError("训练/验证/测试重复使用了同一完整试次。")
            for session in {r["session"] for r in first.rows} & {r["session"] for r in second.rows}:
                spans_a = [(r["raw_start"], r["raw_end"]) for r in first.rows if r["session"] == session]
                spans_b = [(r["raw_start"], r["raw_end"]) for r in second.rows if r["session"] == session]
                if any(a < d and c < b for a, b in spans_a for c, d in spans_b):
                    raise ValueError("不同数据划分共享了原始采样点。")


def make_synthetic_session(config: dict, root: str | Path, name: str, seed: int) -> Path:
    """人工构造类别差异，只验证软件链路，不代表真实运动想象可分性。"""
    cfg = validate_config(config)
    path = new_session(root, name)
    rng = np.random.default_rng(seed)
    fs = int(cfg["device"]["sample_rate"])
    n_ch = len(cfg["device"]["channels"])
    collection = cfg["collection"]
    labels = []
    for _ in range(collection["loops"]):
        labels.extend(rng.permutation(len(cfg["classes"])).tolist())
    initial = int(collection["initial_rest_seconds"] * fs)
    task = int(collection["task_seconds"] * fs)
    rest = int(collection["rest_seconds"] * fs)
    total = initial + len(labels) * (task + rest) + fs
    eeg = rng.normal(0, 0.3, (n_ch, total)).astype(np.float32)
    trigger = np.zeros(total, np.uint32)
    events = []
    frequencies = np.linspace(6, 35, len(cfg["classes"]))
    cursor = initial
    for trial_id, label in enumerate(labels, 1):
        times = np.arange(cursor, cursor + task) / fs
        spatial = 0.5 + np.roll(np.linspace(0.2, 1.5, n_ch), label % n_ch)
        eeg[:, cursor:cursor + task] += (4 * spatial[:, None] * np.sin(2 * np.pi * frequencies[label] * times)).astype(np.float32)
        for kind, position, code in (("cue", cursor, label + 1), ("task_end", cursor + task, 9)):
            trigger[position] = code
            events.append({"kind": kind, "trial_id": trial_id, "label": label,
                           "sample_index": position, "event_id": code, "marking": "synthetic"})
        cursor += task + rest
    with h5py.File(path / "data.h5", "w") as handle:
        handle.create_dataset("eeg", data=eeg)
        handle.create_dataset("trigger", data=trigger)
        handle.create_dataset("source_indices", data=np.arange(total, dtype=np.int64))
        handle.attrs["srate"] = fs
        handle.attrs["complete"] = True
    seal_session(path, cfg, events, complete=True, source="synthetic")
    return path
