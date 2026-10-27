"""Preserve forwarded TCP bytes and inspect them without attempting signal repair."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import threading
import time
import uuid

import numpy as np

from .config import load_config
from .receiver_access import ReceiverLease
from .vendor.neuracle_recorder import RingBuffer, _NeuracleThread

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def inspect_capture(folder: str | Path) -> dict:
    """Compare an independent frame decode with the production streaming parser.

    Both decodes assume the configured headerless little-endian float32 stream.
    Agreement does not establish that the sender actually uses this protocol.
    """
    folder = Path(folder)
    meta = json.loads((folder / "capture.json").read_text(encoding="utf-8"))
    blob = (folder / "wire.bin").read_bytes()
    chunks = [json.loads(line) for line in (folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    offset = 0
    for chunk in chunks:
        if chunk["offset"] != offset or chunk["length"] <= 0:
            raise ValueError("原始字节记录与接收日志不连续，不能据此比较解析器。")
        offset += chunk["length"]
    if offset != len(blob):
        raise ValueError("原始字节数与日志不一致；请保留文件排查，不自动补齐。")
    n_eeg, fs = int(meta["declared_eeg_channels"]), float(meta["declared_sample_rate_hz"])
    n_channels, frame_bytes = n_eeg + 1, (n_eeg + 1) * 4
    complete_bytes = len(blob) - len(blob) % frame_bytes
    words = np.frombuffer(blob[:complete_bytes], dtype="<u4").reshape(-1, n_channels)
    eeg = words[:, :n_eeg].copy().view("<f4").T
    trigger = words[:, -1]
    dummy = socket.socket()
    receiver = _NeuracleThread(RingBuffer(n_channels, 4), n_channels, fs,
                               threading.Event(), dummy, recv_bytes=65536)
    parts = []
    try:
        for chunk in chunks:
            a = chunk["offset"]
            decoded = receiver._parse(blob[a:a + chunk["length"]])
            if decoded.shape[1]:
                parts.append(decoded)
        parsed = np.concatenate(parts, axis=1) if parts else np.empty((n_channels, 0))
        parser_matches = (np.array_equal(parsed[:n_eeg], eeg, equal_nan=True)
                          and np.array_equal(parsed[-1].astype(np.uint32), trigger)
                          and receiver.byte_remainder == blob[complete_bytes:])
    finally:
        dummy.close()
    repeats = np.all(eeg[:, 1:] == eeg[:, :-1], axis=0) if eeg.shape[1] > 1 else np.array([], bool)
    edges = np.r_[0, np.flatnonzero(~repeats) + 1, eeg.shape[1]] if eeg.shape[1] else np.array([0])
    run_counts = Counter(np.diff(edges).tolist())
    started, finished = meta.get("started_mono_ns"), meta.get("finished_mono_ns")
    elapsed = (finished - started) / 1e9 if started is not None and finished is not None else 0
    seconds = Counter()
    for chunk in chunks:
        bin_id = max(0, int((chunk["received_mono_ns"] - started) / 1e9))
        seconds[bin_id] += chunk["length"]
    gaps = np.diff([c["received_mono_ns"] for c in chunks]) / 1e6
    result = {
        "purpose": "forwarded_stream_diagnostics_not_training_data",
        "capture_status": meta["status"], "capture_error": meta.get("error", ""),
        "wire_bytes": len(blob), "wire_sha256": hashlib.sha256(blob).hexdigest(),
        "host_receive_chunks": len(chunks), "complete_frames_under_configured_layout": eeg.shape[1],
        "trailing_bytes_preserved": len(blob) - complete_bytes,
        "declared_eeg_channels": n_eeg, "declared_sample_rate_hz": fs,
        "capture_elapsed_seconds": elapsed,
        "received_frames_per_host_second": eeg.shape[1] / elapsed if elapsed > 0 else None,
        "nominal_duration_from_declared_rate_seconds": eeg.shape[1] / fs,
        "nonfinite_eeg_values_under_configured_layout": int(np.count_nonzero(~np.isfinite(eeg))),
        "all_channels_adjacent_repeat_fraction": float(repeats.mean()) if repeats.size else None,
        "identical_vector_run_counts": dict(sorted(run_counts.items())),
        "production_parser_matches_independent_decode": bool(parser_matches) if eeg.shape[1] else None,
        "received_bytes_by_host_second": dict(sorted(seconds.items())),
        "max_host_receive_gap_ms": float(gaps.max()) if gaps.size else None,
        "physical_sample_rate_verified": False, "signal_repaired": False,
        "limitations": [
            "Layout is an assumption: little-endian, sample-major, EEG float32 plus one 4-byte trigger.",
            "recv chunks are not TCP packets or amplifier sample timestamps.",
            "A partial final frame may come from stopping mid-stream; its bytes are preserved.",
            "Parser agreement does not prove vendor protocol, units, or physical sample rate.",
            "No deduplication, interpolation, filtering, or inferred sampling-rate correction.",
        ],
    }
    _write_json(folder / "inspection.json", result)
    repeat_text = "无足够样本" if not repeats.size else f"{repeats.mean()*100:.4f}%"
    comparison = "数据不足" if not eeg.shape[1] else "一致" if parser_matches else "不一致"
    report = f"""# 端口原始字节诊断

状态：{meta["status"]}。错误：{meta.get("error") or "无"}。

- 原始收到字节：{len(blob):,}；主机接收批次：{len(chunks)}。
- 按配置帧格式解读：{eeg.shape[1]:,}个完整采样帧，末尾{len(blob)-complete_bytes}字节原样保留。
- 生产解析器与独立解析的比较：{comparison}。
- 全通道相邻重复比例：{repeat_text}。
- 真实采样率、真实单位及厂家实际协议均尚未验证；以上数量基于所选配置。

wire.bin保存收到的原始字节；chunks.jsonl记录每次recv返回的字节区间和主机到达时间。
这些批次不等于网络数据包，也不是设备采样时间戳。capture.json保存配置和运行状态。
inspection.json保存统计。本工具不生成训练记录，不改变重复值或补造采样点。

若wire.bin按已确认的协议解析后已含重复值，本端后续滤波无法修复源端缺失信息。
若原始字节正常而生产解析不一致，可凭此定位接收解析错误。
当前尚未确认实际协议时，即使两种解析一致，也不能据此认定接收端已经适配正确。
"""
    (folder / "诊断说明.md").write_text(report, encoding="utf-8")
    return result


def capture_stream(*, host: str, port: int, n_eeg: int, declared_fs: float,
                   seconds: float, output_root: str | Path,
                   origin: str = "unverified_neuracle_forwarding", max_bytes: int = 256 * 1024 * 1024,
                   stop_event=None, progress=print) -> Path:
    with ReceiverLease(host, port):
        return _capture_stream(host=host, port=port, n_eeg=n_eeg, declared_fs=declared_fs,
                               seconds=seconds, output_root=output_root, origin=origin,
                               max_bytes=max_bytes, stop_event=stop_event, progress=progress)


def _capture_stream(*, host, port, n_eeg, declared_fs, seconds, output_root,
                    origin, max_bytes, stop_event, progress):
    if not (1 <= n_eeg <= 256 and 0 < declared_fs and math.isfinite(declared_fs)
            and 0 < seconds <= 3600 and math.isfinite(seconds) and max_bytes > 0):
        raise ValueError("通道数、声明采样率、记录时长或容量限制无效。")
    root = Path(output_root)
    folder = root / (time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8] + "_WIRE_DIAG")
    folder.mkdir(parents=True, exist_ok=False)
    meta = {"purpose": "raw_tcp_diagnostics", "origin": origin, "host": host, "port": int(port),
            "declared_eeg_channels": int(n_eeg), "declared_stream_channels": int(n_eeg + 1),
            "declared_sample_rate_hz": float(declared_fs), "requested_seconds": seconds,
            "started_wall_ns": None, "started_mono_ns": None, "finished_mono_ns": None,
            "status": "connecting", "error": "", "bytes_saved": 0, "physical_sample_rate_verified": False}
    _write_json(folder / "capture.json", meta)
    connection = None
    try:
        with (folder / "wire.bin").open("xb") as wire, (folder / "chunks.jsonl").open("x", encoding="utf-8") as journal:
            try:
                connection = socket.create_connection((host, int(port)), timeout=3)
                connection.settimeout(.25)
                meta.update(status="capturing", started_wall_ns=time.time_ns(), started_mono_ns=time.monotonic_ns())
                deadline = meta["started_mono_ns"] + int(seconds * 1e9)
                next_progress = meta["started_mono_ns"]
                while True:
                    now = time.monotonic_ns()
                    if stop_event is not None and stop_event.is_set():
                        meta["status"] = "stopped"
                        break
                    if now >= deadline:
                        meta["status"] = "duration_elapsed"
                        break
                    connection.settimeout(max(.001, min(.25, (deadline-now)/1e9)))
                    try:
                        block = connection.recv(65536)
                    except socket.timeout:
                        continue
                    mono, wall = time.monotonic_ns(), time.time_ns()
                    if not block:
                        meta["status"] = "peer_closed"
                        break
                    if meta["bytes_saved"] + len(block) > max_bytes:
                        meta.update(status="byte_limit", error="达到诊断容量限制；该次未完整保存，不能据此排除丢段。")
                        break
                    row = {"offset": meta["bytes_saved"], "length": len(block),
                           "received_wall_ns": wall, "received_mono_ns": mono}
                    wire.write(block)
                    journal.write(json.dumps(row) + "\n")
                    meta["bytes_saved"] += len(block)
                    if mono >= next_progress:
                        progress(f"已保存原始转发字节 {meta['bytes_saved']:,}；当前只做诊断，不进行训练。")
                        next_progress = mono + 5_000_000_000
            finally:
                wire.flush()
                journal.flush()
                os.fsync(wire.fileno())
                os.fsync(journal.fileno())
    except KeyboardInterrupt:
        meta.update(status="interrupted", error="操作员中断；已收到的原始字节保留。")
    except OSError as exc:
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if connection is not None:
            connection.close()
        meta["finished_mono_ns"] = time.monotonic_ns()
        _write_json(folder / "capture.json", meta)
    inspect_capture(folder)
    return folder


def main():
    ap = argparse.ArgumentParser(description="Neuracle原始转发诊断：保留字节，不自动修复信号。")
    commands = ap.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="连接厂家转发端口并保留原始字节")
    capture.add_argument("--config", type=Path, default=ROOT / "configs/collection_ui_neuracle_ch32.json")
    capture.add_argument("--seconds", type=float, default=60)
    capture.add_argument("--output-root", type=Path, default=ROOT / "runtime/stream_diagnostics")
    capture.add_argument("--host")
    capture.add_argument("--port", type=int)
    inspect = commands.add_parser("inspect", help="离线重新分析已记录的原始字节")
    inspect.add_argument("folder", type=Path)
    args = ap.parse_args()
    if args.command == "inspect":
        result = inspect_capture(args.folder)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    cfg = load_config(args.config)
    device = cfg["device"]
    print("请先停止其他接收程序，开启厂家数据转发；本工具只接收并保存字节，不向设备发送控制命令。", flush=True)
    folder = capture_stream(host=args.host or device.get("host", "127.0.0.1"),
                            port=args.port if args.port is not None else device.get("port", 8712),
                            n_eeg=len(device["channels"]), declared_fs=device["sample_rate"],
                            seconds=args.seconds, output_root=args.output_root,
                            origin="configured_" + device["source"])
    result = json.loads((folder / "inspection.json").read_text(encoding="utf-8"))
    print(f"诊断结果已保存：{folder.resolve()}", flush=True)
    print(f"状态：{result['capture_status']}；收到{result['wire_bytes']:,}字节。", flush=True)
    if result["capture_error"]:
        print(result["capture_error"], flush=True)
    return 0 if result["capture_status"] == "duration_elapsed" and result["wire_bytes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
