"""已保存脑电的本机只读查看器；不连接设备，不修改原始记录。"""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import secrets
import threading
from urllib.parse import parse_qs, urlsplit
import webbrowser
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _lower_bound(dataset, value):
    lo, hi = 0, len(dataset)
    while lo < hi:
        middle = (lo + hi) // 2
        if int(dataset[middle]) < value:
            lo = middle + 1
        else:
            hi = middle
    return lo


def extrema_indices(values, max_bins=1000):
    """按时间顺序保留各区间的最小/最大值，概览不直接跳采丢失尖峰。"""
    n = len(values)
    if n <= max_bins * 2:
        return np.arange(n, dtype=np.int64)
    width = int(math.ceil(n / max_bins))
    count = n // width
    block = values[:count * width].reshape(count, width)
    finite = np.isfinite(block)
    lows = np.argmin(np.where(finite, block, np.inf), axis=1)
    highs = np.argmax(np.where(finite, block, -np.inf), axis=1)
    offsets = np.arange(count) * width
    picks = np.concatenate((offsets + lows, offsets + highs,
                            np.arange(count * width, n), [0, n - 1]))
    return np.unique(picks)


class RecordStore:
    def __init__(self, records_root):
        self.root = Path(records_root).resolve()

    def path(self, identity):
        if not isinstance(identity, str) or not identity or len(identity) > 1000:
            raise ValueError("请先选择一份记录。")
        path = (self.root / identity).resolve()
        if not path.is_relative_to(self.root) or not path.is_dir():
            raise ValueError("记录路径不在已配置的采集目录中。")
        if not (path / "session.json").is_file() or not (path / "data.h5").is_file():
            raise ValueError("记录尚未结束或缺少文件；请等采集完成后刷新。")
        return path

    def metadata(self, identity):
        path = self.path(identity)
        meta = json.loads((path / "session.json").read_text(encoding="utf-8-sig"))
        if meta.get("format") != "handbci.raw.v1":
            raise ValueError("仅支持本项目保存的session记录，厂商文件需另行转换。")
        return path, meta

    def sessions(self):
        rows, skipped = [], 0
        if self.root.is_dir():
            for item in self.root.rglob("session.json"):
                try:
                    if not item.resolve().is_relative_to(self.root):
                        continue
                    identity = item.parent.relative_to(self.root).as_posix()
                    path, meta = self.metadata(identity)
                    cfg = meta["config"]
                    report_path = path / "collection_report.json"
                    report = json.loads(report_path.read_text(encoding="utf-8-sig")) if report_path.is_file() else {}
                    rows.append({"id": identity, "name": meta.get("display_name") or path.name,
                                 "participant": str(cfg.get("session_metadata", {}).get("participant_id", "未填写")),
                                 "source": meta.get("source", "unknown"), "complete": bool(meta.get("complete")),
                                 "n_channels": len(cfg["device"]["channels"]),
                                 "duration": report.get("signal_duration_seconds"),
                                 "description": cfg.get("description", "")})
                except (OSError, ValueError, KeyError, TypeError):
                    skipped += 1
        return {"root": str(self.root), "sessions": sorted(rows, key=lambda row: row["id"], reverse=True),
                "skipped": skipped}

    def details(self, identity):
        path, meta = self.metadata(identity)
        cfg = meta["config"]
        device = cfg["device"]
        fs = float(device["sample_rate"])
        if not math.isfinite(fs) or fs <= 0:
            raise ValueError("记录中的采样率无效。")
        with h5py.File(path / "data.h5", "r") as handle:
            if handle.attrs.get("handbci_format") != "handbci.raw.v1":
                raise ValueError("HDF5记录尚未封存；请稍后刷新，或检查采集错误。")
            shape = handle["eeg"].shape
            if len(shape) != 2 or shape[0] != len(device["channels"]):
                raise ValueError("通道配置与原始数据尺寸不一致。")
            n = shape[1]
            indices = handle["source_indices"]
            if len(indices) != n or not n:
                raise ValueError("记录没有可显示的采样点，或采样索引尺寸不一致。")
            first, last = int(indices[0]), int(indices[-1])
            if last < first:
                raise ValueError("采样索引发生重置，请先检查原始记录。")
            events = [json.loads(event) for event in handle["events_json"].asstr()[:]]
            complete = bool(meta.get("complete") and handle.attrs.get("complete", False))
        classes = {int(c["id"]): c for c in cfg.get("classes", [])}
        starts, trials = {}, []
        for event in events:
            trial_id = event.get("trial_id")
            if event.get("kind") == "cue":
                starts[trial_id] = event
            elif event.get("kind") == "task_end" and trial_id in starts:
                begin = starts.pop(trial_id)
                c = classes.get(begin.get("label"), {})
                trials.append({"id": trial_id, "label": begin.get("label"),
                               "task": c.get("task_text", "未知任务"),
                               "start": (int(begin["sample_index"]) - first) / fs,
                               "end": (int(event["sample_index"]) - first) / fs, "complete": True})
        for trial_id, begin in starts.items():
            trials.append({"id": trial_id, "label": begin.get("label"),
                           "task": classes.get(begin.get("label"), {}).get("task_text", "未知任务"),
                           "start": (int(begin["sample_index"]) - first) / fs, "end": None, "complete": False})
        return {"id": identity, "path": str(path), "participant": str(cfg.get("session_metadata", {}).get("participant_id", "未填写")),
                "source": meta.get("source", "unknown"), "complete": complete, "error": meta.get("error", ""),
                "description": cfg.get("description", ""), "channels": device["channels"],
                "unit": device.get("unit", "unknown"), "sample_rate": fs, "samples": n,
                "duration": (last - first + 1) / fs, "origin_index": first,
                "classes": list(classes.values()), "trials": sorted(trials, key=lambda row: row["start"]),
                "marking_types": sorted({e.get("marking", "unknown") for e in events})}

    def window(self, identity, channels, start, end):
        info = self.details(identity)
        if not isinstance(channels, list) or not 1 <= len(channels) <= 8:
            raise ValueError("每次请选择1至8个通道。")
        if any(isinstance(ch, bool) or not isinstance(ch, int) or not 0 <= ch < len(info["channels"]) for ch in channels):
            raise ValueError("通道编号无效。")
        if len(set(channels)) != len(channels):
            raise ValueError("不能重复选择同一通道。")
        if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end or start >= info["duration"]:
            raise ValueError("请填写有效的开始、结束时间，且开始时间须在记录范围内。")
        end = min(end, info["duration"])
        fs, origin = info["sample_rate"], info["origin_index"]
        traces = []
        with h5py.File(self.path(identity) / "data.h5", "r") as handle:
            indices = handle["source_indices"]
            left = _lower_bound(indices, math.ceil(start * fs + origin - 1e-7))
            right = _lower_bound(indices, math.ceil(end * fs + origin - 1e-7))
            if right - left < 2:
                raise ValueError("所选区间少于两个采样点，请扩大时间范围。")
            source_indices = np.asarray(indices[left:right], dtype=np.int64)
            gaps = int(np.count_nonzero(np.diff(source_indices) != 1))
            if np.any(np.diff(source_indices) <= 0):
                raise ValueError("此区间采样索引有重置或倒序，无法按连续时间绘制。")
            for ch in channels:
                values = np.asarray(handle["eeg"][ch, left:right], dtype=np.float32)
                picks = extrema_indices(values)
                x = (source_indices[picks] - origin) / fs
                y = values[picks]
                points = []
                for k in range(len(picks)):
                    if k and gaps and source_indices[picks[k]] - source_indices[picks[k-1]] != picks[k] - picks[k-1]:
                        points.append(None)
                    points.append([float(x[k]), float(y[k])] if np.isfinite(y[k]) else None)
                traces.append({"name": info["channels"][ch], "index": ch, "points": points,
                               "invalid_samples": int(np.count_nonzero(~np.isfinite(values)))})
        return {"id": identity, "start": start, "end": end, "n_samples": right-left,
                "compressed": right-left > 2000, "gaps": gaps, "unit": info["unit"], "traces": traces,
                "trials": [trial for trial in info["trials"]
                           if trial["start"] < end and (trial["end"] is None or trial["end"] > start)]}


def create_viewer_server(store, port=0):
    token = secrets.token_hex(24)
    assets = Path(__file__).resolve().parent / "viewer_ui"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, value, mime="application/json; charset=utf-8"):
            body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def valid_host(self):
            return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

        def do_GET(self):
            if not self.valid_host():
                return self.reply(403, {"error": "仅允许本机查看。"})
            query = urlsplit(self.path)
            args = parse_qs(query.query)
            try:
                if query.path in {"/", "/app.js", "/styles.css"}:
                    name = "index.html" if query.path == "/" else query.path[1:]
                    body = (assets / name).read_bytes().replace(b"__VIEWER_TOKEN__", token.encode())
                    mime = {"index.html": "text/html; charset=utf-8", "app.js": "text/javascript; charset=utf-8", "styles.css": "text/css; charset=utf-8"}[name]
                    return self.reply(200, body, mime)
                if query.path == "/api/sessions":
                    return self.reply(200, store.sessions())
                identity = args.get("id", [""])[0]
                if query.path == "/api/session":
                    return self.reply(200, store.details(identity))
                if query.path == "/api/window":
                    channels = [int(value) for value in args.get("channels", ["0"])[0].split(",")]
                    return self.reply(200, store.window(identity, channels,
                        float(args.get("start", ["0"])[0]), float(args.get("end", ["10"])[0])))
                return self.reply(404, {"error": "页面不存在。"})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                return self.reply(400, {"error": str(exc)})

        def do_POST(self):
            if not self.valid_host() or self.headers.get("X-Viewer-Token") != token:
                return self.reply(403, {"error": "请求来源不正确。"})
            if self.path != "/api/shutdown":
                return self.reply(404, {"error": "操作不存在。"})
            self.reply(200, {"closed": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()

    return ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description="查看本项目已保存的脑电波形")
    parser.add_argument("--records", type=Path, default=ROOT / "runtime/records")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    server = create_viewer_server(RecordStore(args.records), args.port)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"波形查看页面：{url}", flush=True)
    print(f"读取记录目录：{args.records.resolve()}", flush=True)
    print("采集结束后刷新列表即可查看；退出时点击页面中的“关闭查看器”。", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
