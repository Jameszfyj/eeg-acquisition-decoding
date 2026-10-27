# -*- coding: utf-8 -*-
"""Threaded stdlib web server: a constrained client of the studio CLI.

It never trains in-process.  A submitted job becomes a JSON file plus an
allowlisted ``python -m mi_studio run --config ...`` subprocess; the browser
only reads that job's status, log and artifact links.
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .. import contracts as C
from ..adapters import detect_kind, scan

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
ROOT = HERE.parent.parent
JOB_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")
MAX_BODY = 4 * 1024 * 1024


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _json_default(value):
    try:
        import numpy as np
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return str(value)


@dataclass
class Job:
    job_id: str
    directory: Path
    kind: str
    process: subprocess.Popen | None = None
    created_at: str = field(default_factory=_now)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def status_path(self) -> Path:
        return self.directory / "status.json"

    @property
    def log_path(self) -> Path:
        return self.directory / "job.log"

    def status(self) -> dict[str, Any]:
        data: dict[str, Any] = {"job_id": self.job_id, "kind": self.kind,
                                "directory": str(self.directory),
                                "created_at": self.created_at}
        if self.status_path.is_file():
            try:
                data.update(json.loads(self.status_path.read_text(encoding="utf-8")))
            except ValueError:
                data["state"] = "unreadable"
        else:
            data["state"] = "starting"
        if self.process is not None:
            code = self.process.poll()
            data["pid"] = self.process.pid
            data["returncode"] = code
            if code is not None and data.get("state") not in ("done", "error"):
                data["state"] = "error"
                data["error"] = data.get("error") or f"CLI exited with {code}"
        result = self.directory / "result.json"
        if result.is_file() and "result" not in data:
            try:
                data["result_available"] = True
            except Exception:
                pass
        return data

    def log_chunk(self, offset: int = 0) -> tuple[str, int]:
        if not self.log_path.is_file():
            return "", 0
        size = self.log_path.stat().st_size
        if offset > size:
            offset = 0
        with self.log_path.open("rb") as fh:
            fh.seek(offset)
            chunk = fh.read(256 * 1024)
        return chunk.decode("utf-8", "replace"), offset + len(chunk)

    def cancel(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                               capture_output=True)
            else:
                os.killpg(os.getpgid(self.process.pid), 15)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass


class JobManager:
    def __init__(self, root: Path):
        self.root = root
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.base = root / "runtime" / "studio_v2" / "jobs"
        self.base.mkdir(parents=True, exist_ok=True)
        self._discover()

    def _discover(self) -> None:
        for directory in sorted(self.base.iterdir(), reverse=True):
            if not directory.is_dir():
                continue
            status = directory / "status.json"
            request = directory / "request.json"
            if not status.is_file() and not (directory / "job.log").is_file():
                continue
            kind = "train"
            if request.is_file():
                try:
                    kind = str(json.loads(request.read_text(encoding="utf-8")).get("kind", "train"))
                except ValueError:
                    pass
            self.jobs[directory.name] = Job(directory.name, directory, kind)

    def create(self, kind: str, config: dict[str, Any]) -> Job:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(config.get("name") or kind))[:30]
        job_id = f"{stamp}_{slug}_{secrets.token_hex(3)}"
        directory = self.base / job_id
        directory.mkdir(parents=True, exist_ok=False)
        config = dict(config)
        config["kind"] = "train" if kind == "train" else "calibrate"
        config["out_dir"] = str(directory)
        (directory / "request.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        env = os.environ.copy()
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        log = (directory / "job.log").open("wb")
        flags = 0
        if os.name == "nt":
            flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        proc = subprocess.Popen(
            [sys.executable, "-B", "-X", "utf8", "-m", "mi_studio_v2", "run",
             "--config", str(directory / "request.json")],
            cwd=str(self.root), stdout=log, stderr=subprocess.STDOUT,
            env=env, creationflags=flags,
        )
        job = Job(job_id, directory, config["kind"], process=proc, config=config)
        with self.lock:
            self.jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(job_id)

    def listing(self) -> list[dict[str, Any]]:
        with self.lock:
            jobs = list(self.jobs.values())
        rows = []
        for job in jobs:
            try:
                rows.append(job.status())
            except Exception:
                rows.append({"job_id": job.job_id, "state": "error"})
        rows.sort(key=lambda row: row.get("created_at", ""), reverse=True)
        return rows


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, *, root: Path, token: str = "",
                 roots: list[Path] | None = None):
        super().__init__(address, handler)
        self.root = root
        self.token = token
        self.roots = [p.resolve() for p in (roots or [root]) if p.exists()]
        self.jobs = JobManager(root)


class Handler(BaseHTTPRequestHandler):
    server: StudioServer
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------------ output
    def log_message(self, fmt: str, *args) -> None:
        if os.environ.get("MI_STUDIO_HTTP_LOG"):
            super().log_message(fmt, *args)

    def _send(self, status: int, body: bytes, content_type: str,
              extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False, default=_json_default).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"ok": False, "error": message})

    # ------------------------------------------------------------------ auth / paths
    def _authorised(self, query: dict[str, str]) -> bool:
        token = self.server.token
        if not token:
            return True
        supplied = (self.headers.get("X-Studio-Token") or query.get("token")
                    or self.headers.get("Authorization", "").removeprefix("Bearer ").strip())
        return secrets.compare_digest(str(supplied), str(token))

    def _safe_path(self, raw: str, *, must_exist: bool = True) -> Path:
        if not raw:
            raise ValueError("缺少路径")
        p = Path(raw)
        if not p.is_absolute():
            p = self.server.root / p
        p = p.resolve()
        allowed = any(p == base or base in p.parents for base in self.server.roots)
        if not allowed:
            raise PermissionError(f"路径不在允许的根目录内：{p}")
        if must_exist and not p.exists():
            raise FileNotFoundError(f"路径不存在：{p}")
        return p

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ValueError("请求体过大")
        raw = self.rfile.read(length) if length else b"{}"
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return data

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True
        host = self.headers.get("Host", "")
        try:
            parsed = urllib.parse.urlparse(origin)
        except ValueError:
            return False
        return parsed.netloc == host or parsed.hostname in ("127.0.0.1", "localhost")

    def _guard(self) -> bool:
        if not self._origin_ok():
            self._error(403, "Origin 不被允许")
            return False
        return True

    # ------------------------------------------------------------------ routes
    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = dict(urllib.parse.parse_qsl(parsed.query))
        if not self._authorised(query):
            self._error(401, "需要访问令牌")
            return
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if path == "/api/health":
                return self._health()
            if path == "/api/browse":
                return self._browse(query.get("path", ""))
            if path == "/api/jobs":
                return self._json(200, {"ok": True, "jobs": self.server.jobs.listing()})
            if path.startswith("/api/jobs/"):
                return self._job(path.split("/")[3], query)
            if path == "/api/model":
                return self._model_info(query.get("path", ""))
            if path == "/api/models":
                return self._models()
            if path.startswith("/files/"):
                return self._download(path)
            self._error(404, "未知路径")
        except Exception as exc:
            self._error(400, f"{type(exc).__name__}: {exc}")

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        query = dict(urllib.parse.parse_qsl(parsed.query))
        if not self._authorised(query) or not self._guard():
            if not self._authorised(query):
                self._error(401, "需要访问令牌")
            return
        try:
            if parsed.path == "/api/scan":
                data = self._body()
                path = self._safe_path(str(data.get("path", "")))
                rows = scan(path)
                return self._json(200, {"ok": True, "path": str(path), "sessions": rows})
            if parsed.path == "/api/jobs":
                data = self._body()
                kind = str(data.get("kind") or "train")
                if kind not in ("train", "calibrate"):
                    raise ValueError("kind 只能是 train 或 calibrate")
                config = data.get("config")
                if not isinstance(config, dict):
                    raise ValueError("缺少 config")
                self._validate_config(kind, config)
                job = self.server.jobs.create(kind, config)
                return self._json(200, {"ok": True, "job": job.status()})
            if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/cancel"):
                job_id = parsed.path.split("/")[3]
                job = self.server.jobs.get(job_id)
                if job is None:
                    return self._error(404, "任务不存在")
                job.cancel()
                return self._json(200, {"ok": True, "job": job.status()})
            self._error(404, "未知路径")
        except Exception as exc:
            self._error(400, f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ handlers
    def _static(self, rel: str) -> None:
        target = (STATIC / rel).resolve()
        if STATIC not in target.parents and target != STATIC:
            return self._error(403, "非法静态路径")
        if not target.is_file():
            return self._error(404, "文件不存在")
        mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if mime.startswith("text/") or mime in ("application/javascript",):
            mime += "; charset=utf-8"
        self._send(200, target.read_bytes(), mime)

    def _health(self) -> None:
        self._json(200, {
            "ok": True,
            "version": C.STUDIO_VERSION,
            "root": str(self.server.root),
            "roots": [str(p) for p in self.server.roots],
            "profiles": C.PROFILES,
            "default_profile": C.DEFAULT_PROFILE,
            "default_outputs": C.default_outputs(),
            "default_source_model": os.environ.get('MI_STUDIO_SOURCE_MODEL',''),
            "output_count": C.OUTPUT_COUNT,
            "auth": bool(self.server.token),
            "jobs": self.server.jobs.listing()[:20],
        })

    def _browse(self, raw: str) -> None:
        base = self._safe_path(raw) if raw else self.server.root
        if base.is_file():
            base = base.parent
        entries = []
        try:
            children = sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            children = []
        for child in children[:400]:
            try:
                is_dir = child.is_dir()
            except OSError:
                continue
            row = {"name": child.name, "path": str(child), "dir": is_dir}
            if is_dir and not child.name.startswith("."):
                try:
                    row["kind"] = detect_kind(child)
                except Exception:
                    row["kind"] = None
            entries.append(row)
        rows = [{"name": p.name, "path": str(p)} for p in self.server.roots]
        self._json(200, {"ok": True, "path": str(base), "parent": str(base.parent)
                         if base != base.parent and any(base.parent == r or r in base.parent.parents
                                                      for r in self.server.roots) else None,
                         "roots": rows, "entries": entries})

    def _job(self, job_id: str, query: dict[str, str]) -> None:
        job = self.server.jobs.get(job_id)
        if job is None:
            return self._error(404, "任务不存在")
        offset = int(query.get("log_from") or 0)
        chunk, next_offset = job.log_chunk(offset)
        status = job.status()
        result = None
        result_path = job.directory / "result.json"
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except ValueError:
                result = None
        self._json(200, {"ok": True, "status": status, "log": chunk,
                         "log_offset": next_offset, "result": result})

    def _models(self) -> None:
        rows = []
        for job in self.server.jobs.listing():
            directory = Path(job.get("directory", ""))
            for pattern in ("*.pt", "onnx/*.onnx"):
                for path in sorted(directory.glob(pattern)):
                    rows.append({"job_id": job.get("job_id"), "name": path.name,
                                 "path": str(path), "kind": job.get("kind"),
                                 "state": job.get("state")})
        self._json(200, {"ok": True, "models": rows[:200]})

    def _model_info(self, raw: str) -> None:
        path = self._safe_path(raw)
        if path.suffix.lower() != ".pt":
            raise ValueError("只支持 .pt 模型（ONNX 不能用于校准）")
        # Never import torch in the web process: the model-inspection CLI is a
        # short-lived child, so a model load cannot pin GPU/CPU memory in the server.
        env = os.environ.copy()
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.run(
            [sys.executable, "-B", "-X", "utf8", "-m", "mi_studio_v2", "inspect",
             "--model", str(path)],
            cwd=str(self.server.root), capture_output=True, text=True,
            env=env, timeout=180,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
            raise ValueError("模型载入失败：" + " | ".join(tail))
        data = json.loads(proc.stdout)
        data["ok"] = True
        self._json(200, data)

    def _download(self, path: str) -> None:
        parts = path.split("/")
        if len(parts) < 3:
            return self._error(404, "文件不存在")
        job_id = parts[2]
        job = self.server.jobs.get(job_id)
        if job is None:
            return self._error(404, "任务不存在")
        rel = urllib.parse.unquote("/".join(parts[3:]))
        target = (job.directory / rel).resolve()
        if job.directory not in target.parents and target != job.directory:
            return self._error(403, "非法文件路径")
        if not target.is_file():
            return self._error(404, "文件不存在")
        mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(target.stat().st_size))
        self.send_header("Content-Disposition",
                         f'attachment; filename="{urllib.parse.quote(target.name)}"')
        self.end_headers()
        with target.open("rb") as fh:
            shutil.copyfileobj(fh, self.wfile)

    def _validate_config(self, kind: str, config: dict[str, Any]) -> None:
        sources = config.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError("至少需要一个数据源")
        for item in sources:
            if not isinstance(item, dict) or not item.get("path"):
                raise ValueError("数据源缺少 path")
            self._safe_path(str(item["path"]))
            mapping = item.get("mapping") or {}
            rows = mapping.get("rows")
            if not isinstance(rows, list) or not rows:
                raise ValueError("每个数据源都要有 mapping.rows")
            if any(not bool(row.get("reviewed")) for row in rows):
                raise ValueError("存在未复核的标签映射行；请逐行确认（含「不使用」）")
        if kind == "train":
            C.validate_outputs(config.get("outputs"))
        else:
            if not config.get("source_model"):
                raise ValueError("校准缺少 source_model")
            self._safe_path(str(config["source_model"]))


def serve(host: str = "127.0.0.1", port: int = 57500, root: Path = ROOT,
          token: str = "", open_browser: bool = False) -> int:
    server = StudioServer((host, port), Handler, root=root, token=token)
    url = f"http://{host}:{port}/"
    print(f"[studio] serving {url}  root={root}  token={'yes' if token else 'no'}")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[studio] stopped")
    finally:
        server.server_close()
    return 0
