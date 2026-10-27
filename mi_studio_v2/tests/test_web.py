# -*- coding: utf-8 -*-
import json
import threading
import urllib.error
import urllib.request

import pytest

from mi_studio_v2 import contracts as C
from mi_studio_v2.web.server import Handler, StudioServer


def _serve(tmp_path, token=""):
    server = StudioServer(("127.0.0.1", 0), Handler, root=tmp_path, token=token,
                          roots=[tmp_path])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _request(url, method="GET", body=None, headers=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method,
                                 headers=headers or {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as res:
            return res.status, json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_health_and_path_confinement(tmp_path):
    server, base = _serve(tmp_path)
    try:
        status, data = _request(base + "/api/health")
        assert status == 200 and data["ok"] and data["output_count"] == 6
        status, data = _request(base + "/api/scan", "POST", {"path": "F:/Windows"})
        assert status == 400 and "允许" in data["error"]
    finally:
        server.shutdown()


def test_unreviewed_mapping_is_rejected_before_a_job_starts(tmp_path):
    session = tmp_path / "sess"
    session.mkdir()
    server, base = _serve(tmp_path)
    try:
        config = {
            "kind": "train", "profile": "platform_replay_v1",
            "sources": [{"path": str(session), "mapping": {"rows": [
                {"id": 0, "target": 0, "reviewed": False}]}}],
            "outputs": C.default_outputs(),
        }
        status, data = _request(base + "/api/jobs", "POST",
                                {"kind": "train", "config": config})
        assert status == 400 and "未复核" in data["error"]
    finally:
        server.shutdown()


def test_token_guard(tmp_path):
    server, base = _serve(tmp_path, token="secret")
    try:
        status, _ = _request(base + "/api/health")
        assert status == 401
        status, data = _request(base + "/api/health", headers={
            "Content-Type": "application/json", "X-Studio-Token": "secret"})
        assert status == 200 and data["ok"]
    finally:
        server.shutdown()
