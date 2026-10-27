"""Explicit, single-shot hand commissioning independent of EEG and models."""
from __future__ import annotations

import copy
from datetime import datetime
import json
from pathlib import Path
import threading
import uuid

from .hand_output import HandOutput, connection_options, integer, motion_options


class ManualHandTest:
    def __init__(self, log_path, browser_age):
        self.log_path = Path(log_path)
        self.browser_age = browser_age
        self.lock = threading.RLock()
        self.worker = None
        self.cancel = threading.Event()
        self.state = dict(busy=False, phase="idle", connected=False, angles=[], error="",
                          targets=[], before=[], saved_path="", request_id="")

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state)

    def start(self, request, probe):
        options = connection_options(request)
        if not probe or request.get("probe_id") != probe["probe_id"] or options != probe["options"]:
            raise ValueError("请先对当前串口、波特率和ID点击仅读状态，再发送测试指令。")
        kind = request.get("kind")
        if kind not in ("index_bend", "index_open", "custom"):
            raise ValueError("未知机械手测试指令。")
        targets = request.get("targets")
        if kind == "custom":
            if not isinstance(targets, list) or len(targets) != 6:
                raise ValueError("请填写六路目标位置。")
            for value in targets:
                integer(value, "目标位置", -1, 1000)
            if all(value == -1 for value in targets):
                raise ValueError("至少指定一个通道，-1表示跳过。")
        recovery = request.get("recovery", "start")
        if recovery not in ("hold", "start"):
            raise ValueError("独立测试须选择单目标校准或一次往返。")
        config = dict(mode="inspire", recovery=recovery, **options, **motion_options(request))
        with self.lock:
            if self.state["busy"] or (self.worker and self.worker.is_alive()):
                raise ValueError("机械手测试正在执行，不接收排队指令。")
            self.cancel = threading.Event()
            self.state = dict(busy=True, phase="connecting", connected=False, angles=[], error="",
                              targets=[], before=[], saved_path=str(self.log_path),
                              request_id=uuid.uuid4().hex, source="manual_no_eeg")
            self.worker = threading.Thread(target=self._run,
                args=(config, kind, copy.deepcopy(targets)), daemon=True, name="manual-hand-test")
            self.worker.start()
            return self.snapshot()

    def stop(self):
        self.cancel.set()

    def _event(self, state, event):
        with self.lock:
            self.state.update({k: state[k] for k in ("phase", "connected", "angles", "error")})
            self.state["read_retry_count"] = state.get("read_retry_count", 0)
            self.state["write_ack_recovered_count"] = state.get("write_ack_recovered_count", 0)
            if event:
                self.state["result"] = copy.deepcopy(event)

    def _log(self, event):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(timestamp=datetime.now().astimezone().isoformat(),
                      request_id=self.state["request_id"], source="manual_no_eeg", **event)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def _run(self, config, kind, targets):
        output = None
        failure = None
        try:
            self._log(dict(event="requested", kind=kind, connection=config, targets=targets))
            if self.cancel.is_set():
                raise RuntimeError("测试已取消，未发送动作。")
            output = HandOutput(config, self._event)
            output.cancel = self.cancel
            output.connect()
            before = output.snapshot()["angles"]
            if kind != "custom":
                targets = [-1] * 6
                targets[3] = max(0, min(1000, before[3] + (-100 if kind == "index_bend" else 100)))
                if targets[3] == before[3]:
                    raise ValueError("食指已经处于该方向的刻度边界，未发送动作。")
            with self.lock:
                self.state.update(before=before, targets=targets)
            output.config["profiles"] = [targets]
            self._log(dict(event="dispatch", before=before, targets=targets, physical_action=True))
            if self.cancel.is_set() or not output.submit(dict(class_id=0, sequence=1)):
                raise RuntimeError("测试已取消，未发送动作。")
            while output.thread.is_alive():
                output.thread.join(.05)
                if self.browser_age() > 2:
                    self.cancel.set()
                    with self.lock:
                        self.state["stop_reason"] = "页面超过2秒未更新，已取消动作等待。"
        except Exception as exc:
            failure = ("cancelled" if self.cancel.is_set() else "failed", str(exc))
        finally:
            if output is not None:
                try:
                    output.close()
                except Exception as exc:
                    failure = ("failed", str(exc))
            with self.lock:
                if failure:
                    self.state.update(phase=failure[0], error=failure[1])
                self.state.update(connected=False)
                try:
                    self._log(dict(event="finished", result={**self.snapshot(), "busy": False}))
                except Exception as exc:
                    self.state.update(phase="failed", error=f"测试记录保存失败：{exc}")
                self.state["busy"] = False
