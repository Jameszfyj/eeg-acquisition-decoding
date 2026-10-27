"""不接收目标标签的机械手决策层；设备通信由独立驱动实现。"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .config import HAND_ACTIONS


@dataclass
class HandCommand:
    trial_id: int
    action: str
    class_id: int
    confidence: float
    decision_time: float
    attempt: int


class HandDriver:
    def execute(self, command: HandCommand) -> dict:
        """返回接收确认；通信确认并不等于裁判判定动作正确。"""
        raise NotImplementedError("须依据机械手厂家协议实现。")


class MockHandDriver(HandDriver):
    def __init__(self, log_path: str | Path | None = None):
        self.commands = []
        self.log_path = None if log_path is None else Path(log_path)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.touch(exist_ok=False)

    def execute(self, command: HandCommand):
        if command.action not in HAND_ACTIONS:
            raise ValueError("未知机械手动作。")
        event = {**vars(command), "driver": "mock", "physical_action": False, "transport_ack": True}
        self.commands.append(event)
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event


class DecisionGate:
    def __init__(self, classes: list[dict], confidence=0.7, margin=0.15,
                 stable_seconds=0.3, timeout_seconds=10.0, max_attempts=2, max_gap_seconds=0.5):
        self.classes = classes
        self.threshold = float(confidence)
        self.margin = float(margin)
        self.stable_seconds = float(stable_seconds)
        self.timeout = float(timeout_seconds)
        self.max_attempts = int(max_attempts)
        self.max_gap = float(max_gap_seconds)
        if not 0 <= self.threshold <= 1 or not 0 <= self.margin <= 1 or self.stable_seconds < 0:
            raise ValueError("决策阈值无效。")
        if self.timeout <= 0 or self.max_gap <= 0 or not 1 <= self.max_attempts <= 2:
            raise ValueError("超时和最大观察间隔须为正；尝试次数为1或2。")
        self.active = False
        self.candidate = None

    def begin_trial(self, trial_id: int, now: float):
        self.trial_id, self.start = int(trial_id), float(now)
        self.attempts = 0
        self.active = True
        self.last_time = float(now)
        self.clear_candidate()

    def clear_candidate(self):
        self.candidate, self.candidate_since = None, None

    def retry(self, now: float):
        if self.active or self.attempts == 0 or now < self.start or now - self.start > self.timeout or self.attempts >= self.max_attempts:
            return False
        self.active = True
        self.last_time = now
        self.clear_candidate()
        return True

    def end_trial(self):
        self.active = False
        self.clear_candidate()

    def update(self, probabilities, now: float) -> HandCommand | None:
        p = np.asarray(probabilities, dtype=float)
        if not self.active:
            return None
        if now < self.last_time:
            raise ValueError("决策时间不能倒退。")
        if now - self.last_time > self.max_gap:
            self.clear_candidate()
        self.last_time = now
        if now - self.start > self.timeout:
            self.end_trial()
            return None
        if p.shape != (len(self.classes),) or not np.all(np.isfinite(p)) or np.any(p < 0) or not np.isclose(p.sum(), 1, atol=1e-4):
            self.clear_candidate()
            return None
        top = int(p.argmax())
        second = float(np.partition(p, -2)[-2])
        if p[top] < self.threshold or p[top] - second < self.margin:
            self.clear_candidate()
            return None
        if top != self.candidate:
            self.candidate, self.candidate_since = top, now
        if now - self.candidate_since + 1e-9 < self.stable_seconds:
            return None
        self.attempts += 1
        command = HandCommand(self.trial_id, self.classes[top]["action"], top, float(p[top]), now, self.attempts)
        # 不根据设备ACK自动宣告正确，也不自动消耗第二次尝试；重试须明确调用retry。
        self.end_trial()
        return command
