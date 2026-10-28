"""Continuous, cue-free decoding with optional, explicitly selected hand output.

The demo is an explicit probability demonstration, not EEG classification.
Registered synthetic checkpoints run through the existing TCP/EEG decoder.
ACP uses the acquisition studio's adapter and one leased device connection.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
import copy
import errno
import json
import math
from pathlib import Path
import queue
import threading
import time
import uuid
import urllib.request
import webbrowser

import numpy as np

from .config import ACTION_NAMES, validate_participant_id, write_json
from .online_web import OnlineController, create_online_server
from .subject_online import DEFAULT_REGISTRY, prepare_subject_model
from .hand_output import (HandOutput, idle_state, validate_output, connection_options, load_hand_class,
                          ACTION_CHANNELS, ACTION_LABELS, CHANNELS, calibration_defaults, calibration_options,
                          motion_feedback)
from .hand_test import ManualHandTest
from .action_records import build_class_evidence, instruction_evidence, actual_evidence
from .speech import SpeechCache
from .waveform_preview import WaveformPreview
from .workbench_settings import local_settings, local_path, model_preferences

ROOT = Path(__file__).resolve().parents[1]
DEMO_ID = "probability-demo-v1"
# Temporary software-debug policy. Offline training/calibration tools keep their
# own contracts; the online workbench must never collect or fit calibration.
ONLINE_CALIBRATION_ENABLED = False
CALIBRATION_DISABLED_MESSAGE = '软件调试期间，在线脑电校准暂时禁用；所有模型可直接测试，已有模型参数保持不变。'
DEMO_CLASSES = [{"id": i, "name": f"任务{i+1:02d}", "task_text": f"任务{i+1:02d}（人工概率）",
                 "action": f"slot_{i+1}", "hand_action": key, "action_name": ACTION_LABELS[key]}
                for i, key in enumerate(ACTION_CHANNELS)]


def number(value, name, low, high):
    if isinstance(value, bool):
        raise ValueError(f"{name}须为数字。")
    v = float(value)
    if not math.isfinite(v) or not low <= v <= high:
        raise ValueError(f"{name}须在{low}至{high}之间。")
    return v


class ContinuousGate:
    """One event per stable activation; neutral release or a new class re-arms.

    Confidence rejection is not a trained physiological idle detector.
    Large inference gaps invalidate the stability interval.
    """
    def __init__(self, n_classes, confidence=.70, margin=.15, stable_seconds=.8,
                 cooldown_seconds=2., release_seconds=.5, max_gap_seconds=.5):
        self.n_classes = n_classes
        self.confidence = number(confidence, "概率阈值", .5, .99)
        self.margin = number(margin, "类别差距", 0, .8)
        self.stable = number(stable_seconds, "稳定时间", .2, 20)
        self.cooldown = number(cooldown_seconds, "动作间隔", .5, 20)
        self.release = number(release_seconds, "释放时间", .2, 20)
        self.max_gap = number(max_gap_seconds, "最大预测间隔", .2, 5)
        self.last_time = None
        self.last_command = -float("inf")
        self.latched = None
        self.neutral_since = None
        self.candidate, self.since = None, None
        self.state = {"candidate": None, "progress": 0., "reason": "等待预测", "latched": None}

    def reset_candidate(self):
        self.candidate = self.since = None

    def settings(self):
        return dict(mode='stable', confidence=self.confidence, margin=self.margin,
                    stable_seconds=self.stable, cooldown_seconds=self.cooldown,
                    release_seconds=self.release, max_gap_seconds=self.max_gap)

    def update(self, probabilities, now, *, inhibited=False):
        if self.last_time is not None and now < self.last_time:
            raise ValueError("判决时钟不能倒退。")
        if self.last_time is not None and now-self.last_time > self.max_gap:
            self.reset_candidate()
            self.neutral_since = None
        self.last_time = now
        p = np.asarray(probabilities, dtype=float)
        valid = (p.shape == (self.n_classes,) and np.all(np.isfinite(p)) and
                 np.all(p >= 0) and np.isclose(p.sum(), 1, atol=1e-4))
        if not valid:
            self.reset_candidate()
            self.neutral_since = None
            self.state = {"candidate": None, "progress": 0., "reason": "预测无效", "latched": self.latched}
            return None
        top = int(p.argmax())
        second = float(np.partition(p, -2)[-2])
        eligible = p[top] >= self.confidence and p[top]-second >= self.margin
        if not eligible:
            self.reset_candidate()
            if self.neutral_since is None:
                self.neutral_since = now
            if now-self.neutral_since >= self.release:
                self.latched = None
            reason, progress = "证据不足 · 暂不输出", 0.
        else:
            self.neutral_since = None
            if self.latched is not None and top != self.latched:
                self.latched = None
            if inhibited:
                self.reset_candidate()
                reason, progress = "机械手执行中 · 等待到位", 0.
            elif top == self.latched:
                self.reset_candidate()
                reason, progress = "本次激活已输出 · 等待释放或换类", 0.
            elif now-self.last_command+1e-9 < self.cooldown:
                # A different eligible class releases the previous activation even
                # during cooldown. Returning to it must accumulate again afterwards.
                self.reset_candidate()
                reason, progress = "动作间隔中", 0.
            else:
                if self.candidate != top:
                    self.candidate, self.since = top, now
                progress = min(1., (now-self.since)/self.stable)
                reason = "正在积累稳定证据"
                if progress >= 1.-1e-9:
                    self.latched, self.last_command = top, now
                    self.state = {"candidate": top, "progress": 1., "reason": "已触发模拟动作", "latched": top}
                    self.reset_candidate()
                    return {"class_id": top, "confidence": float(p[top]), "time_seconds": now}
        if inhibited:
            reason = "机械手执行中 · 等待到位"
        self.state = {"candidate": self.candidate, "progress": progress, "reason": reason, "latched": self.latched}
        return None


class VotingGate(ContinuousGate):
    """Fresh rolling votes; every completed action can be followed by the same class.

    Early decisions use rolling votes; timeout uses all valid slices in the round.
    Packet loss never erases valid evidence or moves the round's deadline.
    """
    def __init__(self, n_classes, vote_window_seconds=2., vote_ratio=.65,
                 decision_timeout_seconds=3.5, cooldown_seconds=4., max_gap_seconds=.5):
        super().__init__(n_classes, cooldown_seconds=cooldown_seconds,
                         max_gap_seconds=max_gap_seconds)
        self.window = number(vote_window_seconds, '投票时间', .5, 20)
        self.ratio = number(vote_ratio, '得票比例', .5, 1)
        self.timeout = number(decision_timeout_seconds, '最长判决等待', self.window, 20)
        self.votes = deque()
        self.round_votes = []
        self.round_started = None
        self.state = dict(mode='vote', candidate=None, progress=0., reason='等待预测', latched=None)

    def settings(self):
        return dict(mode='vote', vote_window_seconds=self.window, vote_ratio=self.ratio,
                    decision_timeout_seconds=self.timeout, cooldown_seconds=self.cooldown,
                    max_gap_seconds=self.max_gap)

    def reset_candidate(self):
        super().reset_candidate()
        self.votes.clear()
        self.round_votes.clear()
        self.round_started = None

    def _waiting(self, reason):
        self.state = dict(mode='vote', candidate=None, progress=0., reason=reason,
                          latched=None, votes=[0]*self.n_classes, vote_total=0)

    def update(self, probabilities, now, *, inhibited=False):
        if not math.isfinite(now) or (self.last_time is not None and now < self.last_time):
            raise ValueError('判决时钟无效或倒退。')
        previous = self.last_time
        self.last_time = now
        p = np.asarray(probabilities, dtype=float)
        valid = (p.shape == (self.n_classes,) and np.all(np.isfinite(p)) and
                 np.all(p >= 0) and np.isclose(p.sum(), 1, atol=1e-4))
        if not valid or inhibited or now-self.last_command+1e-9 < self.cooldown:
            self.reset_candidate()
            self._waiting('预测无效' if not valid else '机械手执行中 · 等待到位'
                          if inhibited else '动作间隔中')
            return None
        if previous == now:
            return None  # Re-reading the same prediction must not add another vote.
        if self.round_started is None:
            self.round_started = now
        top = int(p.argmax())
        second = float(np.partition(p, -2)[-2])
        # Exact top ties abstain instead of systematically voting for class zero.
        winner = top if p[top]-second > 1e-6 else None
        self.votes.append((now, winner, p.copy()))
        self.round_votes.append((now, winner, p.copy()))
        return self.tick(now, allow_majority=True)

    def start_round(self, now):
        """A discarded slice can start an empty round without supplying a vote."""
        if self.round_started is None and now-self.last_command+1e-9 >= self.cooldown:
            self.round_started = now

    def tick(self, now, *, allow_majority=False):
        if self.round_started is None:
            return None
        wait = now-self.round_started
        if not math.isfinite(now) or wait < 0:
            raise ValueError('判决时钟无效或倒退。')
        deadline = wait+1e-9 >= self.timeout
        while self.votes and self.votes[0][0] < now-self.window-1e-9:
            self.votes.popleft()
        rows = self.round_votes if deadline else self.votes
        counts = np.zeros(self.n_classes, dtype=int)
        latest = np.full(self.n_classes, -float('inf'))
        for stamp, vote, _ in rows:
            if vote is not None:
                counts[vote] += 1
                latest[vote] = stamp
        means = np.mean([row[2] for row in rows], axis=0) if rows else np.zeros(self.n_classes)
        candidates = np.flatnonzero(counts)
        candidate = (max(map(int, candidates), key=lambda i: (counts[i], means[i], latest[i]))
                     if len(candidates) else None)
        total = len(rows)
        share = float(counts[candidate]/total) if candidate is not None else 0.
        ready = wait+1e-9 >= self.window and int(counts.sum()) >= 3
        reason = ('预测未分出类别 · 等待新证据' if candidate is None else
                  f'投票 {counts[candidate]}/{total} · 正在收集预测' if not ready else
                  f'投票 {counts[candidate]}/{total} · 等待多数')
        self.candidate = candidate
        self.state = dict(mode='vote', candidate=candidate, progress=share, reason=reason,
                          latched=None, votes=counts.tolist(), vote_total=total,
                          vote_share=share, decision_wait_seconds=wait,
                          decision_remaining_seconds=max(0., self.timeout-wait),
                          round_valid_slices=len(self.round_votes))
        if deadline and candidate is None:
            self.state.update(reason='倒计时结束 · 无有效类别票，本轮不输出', decision_reason='deadline_no_votes')
            self.reset_candidate()
            return None
        if candidate is not None and (deadline or (allow_majority and ready and share+1e-9 >= self.ratio)):
            decision_reason = 'deadline_plurality' if deadline else 'majority'
            event = dict(class_id=candidate, confidence=float(means[candidate]), time_seconds=now,
                         decision_mode='vote', decision_reason=decision_reason,
                         vote_counts=counts.tolist(), vote_total=total, vote_share=share,
                         decision_wait_seconds=wait, vote_window_seconds=self.window,
                         vote_scope='all_valid_slices_in_round' if deadline else 'rolling_window',
                         mean_probabilities=means.tolist())
            self.last_command = now
            self.state.update(reason='多数票已触发' if decision_reason == 'majority'
                              else '最长等待已到 · 最高票触发', decision_reason=decision_reason)
            self.reset_candidate()
            return event
        return None


class ContinuousController(OnlineController):
    service_name = 'handbci.continuous.v1'

    def __init__(self, output_root=ROOT / "runtime/continuous", **kwargs):
        super().__init__(output_root, **kwargs)
        from .device_timeline import PIPELINE_REVISION
        self.state['pipeline_revision'] = PIPELINE_REVISION
        self.live_device_clock = None
        self.waveform_preview = WaveformPreview()
        self.target = None
        self.auto_demo = True
        self.output_config = {"mode": "simulation"}
        self.hand_output = None
        self.hand_log = None
        self.motion_completed_at = None
        self.fresh_collection_after = None
        self.fresh_source_cutoff = None
        self.simulation_motion = None
        self.hand_probe = None
        self.calibration_decoder = None
        self.session_purpose = 'legacy'
        self.prompted = None
        self.prompted_settings = {'enabled': False}
        self.prompted_origin_ms = None
        self.speech = SpeechCache(self.output_root/'audio_cache')
        self.eeg_probe_busy = False
        self.eeg_fault_active = threading.Event()
        self.eeg_discovery_busy = False
        self.eeg_probe_cancel = threading.Event()
        self.eeg_probe_done = threading.Event()
        self.eeg_probe_done.set()
        self.disconnect_pending = False
        from .eeg_connection import discovery_hosts
        self.eeg_hosts_path = self.output_root/'eeg_discovery_hosts.json'
        try:
            self.eeg_known_hosts = discovery_hosts(json.loads(self.eeg_hosts_path.read_text(encoding='utf-8')))
        except (OSError, ValueError, TypeError):
            self.eeg_known_hosts = []
        self.manual_hand = ManualHandTest(self.output_root/"manual_hand_tests.jsonl",
                                         lambda: time.monotonic()-self.last_browser_seen)
        self.state.update(classes=copy.deepcopy(DEMO_CLASSES), mode="demo", gate={}, wall_seconds=0.,
                          source_seconds=0., rate_hz=None, expected_rate=None, gap_count=0,
                          stale_seconds=0., saved_path="", physical_action=False, output_mode="simulation", hand=idle_state())
        self.state['stream_quality'] = dict(loss_events=0, corrupt_events=0, missing_samples=0, recovering=False)

    def hand_ports(self):
        return load_hand_class().ports()

    def snapshot(self, *, browser_seen=False):
        with self.condition:
            self._finish_disconnect()
            result = super().snapshot(browser_seen=browser_seen)
            result["manual_hand"] = self.manual_hand.snapshot()
            result['waveform'] = self.waveform_preview.snapshot()
            return result

    def _finish_disconnect(self):
        """Called under the controller lock; never claim closure before cleanup."""
        if not self.disconnect_pending:
            return
        if (self.eeg_probe_busy or not self.eeg_probe_done.is_set() or self.eeg_discovery_busy
                or (self.worker is not None and self.worker.is_alive())
                or (self.manual_hand.worker is not None and self.manual_hand.worker.is_alive())):
            return
        self.disconnect_pending = False
        self.hand_probe = None
        message = ('设备已断开；会话存在异常，请查看错误及已保留记录。' if self.state.get('error') else
                   '设备已断开；当前记录已保存，可重新开始。' if self.state.get('saved_path') else
                   '设备已断开，可重新开始。')
        self.state.update(running=False, probabilities=[],
            device_connection=dict(phase='disconnected', message=message),
            gate=dict(candidate=None, progress=0., reason='设备已断开'))

    def _require_connection_ready(self):
        self._finish_disconnect()
        if self.disconnect_pending:
            raise ValueError('正在保存记录并断开设备，请等待显示“设备已断开”后再开始。')
        self.state['device_connection'] = dict(phase='idle', message='连接在开始采集或检验时建立。')

    def disconnect_devices(self):
        with self.condition:
            if not self.disconnect_pending:
                self.disconnect_pending = True
                self.state['device_connection'] = dict(phase='disconnecting',
                    message='正在停止采集与连接检验、保存记录并释放设备…')
            self.stop()
            self._finish_disconnect()
            return dict(ok=True, device_connection=copy.deepcopy(self.state['device_connection']))

    def browser_snapshot(self, subject_window_id=None):
        # A monitor window must not keep a missing task display alive.
        with self.condition:
            owner = self.state.get('subject_window_id')
            seen = owner == subject_window_id if owner else subject_window_id is None
            return self.snapshot(browser_seen=seen)

    def stop_subject(self, request):
        with self.condition:
            if (self.state.get('subject_window_id')
                    and request.get('subject_window_id') == self.state['subject_window_id']):
                self.stop()
            return {'ok': True}

    def stop_monitor(self, request):
        with self.condition:
            if (self.state.get('monitor_window_id')
                    and request.get('monitor_window_id') == self.state['monitor_window_id']):
                self.stop()
            return {'ok': True}

    def start_hand_test(self, request):
        with self.condition:
            self._require_connection_ready()
            if self.eeg_probe_busy or self.eeg_discovery_busy:
                raise ValueError('设备搜索或连接检验正在进行，请等待完成。')
            if self.worker is not None and self.worker.is_alive():
                raise ValueError("请先停止解码，再使用独立机械手测试。")
            self.last_browser_seen = time.monotonic()
            return self.manual_hand.start(request, self.hand_probe)

    def stop_hand_test(self):
        self.manual_hand.stop()
        return self.manual_hand.snapshot()

    def check_hand_connection(self, request):
        # This endpoint only opens, reads telemetry and closes. It never calls move/set_*.
        options = connection_options(request)
        with self.condition:
            self._require_connection_ready()
            if self.eeg_probe_busy or self.eeg_discovery_busy:
                raise ValueError('设备搜索或连接检验正在进行，请等待完成。')
            if self.worker is not None and self.worker.is_alive():
                raise ValueError("会话运行中不能重新检查或占用机械手串口。")
            if self.manual_hand.snapshot()["busy"]:
                raise ValueError("机械手测试正在执行，请先停止并等待串口释放。")
            self.hand_probe = None
            hand = load_hand_class()(**options)
            try:
                hand.connect()
                status = hand.get_status()
                self.hand_probe = dict(options=options, probe_id=uuid.uuid4().hex)
                return {"connected": True, "closed_after_check": True,
                        "probe_id": self.hand_probe["probe_id"],
                        "status": status, "message": "状态读取完成，串口已关闭；未下发动作。"}
            finally:
                hand.disconnect()

    def models(self):
        from .four_class_online import catalog
        from .six_class_online import catalog as six_catalog
        items = [{"participant_id": "DEMO", "model_id": DEMO_ID,
                 "name": "六类交互演示 · 无脑电 / 无模型", "n_classes": 6,
                 "source": "probability_demo"}] + super().models() + catalog() + six_catalog()
        preferred = local_settings().get('default_model_id')
        return [dict(item, is_default=item['model_id'] == preferred) for item in items]

    def model_details(self, participant, model_id):
        details = self._model_details(participant, model_id)
        from .six_class_online import SIX_MODEL_IDS
        calibration = calibration_defaults()
        saved_path = local_path('hand_calibration', ROOT/'configs/hand_calibration_P01_20261024.json')
        personal_six=participant=='P01' and (model_id in SIX_MODEL_IDS or details.get('runtime') in ('retrained_six','mi_studio_v2') or details.get('native_platform_ifnet'))
        if personal_six and saved_path.is_file():
            saved = json.loads(saved_path.read_text(encoding='utf-8-sig'))
            if (saved.get('participant_id') == participant
                    and saved.get('all_target_poses_user_confirmed') is True):
                calibration = calibration_options(saved, physical=True)
                calibration['revision'] = saved['revision']
                calibration['shared_key'] = 'P01-inspire-six-actions'
                details['hand_connection'] = connection_options(saved)
        details.update(hand_actions=ACTION_LABELS, hand_channels=CHANNELS,
                       hand_calibration=calibration)
        if personal_six:
            details['hand_mapping_revision'] = 'P01-feet-pinch-tongue-little-v1'
        preferences = model_preferences(model_id)
        mapping = preferences.get('hand_actions')
        if mapping is not None:
            details['classes'] = self.display_classes(details['classes'], hand_actions=mapping)
            details['hand_mapping_revision'] = 'local:' + json.dumps(mapping, separators=(',', ':'))
            details['hand_mapping_reset_all'] = True
        details['ui_defaults'] = preferences.get('ui_defaults', {})
        if not ONLINE_CALIBRATION_ENABLED:
            details.update(online_calibration_disabled=True, calibration_notice=CALIBRATION_DISABLED_MESSAGE,
                           requires_calibration=False, requires_prompted_calibration=False)
        return details

    def _model_details(self, participant, model_id):
        from .six_class_online import SIX_MODEL_IDS, prepare as prepare_six
        if model_id in SIX_MODEL_IDS:
            selection, cfg = prepare_six(participant, model_id)
            details = {**selection, 'classes': self.display_classes(cfg['classes']),
                    'channels': cfg['device']['channels'], 'model_channels': 30,
                    'unit': 'uV', 'window_seconds': 2.}
            if selection.get('requires_calibration'):
                from .calibration_store import CalibrationStore
                history = CalibrationStore(self.output_root).history(selection, cfg)
                details['calibration_history'] = history['available']
                details['unavailable_calibrations'] = history['unavailable']
                details['previous_calibration'] = history['available'][0] if history['available'] else None
            return details
        from .four_class_online import CHECKPOINTS, prepare as prepare_four
        if model_id in CHECKPOINTS:
            selection, cfg = prepare_four(participant, model_id)
            return {**selection, 'classes':cfg['classes'], 'channels':cfg['device']['channels'],
                    'model_channels':30, 'unit':cfg['device']['unit'], 'window_seconds':2.,
                    'independent_test_of_this_refit':False,
                    'input_note':'使用采集工作台的ACP直连，按名称对齐30路模型输入；旧数据的历史参考设置未完整记录，效果须实测。'}
        if model_id == DEMO_ID:
            return {"classes": copy.deepcopy(DEMO_CLASSES), "synthetic_model": True,
                    "source": "probability_demo", "window_seconds": None,
                    "channels": [], "sample_rate": None, "unit": "", "n_classes": 6,
                    "model": "无需模型；概率由演示程序生成"}
        details = super().model_details(participant, model_id)
        details["classes"] = self.display_classes(details["classes"])
        if details.get('saved_calibration_ready'):
            details.update(requires_prompted_calibration=False,
                input_note='909次重训模型已恢复配套20轮EA与脚舌校准参数，无需原始脑电重新拟合。',
                calibration_history=[],unavailable_calibrations=[],previous_calibration=None)
            return details
        if details.get('embedded_calibration_ready'):
            details.update(requires_prompted_calibration=False,
                input_note='ONNX已包含经输出层核对的校准偏置；按准备提示对齐2秒窗口，直接使用内置校准。',
                calibration_history=[],unavailable_calibrations=[],previous_calibration=None)
            if details.get('calibration_method')=='embedded_retrained_six_calibration_v1':
                details['input_note']='ONNX已包含双IFNet、三个2秒窗、脚舌路由及20轮校准参数，直接使用完整模型。'
            return details
        if details.get('runtime')=='mi_studio_v2':
            details.update(input_note='v2按准备提示对齐2秒窗口；专用校准头已载入，实时预测不使用目标标签。',
                calibration_history=[],unavailable_calibrations=[],previous_calibration=None)
            return details
        if details['source'] == 'neuracle_acp':
            from .shared_calibration import SharedCalibrationStore, imported_protocol
            selection, cfg = prepare_subject_model(participant, model_id, registry=self.registry)
            protocol = imported_protocol(cfg)
            history = SharedCalibrationStore(self.output_root).history(cfg, protocol)
            details.update(requires_prompted_calibration=True, calibration_rounds=protocol['rounds'],
                calibration_trials=protocol['trials'],calibration_imagery_seconds=protocol['imagery_seconds'],
                calibration_rest_seconds=protocol['rest_seconds'],calibration_history=history['available'],
                unavailable_calibrations=history['unavailable'],
                previous_calibration=history['available'][0] if history['available'] else None)
            if selection.get('runtime')=='retrained_six':
                details.update(requires_calibration=True,calibration_method=selection['calibration_method'])
        return details

    @staticmethod
    def display_classes(classes, actions=None, hand_actions=None):
        result = copy.deepcopy(classes)
        if actions is not None and (not isinstance(actions, list) or len(actions) != len(result)):
            raise ValueError("动作名称数量须与模型类别数相同。")
        if hand_actions is not None and (not isinstance(hand_actions, list) or len(hand_actions) != len(result)
                                         or any(k not in ACTION_CHANNELS for k in hand_actions)):
            raise ValueError("每个类别须选择有效的机械手动作。")
        ordered = {c['id']: list(ACTION_CHANNELS)[i] for i, c in enumerate(sorted(result, key=lambda c: c['id']))} if 2 <= len(result) <= 6 else {}
        for i, c in enumerate(result):
            key = hand_actions[i] if hand_actions is not None else c.get("hand_action", c.get("action"))
            if hand_actions is None and not c.get('hand_action') and len(result) == 6 and c['name'] in ('双脚', '舌头'):
                key = {'双脚':'pinch', '舌头':'little'}[c['name']]
            if key not in ACTION_CHANNELS and 2 <= len(result) <= 6:
                key = ordered[c['id']]
            if key in ACTION_CHANNELS:
                c['hand_action'] = key
            expected_foot = {'left_hand_right_foot_mi': '右脚', 'right_hand_left_foot_mi': '左脚'}.get(c['name'])
            task = c.get('task_text', '')
            if expected_foot and any(foot in task for foot in ('左脚', '右脚')) and expected_foot not in task:
                c['label_warning'] = '原始类别名称与任务文字的左右脚不一致，真实训练意图无法确定；保留原字段。'
            default = ACTION_LABELS.get(key) or c.get("action_name") or ACTION_NAMES.get(c["action"], c["action"])
            text = str(actions[i] if actions is not None else default).strip()
            if not text or len(text) > 40:
                raise ValueError("每个动作名称须为1至40个字符。")
            c["action_name"] = text
        return result

    def prepare(self, request):
        if not ONLINE_CALIBRATION_ENABLED:
            if request.get('session_purpose') == 'calibration':
                raise ValueError(CALIBRATION_DISABLED_MESSAGE)
            # Reusing historical raw EEG fits new parameters too. Ignore stale
            # page choices; only parameters already bound to the model survive.
            request = {k:v for k,v in request.items() if k not in ('calibration_mode','calibration_source')}
            request['calibration_audio'] = False
        mode = request.get("mode", "demo")
        if mode not in ("demo", "simulated", "acp", "four", "six"):
            raise ValueError("未知数据来源。")
        if mode == "demo":
            if request.get("model_id") != DEMO_ID:
                raise ValueError("界面演示须选择演示项，不会使用真实模型生成虚构预测。")
            cfg = None
            selection = {"participant_id": "DEMO", "model_id": DEMO_ID,
                         "name": "六类交互演示", "synthetic_model": True, "model": None}
            classes = self.display_classes(DEMO_CLASSES, request.get("actions"), request.get("hand_actions"))
        elif mode in ('four', 'six'):
            if mode == 'four' and isinstance(request.get('output', {}), dict) and request.get('output', {}).get('mode', 'simulation') != 'simulation':
                raise ValueError('四分类测试入口仅网页显示，不连接机械手。')
            if mode == 'four':
                from .four_class_online import prepare as prepare_four
                selection, cfg = prepare_four(request['participant'], request['model_id'])
            else:
                from .six_class_online import prepare as prepare_six
                selection, cfg = prepare_six(request['participant'], request['model_id'])
            host = str(request.get('host', '192.168.1.20')).strip()
            if not host or len(host) > 253 or any(c.isspace() for c in host):
                raise ValueError('请填写脑电设备IP。')
            port = number(request.get('port',4097), 'ACP端口', 1, 65535)
            if not port.is_integer() or port == 8712:
                raise ValueError('实时测试使用ACP直连，默认4097；不使用厂家转发端口8712。')
            cfg['device'].update(host=host, port=int(port))
            classes = self.display_classes(cfg['classes'], request.get('actions') if mode == 'six' else None,
                                           request.get('hand_actions', model_preferences(request['model_id']).get('hand_actions')) if mode == 'six' else None)
            if ONLINE_CALIBRATION_ENABLED and selection.get('requires_calibration'):
                calibration_mode = request.get('calibration_mode', 'new')
                if calibration_mode not in ('new', 'reuse'):
                    raise ValueError('请选择使用上次校准或重新校准。')
                cfg['session_metadata']['calibration_mode'] = calibration_mode
                selection['calibration_mode'] = calibration_mode
                if calibration_mode == 'reuse':
                    from .calibration_store import CalibrationStore
                    saved = CalibrationStore(self.output_root).load(request.get('calibration_source'), selection, cfg)
                    cfg['session_metadata']['calibration_reuse'] = saved['summary']
                    cfg['session_metadata'].update(unprompted=True,
                        calibration_timing='explicit_reuse_of_completed_calibration; fresh_EEG_windows_only')
                    selection['calibration_reuse'] = saved['summary']
                    selection['input_note'] = '使用已完成的上次校准，接收新的脑电窗口后持续解码；在线适配效果待实测。'
                sound = request.get('calibration_audio', False)
                if not isinstance(sound, bool):
                    raise ValueError('校准语音设置须为开关。')
                cfg['session_metadata']['calibration_audio_enabled'] = sound and calibration_mode == 'new'
                if sound and calibration_mode == 'new':
                    cfg['session_metadata']['calibration_timing'] = 'browser_render_and_audio_completion_ack_to_received_source_index; not hardware verified'
        else:
            selection, cfg = prepare_subject_model(request["participant"], request["model_id"], registry=self.registry)
            if not 2 <= len(cfg["classes"]) <= 6:
                raise ValueError("当前圆盘支持2至6类；不会自动增减模型类别。")
            if mode == "simulated" and (not selection["synthetic_model"] or cfg["device"]["source"] != "synthetic_tcp"):
                raise ValueError("本机人工脑电模式只支持synthetic_tcp训练的模型。")
            if mode == "acp":
                if selection["synthetic_model"] or cfg["device"]["source"] != "neuracle_acp":
                    raise ValueError("ACP直连须使用ACP采集记录训练的真实被试模型；旧8712模型的参考方式与输入单位尚未确认，不能直接替换端口使用。")
                if cfg["device"]["unit"] != "uV" or (selection.get('runtime') != 'onnx' and not cfg.get("paradigm_id")):
                    raise ValueError("模型缺少ACP输入单位或范式记录。")
                host = str(request.get("host", "")).strip()
                if not host or len(host) > 253 or any(c.isspace() for c in host):
                    raise ValueError("请填写设备IP。")
                cfg["device"].update(host=host, port=int(number(request.get("port", 4097), "ACP端口", 1, 65535)))
                if float(request.get("port", 4097)) != cfg["device"]["port"]:
                    raise ValueError("ACP端口须为整数。")
                if cfg["device"]["port"] == 8712:
                    raise ValueError("8712不是ACP端口。")
            classes = self.display_classes(cfg["classes"], request.get("actions"), request.get("hand_actions"))
            needs_calibration = ONLINE_CALIBRATION_ENABLED and mode == 'acp' and (request.get('session_purpose') == 'calibration'
                or request.get('prompted', {}).get('enabled')
                or (request.get('calibration_mode') == 'reuse' and request.get('calibration_source'))
                or selection.get('runtime')=='retrained_six')
            if selection.get('runtime')=='mi_studio_v2':
                if ONLINE_CALIBRATION_ENABLED and not selection.get('studio_calibration_ready'):
                    raise ValueError('v2模型请先在专用校准工具完成校准，再导入calibrated.pt。')
                if request.get('session_purpose')=='calibration':
                    raise ValueError('v2模型使用专用特征分类头校准，请打开MI Studio v2。')
                needs_calibration=False
            if selection.get('embedded_calibration_ready'):
                if request.get('session_purpose')=='calibration' or (request.get('calibration_mode')=='reuse' and request.get('calibration_source')):
                    raise ValueError('此ONNX已内置校准参数；若需重新校准，请选择对应的源模型。')
                needs_calibration=False
            if selection.get('saved_calibration_ready'):
                if request.get('session_purpose')=='calibration' or (request.get('calibration_mode')=='reuse' and request.get('calibration_source')):
                    raise ValueError('此条目已恢复20轮校准参数；若需重新校准，请选择未绑定参数的原模型。')
                needs_calibration=False
            if needs_calibration:
                from .shared_calibration import SharedCalibrationStore, imported_protocol
                protocol = imported_protocol(cfg)
                selection.update(requires_calibration=True, imported_calibration=True,
                    calibration_rounds=protocol['rounds'], calibration_trials=protocol['trials'],
                    calibration_method='model_specific_logit_bias_v1',backbone_updated=False)
                if selection.get('runtime')=='retrained_six':
                    from .retrained_runtime import CALIBRATION_METHOD
                    selection['calibration_method']=CALIBRATION_METHOD
                cfg['session_metadata']['brain_calibration'] = protocol
                calibration_mode = request.get('calibration_mode', 'new')
                if calibration_mode not in ('new','reuse'):
                    raise ValueError('请选择重新校准或使用完整校准数据。')
                selection['calibration_mode'] = calibration_mode
                if calibration_mode == 'reuse':
                    saved = SharedCalibrationStore(self.output_root).load(request.get('calibration_source'),cfg,protocol)
                    selection['calibration_reuse'] = saved['summary']
                    cfg['session_metadata']['calibration_reuse'] = saved['summary']
                sound = request.get('calibration_audio',False)
                if not isinstance(sound,bool):
                    raise ValueError('校准语音设置须为开关。')
                cfg['session_metadata']['calibration_audio_enabled'] = sound and calibration_mode == 'new'
        if not ONLINE_CALIBRATION_ENABLED:
            selection.update(model_requires_calibration=bool(selection.get('requires_calibration')),
                             requires_calibration=False, online_calibration_disabled=True,
                             calibration_notice=CALIBRATION_DISABLED_MESSAGE)
            if cfg is not None:
                cfg.setdefault('session_metadata', {}).pop('calibration_reuse', None)
                cfg.setdefault('session_metadata', {}).update(online_calibration_disabled=True,
                    calibration_audio_enabled=False, online_calibration_policy='disabled_for_software_debug')
        settings = request.get("decision", {})
        if not isinstance(settings, dict):
            raise ValueError("判决设置格式无效。")
        settings = {**local_settings().get('decision_defaults', {}), **settings}
        stride = local_settings().get('prediction_stride_seconds', selection.get('prediction_stride_seconds', .1))
        if cfg is not None and stride is not None:
            stride = number(stride, '在线预测步长', .02, 1.)
            cfg.setdefault('session_metadata', {}).setdefault('model_stride_seconds', cfg['window']['stride_seconds'])
            cfg['window']['stride_seconds'] = stride
            selection['prediction_stride_seconds'] = stride
        decision_mode = settings.get('mode', 'stable' if mode == 'four' else 'vote')
        if decision_mode == 'vote':
            gate = VotingGate(len(classes), **{k: settings[k] for k in
                              ('vote_window_seconds', 'vote_ratio', 'decision_timeout_seconds',
                               'cooldown_seconds', 'max_gap_seconds') if k in settings})
        elif decision_mode == 'stable':
            gate = ContinuousGate(len(classes), **{k: settings[k] for k in
                                  ('confidence', 'margin', 'stable_seconds', 'cooldown_seconds',
                                   'release_seconds', 'max_gap_seconds') if k in settings})
        else:
            raise ValueError('请选择投票判决或连续稳定判决。')
        return selection, cfg, classes, gate, mode

    def check(self, request):
        selection, cfg, classes, gate, mode = self.prepare(request)
        output = validate_output(request.get("output", {}), len(classes), mode, classes)
        from .prompted_control import validate_settings
        prompted = validate_settings(request.get('prompted'), mode, classes, gate, output)
        if mode == 'four':
            from .four_class_online import load_checkpoint
            load_checkpoint(selection['model'], selection['sha256'])
        elif selection.get('runtime') == 'onnx':
            from .model_bundle import ONNXDecoder
            ONNXDecoder(selection['model'], prediction_stride_seconds=cfg['window']['stride_seconds'])
        elif selection.get('runtime')=='retrained_six':
            from .retrained_runtime import decoder_from_selection
            decoder_from_selection(selection,cfg)
        elif selection.get('runtime')=='mi_studio_v2':
            from .studio_v2_runtime import StudioV2Decoder
            StudioV2Decoder(selection['model'],cfg,selection['sha256'])
        elif mode == 'six':
            from .six_class_online import load_checkpoint
            if selection.get('right_hand_review'):
                from .six_class_right_review import load_checkpoint
            elif selection.get('requires_calibration'):
                from .six_class_improved import load_checkpoint
            load_checkpoint(selection['model'], selection['sha256'])
        return {"connected": False, "mode": mode, "selection": selection, "classes": classes,
                "output": output, "decision": gate.settings(), "prompted": prompted,
                "message": ('四分类权重及预处理已检查；开始后使用采集工作台的ACP连接，核对通道名称，仅网页显示。旧模型历史参考设置未完整记录，实时效果须实测。'
                            if mode == 'four' else selection['input_note']+' 模型与原采集来源校验通过，尚未连接设备。'
                            if mode == 'six' else "模型和判决设置已检查；尚未连接设备。")}

    def check_eeg_connection(self, request):
        with self.condition:
            self._require_connection_ready()
            if self.eeg_probe_busy or self.eeg_discovery_busy or (self.worker is not None and self.worker.is_alive()):
                raise ValueError('正在测试连接、解码或保存，请等待当前操作结束。')
            self.eeg_probe_busy = True
            self.eeg_probe_done.clear()
            self.eeg_probe_cancel.clear()
            self.state['eeg_connection_test'] = dict(phase='testing', message='正在连接设备并检查连续脑电数据…')
        try:
            selection, cfg, classes, gate, mode = self.prepare(request)
            if mode not in ('six', 'four', 'acp'):
                raise ValueError('连接测试仅用于真实脑电模型。')
            from .eeg_connection import probe_acp
            result = probe_acp(cfg['device'], self.eeg_probe_cancel)
            if cfg['session_metadata'].get('calibration_reuse'):
                from .calibration_store import verify_device
                verify_device(cfg['session_metadata']['calibration_reuse']['device'], result['device'])
            result['model_id'] = selection['model_id']
            result['endpoint'] = f"{cfg['device']['host']}:{cfg['device']['port']}"
            self._remember_eeg_hosts([cfg['device']['host']])
            with self.condition:
                self.state['eeg_connection_test'] = dict(phase='completed', **result)
            return result
        except Exception as exc:
            with self.condition:
                self.state['eeg_connection_test'] = dict(phase='failed', message=str(exc))
            raise
        finally:
            with self.condition:
                self.eeg_probe_busy = False
                self.eeg_probe_done.set()

    def _remember_eeg_hosts(self, hosts):
        from .eeg_connection import discovery_hosts
        candidates = discovery_hosts(hosts + self.eeg_known_hosts)
        if candidates != self.eeg_known_hosts:
            write_json(self.eeg_hosts_path, candidates)
            self.eeg_known_hosts = candidates

    def discover_trigger_devices(self, request=None):
        with self.condition:
            self._require_connection_ready()
            if self.eeg_probe_busy or self.eeg_discovery_busy or (self.worker is not None and self.worker.is_alive()):
                raise ValueError('正在搜索、测试或解码，请等待当前操作结束。')
            self.eeg_discovery_busy = True
            self.state['trigger_discovery'] = dict(phase='searching')
        try:
            from .trigger_connection import discover_trigger_devices
            result = discover_trigger_devices((request or {}).get('known_hosts', []))
            with self.condition:
                self.state['trigger_discovery'] = dict(phase='completed', **result)
            return result
        except Exception as exc:
            with self.condition:
                self.state['trigger_discovery'] = dict(phase='failed', message=str(exc))
            raise
        finally:
            with self.condition:
                self.eeg_discovery_busy = False

    def check_trigger_connection(self, request):
        with self.condition:
            self._require_connection_ready()
            if (self.eeg_probe_busy or self.eeg_discovery_busy or self.manual_hand.snapshot()['busy']
                    or (self.worker is not None and self.worker.is_alive())):
                raise ValueError('正在搜索、测试或解码，请等待当前操作结束。')
            self.eeg_probe_busy = True
            self.eeg_probe_done.clear()
            self.eeg_probe_cancel.clear()
            self.state['trigger_connection_test'] = dict(phase='testing', message='正在连接TriggerBox并核验事件回传…')
        try:
            selection, cfg, classes, gate, mode = self.prepare(request)
            # This check never opens the hand. Only reject a shared serial port;
            # hand poses and movement parameters are irrelevant to marker tests.
            output = request.get('output', {})
            if not isinstance(output, dict) or output.get('mode', 'simulation') not in ('simulation', 'inspire'):
                raise ValueError('动作输出设置格式无效。')
            from .prompted_control import validate_settings
            settings = validate_settings(request.get('prompted'), mode, classes, gate, output)
            if not settings['enabled']:
                raise ValueError('请先选择带提示采集。')
            if settings['marking'] == 'software':
                result = dict(ready=True, mode='software', echo_verified=False, physical_hardware=False,
                              message='软件标记已就绪；指令发出时记录，无需TriggerBox或回传自检。')
                with self.condition:
                    self.state['trigger_connection_test'] = dict(phase='completed', **result)
                return result
            if settings['trigger_host'] == cfg['device']['host']:
                raise ValueError('TriggerBox数据设备不能与脑电设备相同。')
            from .trigger_probe import run_trigger_check
            folder = self.output_root/'trigger_checks'/(time.strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8])
            result = run_trigger_check(folder, cfg['device'], settings, self.eeg_probe_cancel)
            result['diagnostic_path'] = str(folder.resolve())
            with self.condition:
                self.state['trigger_connection_test'] = dict(phase='completed', **result)
            return result
        except Exception as exc:
            with self.condition:
                self.state['trigger_connection_test'] = dict(phase='failed', message=str(exc))
            raise
        finally:
            with self.condition:
                self.eeg_probe_busy = False
                self.eeg_probe_done.set()

    def discover_eeg_devices(self, request=None):
        with self.condition:
            self._require_connection_ready()
            if self.eeg_probe_busy or self.eeg_discovery_busy or (self.worker is not None and self.worker.is_alive()):
                raise ValueError('正在搜索、测试或解码，请等待当前操作结束。')
            self.eeg_discovery_busy = True
            self.state['eeg_discovery'] = dict(phase='searching')
        try:
            from .eeg_connection import discover_eeg, discovery_hosts
            hints = discovery_hosts((request or {}).get('known_hosts', []))
            devices = discover_eeg(known_hosts=discovery_hosts(hints + self.eeg_known_hosts))
            self._remember_eeg_hosts([d['ip'] for d in devices])
            result = dict(devices=devices, message=f'找到{len(devices)}台脑电设备。' if devices else
                '未发现脑电设备，请确认设备已开机，电脑与设备处于同一网络后重新搜索。')
            with self.condition:
                self.state['eeg_discovery'] = dict(phase='completed', **result)
            return result
        except Exception as exc:
            with self.condition:
                self.state['eeg_discovery'] = dict(phase='failed', message=str(exc))
            raise
        finally:
            with self.condition:
                self.eeg_discovery_busy = False

    def start_calibration(self, request):
        if not ONLINE_CALIBRATION_ENABLED:
            raise ValueError(CALIBRATION_DISABLED_MESSAGE)
        # Calibration is a separate EEG-only session. Do not even validate or
        # open the hand/TriggerBox supplied by the decode configuration.
        payload = {k:request[k] for k in ('participant','model_id','mode','host','port',
                   'calibration_audio','calibration_audio_ready') if k in request}
        payload.update(calibration_mode='new', session_purpose='calibration',
                       output={'mode':'simulation'}, prompted={'enabled':False})
        selection, cfg, classes, gate, mode = self.prepare(payload)
        if mode not in ('six','acp') or not selection.get('requires_calibration'):
            raise ValueError('当前模型不需要脑电校准，可直接点击开始解码。')
        return self.start(payload)

    def start_decoding(self, request):
        selection, cfg, classes, gate, mode = self.prepare(request)
        if selection.get('requires_calibration') and not selection.get('calibration_reuse'):
            raise ValueError('请先在脑电校准模块完成校准，或选择使用上次完整校准，再点击开始解码。')
        return self.start(dict(request, session_purpose='decode'))

    def start(self, request):
        with self.condition:
            self._require_connection_ready()
            if self.eeg_probe_busy or self.eeg_discovery_busy:
                raise ValueError('脑电连接测试正在进行，请等待测试结束后再开始解码。')
            if self.manual_hand.snapshot()["busy"]:
                raise ValueError("机械手独立测试正在执行，请先停止并等待完成。")
            if self.worker is not None and self.worker.is_alive():
                raise ValueError("已有会话正在运行或保存，请先停止并等待保存完成。")
            selection, cfg, classes, gate, mode = self.prepare(request)
            purpose = request.get('session_purpose', 'legacy')
            subject_window_id = request.get('subject_window_id')
            monitor_window_id = request.get('monitor_window_id')
            if monitor_window_id is not None:
                try:
                    monitor_window_id = str(uuid.UUID(monitor_window_id))
                except (ValueError, TypeError, AttributeError):
                    raise ValueError('观察窗口标识无效，请刷新工作台。')
                if purpose != 'decode' or subject_window_id is not None:
                    raise ValueError('观察窗口仅用于主屏自由解码。')
            if subject_window_id is not None:
                try:
                    subject_window_id = str(uuid.UUID(subject_window_id))
                except (ValueError, TypeError, AttributeError):
                    raise ValueError('任务窗口标识无效，请从工作台重新打开。')
                if purpose != 'decode':
                    raise ValueError('独立任务窗口仅用于开始解码。')
            if purpose not in ('legacy','calibration','decode'):
                raise ValueError('未知会话用途。')
            if purpose != 'calibration' and selection.get('requires_calibration') and not selection.get('calibration_reuse'):
                raise ValueError('请先完成或选用完整校准，再开始解码。')
            if purpose == 'calibration' and (request.get('output',{}).get('mode','simulation') != 'simulation'
                                            or request.get('prompted',{}).get('enabled')):
                raise ValueError('脑电校准会话不允许启用机械手或TriggerBox任务采集。')
            if selection.get('requires_calibration') and cfg['session_metadata'].get('calibration_audio_enabled'):
                if request.get('calibration_audio_ready') is not True:
                    raise ValueError('请先准备并解锁校准语音，再开始校准。')
            output = validate_output(request.get("output", {}), len(classes), mode, classes)
            from .prompted_control import validate_settings
            prompted = validate_settings(request.get('prompted'), mode, classes, gate, output)
            if monitor_window_id and prompted['enabled']:
                raise ValueError('带提示采集须从任务窗口开始。')
            if prompted['enabled']:
                gate.cooldown = prompted['rest_seconds']+prompted['prepare_seconds']
            if prompted['enabled'] and prompted['trigger_host'] == cfg['device']['host']:
                raise ValueError('TriggerBox无线数据地址不能与脑电设备地址相同。')
            selection = {**selection, 'driver': 'inspire' if output['mode'] == 'inspire' else 'mock',
                         'physical_output_enabled': output['mode'] == 'inspire'}
            self.stop_event.clear()
            self.eeg_fault_active.clear()
            self.trace.clear()
            self.waveform_preview = WaveformPreview(cfg['device']['channels'], cfg['device']['sample_rate'],
                                                    cfg['device']['unit']) if cfg else WaveformPreview()
            self.target, self.auto_demo = None, True
            self.output_config, self.hand_output = output, None
            self.motion_completed_at = None
            self.fresh_collection_after = None
            self.fresh_source_cutoff = None
            self.live_device_clock = None
            self.simulation_motion = None
            self.calibration_decoder = None
            self.session_purpose = purpose
            if cfg is not None:
                cfg.setdefault('session_metadata',{})['session_purpose'] = purpose
            self.prompted = None
            self.prompted_settings = prompted
            self.prompted_origin_ms = None
            self.last_browser_seen = time.monotonic()
            self.state = {"status": "starting", "running": True, "session": "", "error": "",
                          "session_purpose": purpose,
                          "subject_window_id": subject_window_id,
                          "monitor_window_id": monitor_window_id,
                          "samples_received": 0, "n_predictions": 0, "commands": [], "probabilities": [],
                          "phase": "warming", "inference_ms": None, "report": None,
                          "calibration": {"required": bool(selection.get('requires_calibration')),
                              "ready": not ONLINE_CALIBRATION_ENABLED or bool(selection.get('calibration_reuse')),
                              "disabled": not ONLINE_CALIBRATION_ENABLED,
                              "completed": selection.get('calibration_trials',18) if selection.get('calibration_reuse') else 0,
                              "reused": bool(selection.get('calibration_reuse')),
                              "reused_from": selection.get('calibration_reuse'),
                              "total": selection.get('calibration_trials',18), "phase": "warming"},
                          "mode": mode, "classes": classes, "selection": selection, "gate": gate.state,
                          "decision_settings": gate.settings(),
                          "prompted": dict(prompted, phase='connecting') if prompted['enabled'] else prompted,
                          "wall_seconds": 0., "source_seconds": 0., "rate_hz": None,
                          "expected_rate": cfg["device"]["sample_rate"] if cfg else None,
                          "eeg_endpoint": {k:cfg['device'][k] for k in ('host','port')} if cfg else None,
                          "gap_count": 0, "stale_seconds": 0., "saved_path": "", "physical_action": False,
                          "stream_quality": dict(loss_events=0, corrupt_events=0, missing_samples=0, recovering=False),
                          "output_mode": output["mode"], "hand": idle_state(output["mode"]),
                          "output_config": copy.deepcopy(output),
                          "window_seconds": cfg["window"]["seconds"] if cfg else None,
                          "simulation_label": None, "auto_demo": True, "unit": cfg["device"]["unit"] if cfg else ""}
            self.worker = threading.Thread(target=self._continuous, args=(selection, cfg, classes, gate, mode),
                                           daemon=True, name="continuous-online")
            self.worker.start()
            return self.snapshot()

    def acknowledge_calibration(self, request):
        with self.condition:
            if (not self.state['running'] or self.stop_event.is_set()
                    or self.calibration_decoder is None):
                raise ValueError('没有正在进行的脑电校准。')
            return self.calibration_decoder.acknowledge(request.get('prompt_id'), request.get('session_id'),
                                                        request.get('audio'))

    def acknowledge_prompted(self, request):
        with self.condition:
            owner = self.state.get('subject_window_id')
            if owner and request.get('subject_window_id') != owner:
                raise ValueError('只有本次任务窗口可以确认提示显示。')
            if not self.state['running'] or self.stop_event.is_set() or self.prompted is None:
                raise ValueError('没有正在进行的带提示采集。')
            result = self.prompted.acknowledge(request)
            self.state['prompted'] = self.prompted.snapshot()
            return result

    def prepare_calibration_audio(self, request):
        from .six_class_online import NAMES
        with self.condition:
            if self.state['running']:
                raise ValueError('请先停止当前会话，再准备或试听语音。')
        allowed = ['开始', '休息', '结束', '校准完成', '拇指', '食指', '中指', '无名指', '小指', 'OK手势'] + NAMES
        for entry in super().models():
            try:
                _, cfg = prepare_subject_model(entry['participant_id'],entry['model_id'],registry=self.registry)
                allowed.extend(c.get('task_text') or c['name'] for c in cfg['classes'])
            except (OSError,ValueError,KeyError,RuntimeError,EOFError):
                continue
        requested = request.get('texts')
        if not isinstance(requested, list) or any(text not in allowed for text in requested):
            raise ValueError('提示语音文字不匹配。')
        return self.speech.prepare(requested)

    def stop(self):
        with self.condition:
            self.eeg_probe_cancel.set()
            if self.eeg_probe_busy:
                for key in ('trigger_connection_test','eeg_connection_test'):
                    if self.state.get(key,{}).get('phase')=='testing':
                        self.state[key]=dict(phase='cancelling',message='正在取消连接检验并释放设备…')
            self.manual_hand.stop()
            self.stop_event.set()
            if self.hand_output is not None:
                self.hand_output.cancel.set()
            if self.state["running"]:
                self.state.update(status="stopping", gate={"candidate": None, "progress": 0., "reason": "正在停止并保存"})

    def set_simulation(self, label):
        with self.condition:
            if not self.state["running"] or self.state["mode"] not in ("demo", "simulated"):
                raise ValueError("仅运行中的演示或人工脑电允许切换输入。")
            if isinstance(label, bool) or not isinstance(label, int) or not -2 <= label < len(self.state["classes"]):
                raise ValueError("演示类别无效。")
            self.auto_demo = label == -2
            self.target = label if label >= 0 else None
            self.state.update(simulation_label=self.target, auto_demo=self.auto_demo)
            return {"simulation_label": self.target, "auto_demo": self.auto_demo}

    def _begin_fresh_window(self, completed_mono, gate):
        if isinstance(gate, VotingGate) and self.state['mode'] == 'six':
            self.fresh_collection_after = completed_mono+gate.cooldown
            self.fresh_source_cutoff = None

    def _fresh_window_ready(self, window, gate):
        if self.fresh_collection_after is None:
            return True
        if time.monotonic() < self.fresh_collection_after:
            reason = '回位后间隔中 · 尚未收集新窗口'
        elif window is None:
            reason = '等待新脑电窗口信息 · 尚未投票'
        else:
            if self.fresh_source_cutoff is None:
                # Rest ends on the host clock. A delayed last packet must not
                # move that boundary back onto pre-return samples.
                clock = self.live_device_clock
                stamp = clock.estimate(int(self.fresh_collection_after*1e9)) if clock else None
                self.fresh_source_cutoff = (math.ceil((stamp-clock.origin_ms)*clock.fs/1000)
                    if stamp is not None else window['received_end_index']+1)
            if window['source_start_index'] >= self.fresh_source_cutoff:
                self.fresh_collection_after = None
                return True
            reason = '正在收集完整的新脑电窗口 · 尚未投票'
        gate.reset_candidate()
        gate._waiting(reason)
        gate.state['fresh_source_cutoff'] = self.fresh_source_cutoff
        return False

    def _prediction(self, p, elapsed, inference_ms, gate, predictions, commands, inference_details=None,
                    prediction_window=None, input_age_seconds=0.):
        with self.condition:
            if self.session_purpose == 'calibration':
                return
            if self.eeg_fault_active.is_set():
                gate.reset_candidate()
                self.state['gate'] = dict(candidate=None,progress=0.,reason='接收异常 · 暂停判决')
                return
            if self.stop_event.is_set() or (self.state.get('calibration', {}).get('required')
                                           and not self.state['calibration'].get('ready')):
                return
            self._advance_motion(elapsed, gate)
            prompted_ready = True
            if self.prompted is not None:
                prompted_ready = self.prompted.before_prediction(prediction_window, self.prompted_origin_ms)
            if self.state["hand"]["busy"]:
                # Track release/change while moving, but never accumulate or queue.
                gate.update(p, elapsed, inhibited=True)
                if self.output_config['mode'] == 'simulation':
                    gate.state['reason'] = '网页模拟往返中 · 暂不输出下一动作'
                event = None
            elif self.prompted is not None and not prompted_ready:
                gate.reset_candidate()
                gate._waiting('按目标提示想象 · 正在等待本次完整脑电窗口' if self.prompted.phase == 'decoding'
                              else '休息' if self.prompted.phase == 'executing'
                              else '保持本次运动想象' if self.prompted.phase in ('awaiting_display', 'marking_cue', 'cue_hold')
                              else '带提示采集 · '+self.prompted.phase)
                event = None
            elif self.prompted is None and isinstance(gate, VotingGate) and not self._fresh_window_ready(prediction_window, gate):
                event = None
            else:
                event = gate.update(p, elapsed)
                if self.prompted is not None:
                    if self.prompted.first_prediction is None:
                        self.prompted.first_prediction = elapsed
                    if event is None and elapsed-self.prompted.first_prediction >= gate.timeout+.2:
                        # Preserve the existing highest-vote timeout. Only when it
                        # still cannot produce a valid decision is this no-output.
                        self.prompted.no_output()
            self.state.update(probabilities=np.asarray(p).tolist(), gate=dict(gate.state),
                              inference_ms=inference_ms, phase="decoding", status="running")
            self.state["n_predictions"] += 1
            row = {"wall_seconds": elapsed, "probabilities": np.asarray(p).tolist(), "gate": gate.state,
                   "input_age_seconds": input_age_seconds, "delayed_input": input_age_seconds > 1.}
            if prediction_window is not None:
                row['input_window'] = prediction_window
                row['fresh_source_cutoff'] = self.fresh_source_cutoff
            if inference_details is not None:
                row['router'] = inference_details
            predictions.write(json.dumps(row, ensure_ascii=False)+"\n")
            if event:
                self._emit_decision(event, gate, commands)
            if self.prompted is not None:
                self.state['prompted'] = self.prompted.snapshot()

    def _advance_motion(self, elapsed, gate):
        if self.simulation_motion is not None:
            age = time.monotonic()-self.simulation_motion['started_mono']
            hold = self.simulation_motion.get('hold_seconds', 0.)
            if age >= 1.2+hold:
                self._finish_simulation()
                gate.last_command = elapsed
                gate.reset_candidate()
                self._begin_fresh_window(time.monotonic(), gate)
            elif age >= .6+hold:
                self.state['hand']['phase'] = 'recovering'
            elif age >= .6:
                self.state['hand']['phase'] = 'holding'
        if self.motion_completed_at is not None:
            gate.last_command = elapsed-(time.monotonic()-self.motion_completed_at)
            gate.reset_candidate()
            self._begin_fresh_window(self.motion_completed_at, gate)
            self.motion_completed_at = None

    def _vote_tick(self, elapsed, gate, commands):
        """Deadlines advance even while every new slice is rejected or absent."""
        with self.condition:
            if (not isinstance(gate, VotingGate) or self.stop_event.is_set()
                    or self.eeg_fault_active.is_set() or self.session_purpose == 'calibration'):
                return
            self._advance_motion(elapsed, gate)
            if self.state['hand']['busy'] or (self.prompted is not None and self.prompted.phase != 'decoding'):
                return
            event = gate.tick(elapsed)
            self.state['gate'] = dict(gate.state)
            if event:
                self._emit_decision(event, gate, commands)
            elif gate.state.get('decision_reason') == 'deadline_no_votes' and self.prompted is not None:
                self.prompted.no_output()
            if self.prompted is not None:
                self.state['prompted'] = self.prompted.snapshot()

    def _emit_decision(self, event, gate, commands):
        c = self.state["classes"][event["class_id"]]
        real = self.output_config["mode"] == "inspire"
        simulate_motion = not real and self.state['mode'] != 'four'
        event.update(action=c["action"], action_name=c["action_name"], task=c.get("task_text", c["name"]),
                     model_class_id=c['id'], model_class_name=c['name'], hand_action=c.get('hand_action'),
                     driver="inspire" if real else "mock", physical_action=real,
                     recovery='start' if real or simulate_motion else None,
                     execution_status="executing" if real else "simulating" if simulate_motion else "simulated",
                     sequence=self.state.get("command_count", 0)+1)
        if self.prompted is not None:
            self.prompted.decision(event)
        event['class_evidence'] = build_class_evidence(self.state['classes'], event, event.get('mean_probabilities', self.state.get('probabilities', [])))
        event['instruction_evidence'] = instruction_evidence(event)
        if real:
            if self.hand_output is None or not self.hand_output.submit(event):
                raise RuntimeError("机械手未就绪，未排队或重发本次指令。")
            self.state["hand"].update(busy=True, phase="executing")
            self.state["physical_action"] = True
            gate.state["reason"] = "已下发机械手动作 · 等待到位"
            self.state["gate"] = dict(gate.state)
        elif simulate_motion:
            hold = self.output_config.get('action_profiles', {}).get(c.get('hand_action'), {}).get('hold_seconds', 0.)
            self.simulation_motion = dict(sequence=event['sequence'], started_mono=time.monotonic(), hold_seconds=hold)
            self.state['hand'].update(busy=True, phase='executing')
            gate.state['reason'] = '网页模拟：动作去程，随后返回起始姿态'
            self.state['gate'] = dict(gate.state)
        self.state["commands"].append(event)
        # Full history stays on disk; the browser retains the latest 100.
        self.state["commands"] = self.state["commands"][-100:]
        self.state["command_count"] = self.state.get("command_count", 0)+1
        event["sequence"] = self.state["command_count"]
        commands.write(json.dumps(event, ensure_ascii=False)+"\n")
        commands.flush()

    def _finish_simulation(self, *, cancelled=False):
        with self.condition:
            motion = self.simulation_motion
            if motion is None:
                return
            status = 'cancelled' if cancelled else 'simulated'
            duration = time.monotonic()-motion['started_mono']
            result = dict(kind='simulation_result', sequence=motion['sequence'], physical_action=False,
                          execution_status=status, recovery='start', simulation_duration_seconds=duration)
            if self.prompted is not None:
                self.prompted.motion(result)
            for command in self.state['commands']:
                if command['sequence'] == motion['sequence']:
                    command.update(execution_status=status, simulation_duration_seconds=duration)
            self.state['hand'].update(busy=False, phase='simulation', last_duration_seconds=None)
            if self.hand_log is not None:
                self.hand_log.write(json.dumps(result, ensure_ascii=False)+'\n')
                self.hand_log.flush()
            self.simulation_motion = None

    def _hand_event(self, state, event):
        with self.condition:
            self.state["hand"] = state
            if event:
                if event.get('kind') == 'initialization_result':
                    self.state['hand_initialization'] = event
                    self.state['physical_action'] = self.state['physical_action'] or event['physical_action']
                    if self.hand_log is not None:
                        self.hand_log.write(json.dumps(event, ensure_ascii=False)+'\n')
                        self.hand_log.flush()
                    if event['execution_status']=='failed':self.stop_event.set()
                    return
                self.motion_completed_at = event["completed_mono"]
                event = copy.deepcopy(event)
                event['feedback'] = motion_feedback(event)
                for command in self.state["commands"]:
                    if command["sequence"] == event["sequence"]:
                        command.update(execution_status=event["execution_status"],
                                       duration_seconds=event["duration_seconds"], error=event["error"],
                                       actual_feedback=copy.deepcopy(event['feedback']))
                        command['actual_evidence'] = actual_evidence(command, event['feedback'])
                        for key in ('class_evidence', 'instruction_evidence', 'actual_evidence'):
                            event[key] = copy.deepcopy(command.get(key))
                        event['instruction'] = {key: command.get(key) for key in
                            ('hand_action', 'action_name', 'task', 'model_class_id',
                             'target_label', 'target_hand_action', 'target_task')}
                if self.prompted is not None:
                    self.prompted.motion(event)
                if self.hand_log is not None:
                    self.hand_log.write(json.dumps(event, ensure_ascii=False)+"\n")
                    self.hand_log.flush()
                if event["execution_status"] == "failed":
                    self.stop_event.set()

    def _continuous(self, selection, cfg, classes, gate, mode):
        status, error, session = "stopped", "", None
        try:
            participant = validate_participant_id(selection["participant_id"])
            session = self.output_root / (time.strftime("%Y%m%d_%H%M%S_")+uuid.uuid4().hex[:8]+"_"+participant)
            session.mkdir(parents=True, exist_ok=False)
            write_json(session/"model_selection.json", {**selection, "mode": mode, "classes": classes,
                       "session_purpose": self.session_purpose,
                       "decision": gate.settings(),
                       "prompted": self.prompted_settings,
                       "labelled_for_training": False, "output": self.output_config, "config": cfg})
            with self.condition:
                self.state["session"] = str(session.resolve())
            with (session/"predictions.jsonl").open("x", encoding="utf-8") as predictions, \
                 (session/"commands.jsonl").open("x", encoding="utf-8") as commands:
                with (session/"hand_events.jsonl").open("x", encoding="utf-8") as hand_log:
                    self.hand_log = hand_log
                    try:
                        if self.output_config["mode"] == "inspire":
                            self.hand_output = HandOutput(self.output_config, self._hand_event)
                            try:
                                self.hand_output.connect()
                            finally:
                                write_json(session/'hand_connection.json', self.hand_output.snapshot())
                            try:
                                self.hand_output.initialize()
                            finally:
                                write_json(session/'hand_initialization.json', self.hand_output.snapshot().get('initialization'))
                        if not self.stop_event.is_set():
                            if mode == "demo":
                                self._demo(gate, predictions, commands)
                            else:
                                self._model_loop(selection, cfg, mode, gate, predictions, commands, session)
                    finally:
                        self._finish_simulation(cancelled=True)
                        if self.hand_output is not None:
                            self.hand_output.close()
                        self.hand_log = None
                    if self.state["hand"]["phase"] == "failed":
                        raise RuntimeError(self.state["hand"]["error"] or "机械手动作失败。")
        except Exception as exc:
            status, error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            with self.condition:
                report = {"status": status, "error": error, "mode": mode,
                          "physical_action": self.state["physical_action"], "labelled_for_training": False,
                          "output_mode": self.output_config["mode"], "hand": self.state["hand"],
                          "n_predictions": self.state["n_predictions"], "n_commands": self.state.get("command_count", 0),
                          "samples_received": self.state["samples_received"], "wall_seconds": self.state["wall_seconds"],
                          "gap_count": self.state["gap_count"], "model_id": selection["model_id"]}
                report['stream_timing'] = self.state.get('stream_timing')
                report['raw_samples_saved'] = self.state.get('raw_samples_saved')
                report['stream_recovery'] = self.state.get('stream_recovery')
                report['stream_quality'] = self.state.get('stream_quality')
                report['calibration'] = self.state.get('calibration', {})
                report['session_purpose'] = self.session_purpose
                report['prompted'] = self.state.get('prompted', {'enabled': False})
                try:
                    if session:
                        write_json(session/"report.json", report)
                except OSError as exc:
                    status, error = "failed", f"保存失败：{exc}"
                self.state.update(status=status, error=error, running=False, phase="finished", report=report,
                                  saved_path=str(session.resolve()) if session else "",
                                  gate={"candidate": None, "progress": 0., "reason": "已停止"})

    def _heartbeat(self):
        limit = 3 if self.prompted_settings['enabled'] else 15
        if time.monotonic()-self.last_browser_seen > limit:
            if self.prompted_settings['enabled']:
                raise RuntimeError('带提示采集网页超过3秒未响应，已停止并保存。')
            raise RuntimeError("网页超过15秒未响应，已停止解码并保存。")

    def _demo(self, gate, predictions, commands):
        origin = time.monotonic()
        p = np.full(6, 1/6)
        while not self.stop_event.wait(.05):
            self._heartbeat()
            elapsed = time.monotonic()-origin
            with self.condition:
                target = int(elapsed//5) % 6 if self.auto_demo and elapsed % 5 < 3.8 else self.target if not self.auto_demo else None
                self.state.update(wall_seconds=elapsed, simulation_label=target)
            wanted = np.full(6, .025) if target is not None else np.full(6, 1/6)
            if target is not None:
                wanted[target] = .875
            p += (wanted-p)*.22
            self._prediction(p, elapsed, None, gate, predictions, commands)

    def _model_loop(self, selection, cfg, mode, gate, predictions, commands, session):
        from .runtime import OnlineDecoder
        from .training import load_model
        if mode == 'four':
            from .four_class_online import load_checkpoint, FourClassDecoder
            payload, model = load_checkpoint(selection['model'], selection['sha256'])
            decoder = FourClassDecoder(payload, model)
        elif mode == 'six':
            from .six_class_online import load_checkpoint, SixClassDecoder
            if selection.get('right_hand_review'):
                from .six_class_right_review import load_checkpoint, RightHandSixClassDecoder as SixClassDecoder
            elif selection.get('requires_calibration') or selection.get('model_requires_calibration'):
                from .six_class_improved import load_checkpoint, ImprovedSixClassDecoder as SixClassDecoder
            payload, model = load_checkpoint(selection['model'], selection['sha256'])
            decoder = SixClassDecoder(payload, model, cfg)
            if selection.get('requires_calibration'):
                reuse = cfg['session_metadata'].get('calibration_reuse')
                if reuse:
                    from .calibration_store import CalibrationStore
                    saved = CalibrationStore(self.output_root).load(reuse['id'], selection, cfg)
                    if saved['summary'] != reuse:
                        raise ValueError('上次校准文件在开始前发生变化，请重新选择。')
                    decoder.restore_calibration(saved)
                with self.condition:
                    self.calibration_decoder = decoder
                    self.state['calibration'] = decoder.snapshot_calibration()
        elif selection.get('runtime') == 'mi_studio_v2':
            from .studio_v2_runtime import StudioV2Decoder
            decoder = StudioV2Decoder(selection['model'],cfg,selection['sha256'])
        elif selection.get('runtime') == 'onnx':
            from .model_bundle import ONNXDecoder
            decoder = ONNXDecoder(selection['model'], prediction_stride_seconds=cfg['window']['stride_seconds'])
        elif selection.get('runtime')=='retrained_six':
            from .retrained_runtime import decoder_from_selection
            decoder=decoder_from_selection(selection,cfg)
        else:
            payload, model = load_model(selection["model"], "cpu")
            payload = dict(payload, config=copy.deepcopy(payload['config']))
            payload['config'].setdefault('decision', {})['update_seconds'] = cfg['window']['stride_seconds']
            decoder = OnlineDecoder(payload, model)
        if selection.get('imported_calibration'):
            from .shared_calibration import ImportedCalibrationDecoder, SharedCalibrationStore, imported_protocol
            decoder = ImportedCalibrationDecoder(decoder,cfg,selection)
            reuse = cfg['session_metadata'].get('calibration_reuse')
            if reuse:
                saved = SharedCalibrationStore(self.output_root).load(reuse['id'],cfg,imported_protocol(cfg),arrays=True)
                if saved['summary'] != reuse:
                    raise ValueError('共用校准数据发生变化，请重新选择。')
                decoder.restore_calibration(saved)
            with self.condition:
                self.calibration_decoder = decoder
                self.state['calibration'] = decoder.snapshot_calibration()
        from .receiver_access import ReceiverLease
        lease = ReceiverLease(cfg["device"]["host"], cfg["device"]["port"]) if mode in ('acp','four','six') else nullcontext()
        try:
            with lease:
                if self.stop_event.is_set():
                    return
                if mode in ('acp','four','six'):
                    self._acp_loop(cfg, decoder, gate, predictions, commands, session)
                else:
                    self._tcp_loop(cfg, decoder, gate, predictions, commands, session)
        finally:
            if ONLINE_CALIBRATION_ENABLED and hasattr(decoder, 'save_calibration'):
                decoder.save_calibration(session)
            with self.condition:
                self.calibration_decoder = None

    def _consume_chunk(self, raw, ix, origin, decoder, gate, predictions, commands, rates,
                       received_timing=None, stored_samples=None):
        self._heartbeat()
        elapsed = time.monotonic()-origin
        rates.append((time.monotonic(), len(ix)))
        while rates and time.monotonic()-rates[0][0] > 3:
            rates.popleft()
        processing_rate = sum(n for _, n in rates)/(rates[-1][0]-rates[0][0]+len(ix)/decoder.stream.fs) if rates else 0
        rate = processing_rate
        input_age = 0.
        if received_timing is not None:
            timing = received_timing.snapshot(int(ix[-1])+1 if stored_samples is None else stored_samples)
            rate = timing['received_rate_hz']
            with self.condition:
                self.state.update(stream_timing=timing, rate_hz=rate, processing_rate_hz=processing_rate)
            received_timing.check(timing, allow_packet_loss=True)
            clock = timing['device_timeline']
            if clock['estimated_device_now_ms'] is not None:
                input_age = max(0., (clock['estimated_device_now_ms']-clock['device_origin_ms'])/1000
                                -int(ix[-1])/decoder.stream.fs)
        elif elapsed > 3.5 and rates[-1][0]-rates[0][0] >= 1.5 and not .8*decoder.stream.fs <= rate <= 1.2*decoder.stream.fs:
            raise RuntimeError(f"实际接收速率约{rate:.1f} Hz，与模型要求{decoder.stream.fs:g} Hz不符；已停止，未补点或改采样率。")
        started = time.perf_counter()
        with self.condition:
            self.waveform_preview.append(raw, ix)
        if hasattr(decoder,'set_prompt_anchor') and self.prompted is not None:
            mark=self.prompted.prepare_mark
            anchor=None
            if mark and not mark.get('verification_pending'):
                anchor=(round((mark['observed_device_ms']-self.prompted_origin_ms)*decoder.stream.fs/1000)
                        if mark.get('echo_verified') and mark.get('observed_device_ms') is not None
                        and self.prompted_origin_ms is not None else mark.get('estimated_source_index'))
            decoder.set_prompt_anchor(anchor)
        result = decoder.process(raw, ix)
        ms = (time.perf_counter()-started)*1000
        calibration = decoder.snapshot_calibration() if hasattr(decoder, 'snapshot_calibration') else None
        with self.condition:
            self.state.update(samples_received=self.state["samples_received"]+len(ix) if stored_samples is None else stored_samples, wall_seconds=elapsed,
                              source_seconds=(int(ix[-1])+1)/decoder.stream.fs, rate_hz=rate, stale_seconds=0.)
            if calibration is not None:
                self.state['calibration'] = calibration
                if self.session_purpose == 'calibration' and calibration['ready']:
                    self.state.update(phase='calibration_complete', status='running', probabilities=[],
                        gate={'candidate':None,'progress':0.,'reason':'校准完成 · 保存后等待手动开始解码'})
                    gate.reset_candidate()
                    self.stop_event.set()
                    return
                if not calibration['ready']:
                    self.state.update(phase='calibrating', status='running', probabilities=[],
                        gate={'candidate':None, 'progress':0., 'reason':'脑电校准中 · 暂不输出'})
                elif result is None and not self.state['n_predictions']:
                    self.state.update(phase='warming',
                        gate={'candidate':None, 'progress':0., 'reason':'校准完成 · 正在积累新的解码窗口'})
            if (result is None and self.state["phase"] == "warming"
                    and getattr(gate, 'round_started', None) is None):
                self.state["gate"] = {"candidate": None, "progress": 0., "reason": "正在积累模型窗口"}
            step = max(1, int(decoder.stream.fs/80))
            self.trace.extend([[int(i)/decoder.stream.fs, float(v)] for i, v in zip(ix[::step], raw[0, ::step])])
        if result is not None:
            window = getattr(decoder, 'last_window', None)
            quality = self.state.get('stream_quality', {})
            if quality.get('recovering') and window is not None:
                if window['source_start_index'] < quality.get('first_fresh_source_index', 0):
                    self._vote_tick(elapsed, gate, commands)
                    return
                with self.condition:
                    quality.update(recovering=False, last_clean_window=dict(window))
            self._prediction(result[1], elapsed, ms, gate, predictions, commands,
                             getattr(decoder, 'last_details', None),
                             prediction_window=getattr(decoder, 'last_window', None), input_age_seconds=input_age)

    def _tcp_loop(self, cfg, decoder, gate, predictions, commands, session):
        from .tcp_simulator import TcpEEGSimulator
        from .acquisition import make_recorder, connect, ensure_healthy
        from .data import seal_session
        cfg = copy.deepcopy(cfg)
        cfg["device"].update(host="127.0.0.1", port=0)
        simulator, recorder = TcpEEGSimulator(cfg, port=0, progress=lambda *_: None), None
        origin, last_index, last_arrival = time.monotonic(), -1, time.monotonic()
        rates, events, error, generation = deque(), [], "", None
        try:
            simulator.start()
            cfg["device"]["port"] = simulator.port
            recorder = make_recorder(cfg)
            recorder.start_recording(str(session/"data.h5"), write_chunk_seconds=1)
            connect(recorder, cfg)
            epoch, previous_target = recorder.epoch, object()
            while not self.stop_event.wait(.01):
                self._heartbeat()
                ensure_healthy(recorder, epoch)
                simulator.raise_if_failed()
                elapsed = time.monotonic()-origin
                with self.condition:
                    target = int(elapsed//8) % len(cfg["classes"]) if self.auto_demo and elapsed % 8 < 6.5 else self.target if not self.auto_demo else None
                    self.state["simulation_label"] = target
                if simulator.connected.is_set() and target != previous_target:
                    simulator.send_event(target+1 if target is not None else 9)
                    previous_target = target
                if generation is None:
                    generation = recorder.raw_buffer.generation
                raw, ix, overrun = recorder.raw_buffer.read_after_source_index(last_index, generation=generation)
                if overrun:
                    raise RuntimeError("接收缓存溢出或被重置，已停止；未补点。")
                if not len(ix):
                    stale = time.monotonic()-last_arrival
                    with self.condition:
                        self.state["stale_seconds"] = stale
                    if stale > 1:
                        raise RuntimeError("超过1秒没有新脑电，已停止。")
                    continue
                last_arrival, last_index = time.monotonic(), int(ix[-1])
                self._consume_chunk(raw[:len(cfg["device"]["channels"])], ix, origin, decoder, gate, predictions, commands, rates)
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            try:
                if recorder:
                    recorder.disconnect()
            finally:
                simulator.stop()
                if (session/"data.h5").is_file():
                    seal_session(session, cfg, events, complete=not error, source="synthetic_tcp", error=error)
                    import h5py
                    with h5py.File(session/"data.h5", "r") as h:
                        received = int(h["eeg"].shape[1])
                    with self.condition:
                        self.state.update(samples_received=received, source_seconds=received/decoder.stream.fs)

    def _acp_loop(self, cfg, decoder, gate, predictions, commands, session):
        from .studio.devices import NeusenAdapter
        from .receive_timing import ReceiveTiming
        from .stream_recovery import StreamRecovery, ForwardGapRecovery, packet_discontinuity
        import h5py
        packets, faults, protocol_events = queue.Queue(4096), queue.Queue(), queue.Queue()
        markers, prompted = None, None
        received_origin_ms, latest_source = None, -1
        received_packet_serial = 0
        initialization_end_source, decoder_started = None, False
        timing = ReceiveTiming(cfg['device']['sample_rate'])
        self.live_device_clock = timing.device_clock
        recovery = StreamRecovery(len(cfg['device']['channels']),5)
        forward_gaps = ForwardGapRecovery()
        waiting_for_stream, waiting_reason, waiting_samples = False, '', 0
        def packet(p):
            nonlocal received_origin_ms, latest_source, received_packet_serial
            if p.kind == "eeg":
                if received_origin_ms is None:
                    received_origin_ms = p.device_ms
                latest_source = round((p.device_ms-received_origin_ms)*cfg['device']['sample_rate']/1000)+p.counts.shape[1]-1
                stamp = time.monotonic_ns()
                received_packet_serial += 1
                timing.received(p.counts.shape[1], stamp, p.device_ms)
                try:
                    packets.put_nowait((p, stamp, received_packet_serial))
                except queue.Full:
                    faults.put("输入队列溢出")
            elif markers is not None:
                markers.observe(p)
        def status(kind, detail):
            stamp = time.monotonic_ns()
            protocol_events.put(dict(kind=kind, detail=detail, wall_ns=time.time_ns(),
                                     mono_ns=stamp))
            if detail.get('source') == 'triggerbox_acp':
                return
            if kind in ("down", "configuration_error", "payload_mismatch", "crc_error", "tail_error"):
                if kind in ('down', 'configuration_error'):
                    self.eeg_fault_active.set()
                faults.put(dict(kind=kind, detail=detail, mono_ns=stamp, after_packet_serial=received_packet_serial))
        adapter = NeusenAdapter(packet, status)
        protocol_log = (session/'protocol_events.jsonl').open('x', encoding='utf-8')
        def flush_protocol_events():
            while not protocol_events.empty():
                protocol_log.write(json.dumps(protocol_events.get_nowait(), ensure_ascii=False)+'\n')
            protocol_log.flush()
        def wait_for_stream(reason):
            nonlocal waiting_for_stream, waiting_reason, waiting_samples
            if waiting_for_stream:
                return
            waiting_for_stream, waiting_reason, waiting_samples = True, reason, 0
            protocol_events.put(dict(kind='stream_wait', detail=dict(reason=reason),
                                     wall_ns=time.time_ns(), mono_ns=time.monotonic_ns()))
            with self.condition:
                self.state.update(phase='waiting_stream', probabilities=[])
                self.state['transport_wait'] = dict(waiting=True, reason=reason, loss_confirmed=False)

        def note_loss(event):
            with self.condition:
                quality = self.state['stream_quality']
                quality.update(loss_events=quality['loss_events']+1,
                    missing_samples=quality['missing_samples']+max(0, round(event.get('device_gap_ms', 0)*cfg['device']['sample_rate']/1000)),
                    recovering=True, reason=event['reason'],
                    last_gap_ms=event.get('device_gap_ms', 0),
                    first_fresh_source_index=event['first_new_source_index'])
                self.state.update(stream_recovery=event, phase='recovering_stream', probabilities=[])
                if isinstance(gate, VotingGate):
                    if (self.session_purpose != 'calibration' and not self.state['hand']['busy']
                            and self.fresh_collection_after is None
                            and (prompted is None or prompted.phase == 'decoding')):
                        gate.start_round(time.monotonic()-origin)
                else:
                    gate.reset_candidate()
                if prompted is not None and prompted.current is not None:
                    prompted.current['packet_loss_events'] = prompted.current.get('packet_loss_events', 0)+1
        def check_protocol_faults():
            flush_protocol_events()
            while not faults.empty():
                fault = faults.get_nowait()
                if isinstance(fault, dict) and fault['kind'] in ('crc_error','tail_error','payload_mismatch'):
                    if hasattr(decoder,'recover_stream'):
                        recovery.request(fault['detail'], after_mono_ns=fault['mono_ns'],
                                         after_packet_serial=fault['after_packet_serial'])
                        with self.condition:
                            quality = self.state['stream_quality']
                            quality.update(corrupt_events=quality['corrupt_events']+1,
                                reason='校验异常；根据随后有效包的设备时间与包序号判断连续性')
                        continue
                    calibration = self.state.get('calibration', {})
                    complete = calibration.get('required') and calibration.get('ready')
                    prefix = ('已载入上次校准；' if calibration.get('reused') else '校准已完成；') if complete else ''
                    phase = ('脑电校准期间' if calibration.get('required') and not calibration.get('ready')
                             else '持续解码期间' if self.state['n_predictions'] else '准备解码时')
                    suffix = '完整校准记录保留，可选择使用上次校准。' if complete else ''
                    raise RuntimeError(prefix+phase+'收到的脑电数据包CRC校验失败，已停止。'
                                       '会话和通信异常记录已保存。'+suffix)
                raise RuntimeError(str(fault))
        device = cfg["device"]
        # Same reference, units and preprocessing as the studio-created checkpoint;
        # input samples are adapter raw uV, never the display-filtered preview.
        options = {"host": device["host"], "port": device["port"], "channels": len(device["channels"]),
                   "fs": device["sample_rate"], "points": 5, "gain": 12}
        origin, index, previous = time.monotonic(), 0, None
        source_origin_ms = None
        rates = deque()
        try:
            adapter.connect(options)
            named_stream = cfg.get('session_metadata', {}).get('presentation') in (
                'continuous_four_class_monitor', 'continuous_six_class_observation')
            order = list(range(len(adapter.labels)))
            if cfg.get('session_metadata', {}).get('external_model') and adapter.labels_verified:
                if len(adapter.labels) != len(device['channels']) or set(adapter.labels) != set(device['channels']):
                    raise ValueError('设备完整通道表与模型包不符。')
                order = [adapter.labels.index(ch) for ch in device['channels']]
                named_stream = True
            elif named_stream and adapter.labels_verified:
                from .four_class_online import channel_order
                order = channel_order(adapter.labels)
            elif not adapter.labels_verified or adapter.labels != device["channels"]:
                raise ValueError("设备通道名称或顺序与训练模型不一致；未按编号猜测通道。")
            if not adapter.labels_verified:
                raise ValueError('设备通道标签未经核对，无法对齐模型输入。')
            if adapter.unit != device["unit"] or adapter.settings["fs"] != device["sample_rate"]:
                raise ValueError("设备单位或采样率与模型不一致。")
            if cfg.get('session_metadata', {}).get('calibration_reuse'):
                from .calibration_store import verify_device
                verify_device(cfg['session_metadata']['calibration_reuse']['device'], adapter.describe())
            write_json(session/"device.json", adapter.describe())
            if self.prompted_settings['enabled']:
                from .prompted_control import OnlineMarkers, PromptedControl
                markers = OnlineMarkers(session, adapter, self.prompted_settings, packet, status, timing.device_clock)
            if named_stream:
                write_json(session/'input.json', {'source':'neusen_acp', 'device_channels':adapter.labels,
                    'model_input_order':device['channels'], 'reorder_indices':order, 'unit':adapter.unit,
                    'filter_delay_seconds':decoder.stream.delay_seconds,
                    'input_contract':cfg['session_metadata'].get('input_contract'),
                    'calibration_timing':cfg['session_metadata'].get('calibration_timing'),
                    'stream_adaptation_validated':False,
                    'source_quality':cfg['session_metadata'].get('source_quality'),
                    'online_control_validated':cfg['session_metadata'].get('online_control_validated',False),
                    'historical_reference_equivalence_verified':False, 'labelled_for_training':False})
            with h5py.File(session/"raw_acp.h5", "x") as h:
                h.attrs["unit"], h.attrs["sample_rate"] = adapter.unit, adapter.settings["fs"]
                h.attrs["labelled_for_training"] = False
                h.attrs["complete"] = False
                h.attrs["channels"] = json.dumps(adapter.labels, ensure_ascii=False)
                h.attrs['model_input_indices'] = json.dumps(order)
                h.attrs['contains_discontinuities'] = False
                eeg = h.create_dataset("eeg", (len(adapter.labels), 0), maxshape=(len(adapter.labels), None), dtype="f4", chunks=True)
                counts = h.create_dataset('counts', (len(adapter.labels),0), maxshape=(len(adapter.labels),None), dtype='i4', chunks=True) if named_stream else None
                stamps = h.create_dataset("device_ms", (0,), maxshape=(None,), dtype="f8", chunks=True)
                sequence = h.create_dataset("packet_id", (0,), maxshape=(None,), dtype="u1", chunks=True)
                arrivals = h.create_dataset('arrival_mono_ns', (0,), maxshape=(None,), dtype='i8', chunks=True)
                source_indices = h.create_dataset('source_index',(0,),maxshape=(None,),dtype='i8',chunks=True)
                h.attrs['arrival_clock'] = 'time.monotonic_ns at ACP parser callback'
                pending, pending_samples = [], 0
                batch_samples = max(1, round(adapter.settings['fs']*.04))
                h.attrs['processing_batch_samples'] = batch_samples
                def persist_pending():
                    nonlocal index, pending_samples
                    if not pending:return None
                    fs = adapter.settings['fs']
                    native = np.concatenate([p.counts for p,_ in pending], axis=1)
                    raw = native.astype(np.float32)*adapter.count_scale
                    n = native.shape[1]
                    eeg.resize(index+n, axis=1);eeg[:,index:index+n] = raw
                    if counts is not None:
                        counts.resize(index+n, axis=1);counts[:,index:index+n] = native
                    ix = np.concatenate([round((p.device_ms-source_origin_ms)*fs/1000)+np.arange(p.counts.shape[1],dtype=np.int64) for p,_ in pending])
                    for dataset, values in (
                        (stamps, np.concatenate([p.device_ms+np.arange(p.counts.shape[1])*1000/fs for p,_ in pending])),
                        (sequence, np.concatenate([np.full(p.counts.shape[1],p.packet_id,np.uint8) for p,_ in pending])),
                        (arrivals, np.concatenate([np.full(p.counts.shape[1],stamp,np.int64) for p,stamp in pending])),
                        (source_indices,ix)):
                        dataset.resize(index+n, axis=0);dataset[index:index+n] = values
                    index += n;pending.clear();pending_samples = 0
                    with self.condition:self.state['raw_samples_saved'] = index
                    return raw,ix
                try:
                    if markers is not None:
                        markers.connect()
                        # Preserve setup/self-test EEG, but never decode delayed
                        # startup input or treat its queue as an inference fault.
                        initialization_end_source = latest_source
                        prompted = PromptedControl(session, self.prompted_settings, self.state['classes'],
                            gate, markers, lambda: latest_source, adapter.settings['fs'],
                            self.output_config['mode'] == 'inspire')
                        with self.condition:
                            self.prompted = prompted
                            self.state['prompted'] = prompted.snapshot()
                    with (session/'stream_timing.jsonl').open('x',encoding='utf-8') as timing_log:
                        last_timing_log = -float('inf')
                        while not self.stop_event.is_set():
                            self._heartbeat()
                            check_protocol_faults()
                            if prompted is not None:
                                with self.condition:
                                    calibration = self.state.get('calibration', {})
                                    prompted.tick(not calibration.get('required') or calibration.get('ready', False))
                                    self.state['prompted'] = prompted.snapshot()
                                    if prompted.phase == 'complete':
                                        self.stop_event.set()
                                        break
                            self._vote_tick(time.monotonic()-origin, gate, commands)
                            try:
                                p, arrival, packet_serial = packets.get(timeout=.1)
                            except queue.Empty:
                                snapshot = timing.snapshot(index)
                                stale = snapshot['last_arrival_age_seconds']
                                stale = stale if stale is not None else time.monotonic()-origin
                                with self.condition:self.state['stale_seconds'] = stale
                                if stale > 1:
                                    wait_for_stream('脑电接收暂停超过1秒')
                                continue
                            check_protocol_faults()
                            fs = adapter.settings['fs']
                            if source_origin_ms is None:
                                source_origin_ms = p.device_ms
                                self.prompted_origin_ms = source_origin_ms
                                h.attrs['source_origin_device_ms'] = source_origin_ms
                            if recovery.pending and recovery.accepts(p,previous,fs,arrival,packet_serial):
                                event = recovery.complete()
                                event.update(filters_reset=False,
                                    sample_continuity_confirmed=previous is not None and packet_discontinuity(previous,p,fs) is None)
                                protocol_events.put(dict(kind='crc_followup', detail=event,
                                    wall_ns=time.time_ns(), mono_ns=time.monotonic_ns()))
                                # Only verified sample discontinuity below resets
                                # the decoder; a corrupt prefix may precede a
                                # complete retransmitted copy of the same frame.
                            if previous is not None:
                                gap_detail = packet_discontinuity(previous, p, fs)
                                if gap_detail is not None:
                                    h.attrs['contains_discontinuities'] = True
                                    with self.condition:self.state['gap_count'] += 1
                                    protocol_events.put(dict(kind='stream_gap',detail=gap_detail,
                                        wall_ns=time.time_ns(),mono_ns=time.monotonic_ns()))
                                    reason = f"脑电缺口{gap_detail['device_gap_ms']:g}毫秒；仅剔除包含缺口的窗口"
                                    try:
                                        event = forward_gaps.accept(gap_detail)
                                        if not hasattr(decoder, 'recover_stream'):
                                            raise RuntimeError('当前模型不支持中断后重新积累窗口，已停止并保存。')
                                    except RuntimeError:
                                        # Retain the offending verified frame even when recovery is refused.
                                        pending.append((p,arrival));pending_samples += p.counts.shape[1]
                                        raise
                                    archived = pending_samples
                                    if archived:
                                        before_raw,before_ix = persist_pending()
                                        if initialization_end_source is None or decoder_started:
                                            self._consume_chunk(before_raw[order] if named_stream else before_raw,before_ix,
                                                origin,decoder,gate,predictions,commands,rates,
                                                received_timing=timing,stored_samples=index)
                                        else:
                                            archived = 0  # Hardware self-test data is deliberately archive-only.
                                    first_source = round((p.device_ms-source_origin_ms)*fs/1000)
                                    decoder.recover_stream(first_source, reason)
                                    rates.clear()
                                    event.update(first_new_source_index=first_source,
                                                 processed_before_gap=archived)
                                    protocol_events.put(dict(kind='stream_recovery',detail=event,
                                        wall_ns=time.time_ns(),mono_ns=time.monotonic_ns()))
                                    note_loss(event)
                                    with self.condition:
                                        if hasattr(decoder,'snapshot_calibration'):
                                            self.state['calibration']=decoder.snapshot_calibration()
                            pending.append((p,arrival));pending_samples += p.counts.shape[1]
                            previous = p
                            if pending_samples < batch_samples:continue
                            raw,ix = persist_pending()
                            snapshot = timing.snapshot(index)
                            if waiting_for_stream:
                                protocol_events.put(dict(kind='stream_resumed', detail=dict(
                                    reason=waiting_reason, resumed_source_index=int(ix[0]),
                                    filters_reset=False, archived_without_inference=0, repaired_samples=0),
                                    wall_ns=time.time_ns(), mono_ns=time.monotonic_ns()))
                                waiting_for_stream = False
                                if faults.empty():
                                    self.eeg_fault_active.clear()
                                with self.condition:
                                    self.state['transport_wait'] = dict(waiting=False, loss_confirmed=False)
                            if initialization_end_source is not None and not decoder_started:
                                if int(ix[-1]) <= initialization_end_source:
                                    with self.condition:
                                        self.state.update(samples_received=index, wall_seconds=time.monotonic()-origin)
                                    continue
                                if int(ix[0]) > 0:
                                    decoder.recover_stream(int(ix[0]), 'TriggerBox连接自检结束；此前脑电仅归档，正式窗口重新积累')
                                    protocol_log.write(json.dumps(dict(kind='trigger_initialization_complete',
                                        first_decoded_source_index=int(ix[0]), setup_source_end=initialization_end_source),
                                        ensure_ascii=False)+'\n')
                                    protocol_log.flush()
                                decoder_started = True
                            now = time.monotonic()
                            if now-last_timing_log >= 1:
                                timing_log.write(json.dumps(dict(wall_seconds=now-origin,**timing.snapshot(index)),ensure_ascii=False)+'\n')
                                timing_log.flush();last_timing_log = now
                            self._consume_chunk(raw[order] if named_stream else raw,ix,origin,decoder,gate,
                                                predictions,commands,rates,received_timing=timing,stored_samples=index)
                        h.attrs['complete'] = True
                finally:
                    persist_pending()  # Preserve the partial tail or offending packet without feeding it to the decoder.
                    h.attrs['stream_quality'] = json.dumps(self.state['stream_quality'], ensure_ascii=False)
                    if prompted is not None:
                        # Prevent new motion, settle any in-flight operation before
                        # freezing trial results. No receiver/controller lock held.
                        if self.hand_output is not None:
                            self.hand_output.cancel.set()
                            self.hand_output.close()
                            self.hand_output = None
                        with self.condition:
                            self._finish_simulation(cancelled=True)
                        summary = prompted.finalize(h)
                        with self.condition:
                            self.state['prompted'] = dict(prompted.snapshot(), summary=summary)
                    write_json(session/'stream_timing.json',timing.snapshot(index))
        finally:
            if self.hand_output is not None:
                self.hand_output.cancel.set()
            try:
                try:
                    if markers is not None:
                        markers.close()
                finally:
                    adapter.disconnect()
            finally:
                flush_protocol_events()
                protocol_log.close()


def existing_workbench_url(port):
    if not 1 <= port <= 65535:
        return None
    url = f'http://127.0.0.1:{port}/'
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url+'api/service', timeout=1.5) as response:
            info = json.loads(response.read(16384))
        if (isinstance(info, dict) and info.get('application') == ContinuousController.service_name
                and isinstance(info.get('root'), str) and Path(info['root']).resolve() == ROOT.resolve()):
            return url
    except (OSError, ValueError, TypeError):
        pass
    return None


def open_existing_workbench(url, *, no_browser):
    print(f'解码工作台已经运行：{url} · 使用现有服务', flush=True)
    if not no_browser:
        webbrowser.open(url)


def main(argv=None):
    parser = argparse.ArgumentParser(description="六类脑电持续解码 · 网页模拟或真实机械手输出")
    parser.add_argument("--port", type=int, default=local_settings().get('port', 57474))
    parser.add_argument("--output", type=Path, default=ROOT/"runtime/continuous")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    if local_settings().get('studio_v2'):
        from .studio_v2_runtime import ensure_tool_server
        ensure_tool_server()
    existing = existing_workbench_url(args.port)
    if existing:
        open_existing_workbench(existing, no_browser=args.no_browser)
        return
    import torch
    torch.set_num_threads(2)
    controller = ContinuousController(args.output, registry=args.registry)
    try:
        server = create_online_server(controller, port=args.port, assets=Path(__file__).parent/"continuous_ui")
    except OSError as exc:
        existing = existing_workbench_url(args.port)
        if existing:
            open_existing_workbench(existing, no_browser=args.no_browser)
            return
        if exc.errno == errno.EADDRINUSE or getattr(exc, 'winerror', None) in (10048, 10013):
            print(f'无法启动：本机端口{args.port}被其它程序占用，请先关闭占用程序后重试。', flush=True)
            raise SystemExit(1) from None
        raise
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"六类持续解码：{url} · 启动时不连接设备", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        controller.eeg_probe_done.wait(30)
        if controller.worker:
            controller.worker.join(10)
        if controller.manual_hand.worker:
            controller.manual_hand.worker.join(35)
        server.server_close()


if __name__ == "__main__":
    main()
