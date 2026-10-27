"""Optional Inspire output through the supplied driver; no I/O at import time."""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import re
import threading
import time

DRIVER_PATH = Path(__file__).resolve().parent/'vendor/inspire_serial.py'
CHANNELS = ("小指", "无名指", "中指", "食指", "拇指弯曲", "拇指旋转")
ACTION_CHANNELS = {"thumb": (4,), "index": (3,), "middle": (2,),
                   "ring": (1,), "little": (0,), "pinch": (3, 4, 5)}
ACTION_LABELS = dict(zip(ACTION_CHANNELS, ("拇指单独动作", "食指单独动作", "中指单独动作",
                                          "无名指单独动作", "小指单独动作", "OK 手势")))
FAULT_NAMES = ('堵转', '过温', '过流', '电机异常', '通信异常')


def motion_feedback(result):
    """Describe verified joint readbacks, without treating dispatch as movement."""
    physical = result.get('physical_action') is True
    tolerance = result.get('tolerance')
    stages = {s.get('stage'): s.get('angles') for s in result.get('stages', []) if isinstance(s, dict)}
    targets, target_angles = result.get('targets'), stages.get('target')
    return_targets, return_angles = result.get('return_targets'), stages.get('return_start')

    def positions_match(requested, actual):
        if (not physical or type(tolerance) is not int or not 1 <= tolerance <= 100
                or not isinstance(requested, list) or len(requested) != 6
                or not isinstance(actual, list) or len(actual) != 6):
            return False
        if (any(type(v) is not int or not -1 <= v <= 1000 for v in requested)
                or all(v == -1 for v in requested)
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1000
                       for v in actual)):
            return False
        return all(v == -1 or abs(v - a) <= tolerance for v, a in zip(requested, actual))

    target_confirmed = positions_match(targets, target_angles)
    return_confirmed = positions_match(return_targets, return_angles)
    actual_action = None
    if target_confirmed:
        active = tuple(i for i, value in enumerate(targets) if value != -1)
        for key, channels in ACTION_CHANNELS.items():
            if active == channels or key == 'pinch' and active == (3, 4):
                actual_action = key
                break
    return_required = result.get('recovery') == 'start'
    verified = (result.get('execution_status') == 'completed' and target_confirmed
                and (not return_required or return_confirmed))
    return dict(source='joint_position_readback' if physical else 'none',
                actual_hand_action=actual_action, action_name=ACTION_LABELS.get(actual_action),
                target_confirmed=target_confirmed, return_confirmed=return_confirmed,
                return_required=return_required, verified=verified, tolerance=tolerance,
                targets=copy.deepcopy(targets), target_angles=copy.deepcopy(target_angles),
                return_targets=copy.deepcopy(return_targets), return_angles=copy.deepcopy(return_angles))


def health_fault_message(status, max_temperature):
    issues = []
    for channel, code in zip(CHANNELS, status['errors']):
        if code:
            names = [name for bit, name in enumerate(FAULT_NAMES) if code & (1 << bit)]
            if code & ~31:names.append('未知故障位')
            issues.append(f"{channel}故障码{code}（{'、'.join(names)}）")
    for channel, temperature in zip(CHANNELS, status['temperatures']):
        if temperature >= max_temperature:
            issues.append(f'{channel}{temperature}℃，达到温度阈值{max_temperature}℃')
    return '；'.join(issues)


def calibration_defaults():
    # Unknown positions stay empty. In particular, no guessed thumb rotation/OK pose.
    return {"baseline": [None]*6, "action_profiles": {
        key: {"targets": [None]*6, "speed": 100, "force": 300, "timeout": 8.,
              "tolerance": 20, "max_temperature": 55, "hold_seconds": 0., "recovery": "start"}
        for key in ACTION_CHANNELS}}


def calibration_options(raw, *, physical):
    baseline = raw.get("baseline", [None]*6)
    profiles = raw.get("action_profiles")
    if not isinstance(baseline, list) or len(baseline) != 6 or not isinstance(profiles, dict):
        raise ValueError("请配置六路基准位置和六个动作姿态。")
    def positions(values):
        if not isinstance(values, list) or len(values) != 6:
            raise ValueError("每个姿态须填写六路位置。")
        for value in values:
            if value is None and not physical:
                continue
            integer(value, "位置刻度", -1, 1000)
        return list(values)
    baseline = positions(baseline)
    result = {}
    for key, channels in ACTION_CHANNELS.items():
        profile = profiles.get(key)
        if not isinstance(profile, dict):
            raise ValueError(f"缺少{ACTION_LABELS[key]}的姿态。")
        targets = positions(profile.get("targets"))
        recovery = profile.get("recovery", "start")
        if recovery != "start":
            raise ValueError("六个动作均须完成一次往返：目标到位后返回本次动作前的位置；请刷新旧配置。")
        if physical:
            if any(baseline[i] < 0 for i in range(5)):
                raise ValueError("真实输出须填写五路弯曲通道的基准，拇指旋转可用-1跳过。")
            if any(targets[i] < 0 for i in channels):
                raise ValueError(f"{ACTION_LABELS[key]}的运动通道必须填写位置；OK须同时配置食指、拇指弯曲和旋转。")
            if any(targets[i] != baseline[i] for i in range(6) if i not in channels):
                raise ValueError(f"{ACTION_LABELS[key]}的其它通道须等于基准位置。")
            if not any(targets[i] != baseline[i] for i in channels):
                raise ValueError(f"{ACTION_LABELS[key]}与基准相同，未配置运动。")
            if key == "pinch" and baseline[5] < 0:
                raise ValueError("OK须校准拇指旋转的基准和目标位置；已经对准时可保持同一位置。")
            if key == "pinch" and any(targets[i] == baseline[i] for i in (3, 4)):
                raise ValueError("OK的食指和拇指弯曲目标均须偏离伸直基准。")
        result[key] = dict(targets=targets, recovery=recovery, **motion_options(profile))
    return {"baseline": baseline, "action_profiles": result,
            "calibration_status": "user_configured_not_hardware_verified" if physical else "simulation_only"}


def load_hand_class():
    spec = importlib.util.spec_from_file_location("handbci_supplied_inspire", DRIVER_PATH)
    if spec is None or spec.loader is None:
        raise ValueError("机械手接口文件不可用。")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.InspireHand


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name}须为{low}至{high}的整数。")
    return value


def validate_output(raw, n_classes, source_mode, classes=None):
    if not isinstance(raw, dict):
        raise ValueError("机械手输出配置须为对象。")
    mode = raw.get("mode", "simulation")
    if mode == "simulation":
        return {"mode": "simulation", **(calibration_options(raw, physical=False)
                if "action_profiles" in raw else {})}
    if mode != "inspire":
        raise ValueError("请选择模拟输出或真实机械手。")
    if source_mode not in ("acp", "six"):
        raise ValueError("交互演示和人工脑电仅允许模拟输出；真实机械手须配合真实ACP模型。")
    result = connection_options(raw)
    if "action_profiles" in raw:
        calibration = calibration_options(raw, physical=True)
        if classes is None or len(classes) != n_classes or any(c.get("hand_action") not in ACTION_CHANNELS for c in classes):
            raise ValueError("每个模型类别须明确映射到机械手动作。")
        result.update(mode="inspire", **motion_options(raw), **calibration,
                      class_actions=[c["hand_action"] for c in classes])
        return result
    profiles = raw.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != n_classes:
        raise ValueError("请为每个模型类别配置一组六路目标位置。")
    for values in profiles:
        if not isinstance(values, list) or len(values) != 6:
            raise ValueError("每个动作须包含六个目标值，按小指至拇指旋转顺序填写。")
        for v in values:
            integer(v, "目标位置", -1, 1000)
        if all(v == -1 for v in values):
            raise ValueError("每个动作至少指定一个通道，不能全部跳过。")
    result.update(mode="inspire", profiles=copy.deepcopy(profiles), recovery="start", **motion_options(raw))
    return result


def motion_options(raw):
    result = dict(speed=integer(raw.get("speed", 100), "速度", 1, 1000),
                  force=integer(raw.get("force", 300), "力阈值", 1, 3000),
                  tolerance=integer(raw.get("tolerance", 20), "到位容差", 1, 100),
                  max_temperature=integer(raw.get("max_temperature", 55), "温度阈值", 1, 100))
    timeout = raw.get("timeout", 8.)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not .5 <= timeout <= 30:
        raise ValueError("动作超时须为0.5至30秒。")
    result["timeout"] = float(timeout)
    hold_seconds = raw.get("hold_seconds", 0.)
    if isinstance(hold_seconds, bool) or not isinstance(hold_seconds, (int, float)) or not math.isfinite(hold_seconds) or not 0 <= hold_seconds <= 10:
        raise ValueError("目标停留须为0至10秒。")
    result['hold_seconds'] = float(hold_seconds)
    return result


def connection_options(raw):
    port = raw.get("port", "")
    if not isinstance(port, str) or not re.fullmatch(r"COM[1-9][0-9]*", port.strip(), re.I):
        raise ValueError("请选择或填写机械手串口，例如COM3。")
    return {"port": port.strip().upper(),
            "baudrate": integer(raw.get("baudrate", 115200), "波特率", 1, 4_000_000),
            "hand_id": integer(raw.get("hand_id", 1), "机械手ID", 1, 16383)}


def idle_state(mode="simulation"):
    return {"mode": mode, "connected": False, "busy": False,
            "phase": "simulation" if mode == "simulation" else "disconnected",
            "error": "", "angles": [], "last_duration_seconds": None}


class HandOutput:
    """One active motion, no queued predictions, no automatic write retry."""
    def __init__(self, config, on_event, *, hand_factory=None):
        self.config = copy.deepcopy(config)
        self.on_event = on_event
        self.factory = hand_factory
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.thread, self.hand = None, None
        self.state = idle_state("inspire")

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state)

    def _emit(self, event=None):
        self.on_event(self.snapshot(), event)

    def _record_health(self, status, max_temperature, stage, result=None):
        health = dict(stage=stage, wall_ns=time.time_ns(), max_temperature=max_temperature,
                      status=copy.deepcopy(status))
        with self.lock:
            self.state['health'] = health
        if result is not None:
            result.setdefault('health_checks', []).append(health)
        return health_fault_message(status, max_temperature)

    def connect(self):
        factory = self.factory or load_hand_class()
        self.hand = factory(**{k:self.config[k] for k in ("port", "baudrate", "hand_id")})
        try:
            self.hand.connect()
            status = self.hand.get_status()
            fault = self._record_health(status, self.config['max_temperature'], 'connect')
            if fault:
                raise RuntimeError(f'机械手连接检查未通过：{fault}；未启用动作输出。')
            with self.lock:
                self.state.update(connected=True, phase="ready", angles=status["angles"],
                                  read_retry_count=status.get("read_retry_count", 0),
                                  write_ack_recovered_count=status.get("write_ack_recovered_count", 0))
            self._emit()
        except Exception as exc:
            self.hand.disconnect()
            with self.lock:
                self.state.update(connected=False, phase='failed', error=str(exc))
            self._emit()
            raise

    def submit(self, event):
        with self.lock:
            if self.state["busy"] or self.cancel.is_set() or self.state["error"] or not self.state["connected"]:
                return False
            self.state.update(busy=True, phase="executing")
            self.thread = threading.Thread(target=self._execute, args=(copy.deepcopy(event),), daemon=True,
                                           name="inspire-motion")
            self.thread.start()
        return True

    def initialize(self):
        if 'baseline' not in self.config:
            return None
        targets = list(self.config['baseline'][:5])+[-1]
        result = dict(kind='initialization_result', targets=targets, stages=[],
                      execution_status='completed', physical_action=False,
                      target_dispatch_attempted=False, error='')
        started = time.monotonic()
        with self.lock:
            if not self.state['connected'] or self.state['busy']:
                raise RuntimeError('机械手尚未就绪，未执行初始归位。')
            self.state.update(busy=True, phase='initializing')
        self._emit()
        try:
            if self.cancel.is_set():
                raise RuntimeError('初始归位已取消，未发送伸直目标。')
            status = self.hand.get_status()
            fault = self._record_health(status, self.config['max_temperature'], 'before_initialization', result)
            if fault:
                raise RuntimeError(f'初始归位检查未通过：{fault}；未发送伸直目标。')
            before = status['angles']
            if not isinstance(before, list) or len(before) != 6:
                raise RuntimeError('无法读取初始六路位置，未发送伸直目标。')
            for value in before:integer(value, '初始位置', 0, 1000)
            result['before'] = list(before)
            needs_motion = any(abs(before[i]-targets[i]) > self.config['tolerance'] for i in range(5))
            if needs_motion:
                result.update(physical_action=True, target_dispatch_attempted=True)
                actual = self.hand.move(targets, cancel_event=self.cancel,
                    **{k:self.config[k] for k in ('speed','force','timeout','tolerance','max_temperature')})
                status = self.hand.get_status()
                fault = self._record_health(status, self.config['max_temperature'], 'after_initialization', result)
                if fault:raise RuntimeError(f'初始归位后检查未通过：{fault}')
                actual = status['angles']
            else:
                actual = before
            if any(abs(actual[i]-targets[i]) > self.config['tolerance'] for i in range(5)):
                raise RuntimeError(f'五指伸直尚未到位：target={targets}, actual={actual}')
            result.update(angles=list(actual), skipped=not needs_motion)
            result['stages'].append(dict(stage='straight_baseline', angles=list(actual)))
            if self.cancel.is_set():result['execution_status']='cancelled'
        except Exception as exc:
            if getattr(exc,'device_status',None):
                self._record_health(exc.device_status,self.config['max_temperature'],'initialization_fault',result)
            cancelled = self.cancel.is_set() and not getattr(exc,'hold_failed',False)
            result.update(execution_status='cancelled' if cancelled else 'failed',error=f'初始归位失败：{exc}')
        result.update(duration_seconds=time.monotonic()-started,completed_mono=time.monotonic())
        with self.lock:
            self.state.update(busy=False,phase='ready' if result['execution_status']=='completed' else result['execution_status'],
                              error=result['error'],angles=result.get('angles',self.state['angles']))
            self.state['initialization'] = copy.deepcopy(result)
        self._emit(result)
        if result['execution_status']=='failed':raise RuntimeError(result['error'])
        return result['execution_status']=='completed'

    def _execute(self, event):
        started = time.monotonic()
        key = self.config.get("class_actions", [])[event["class_id"]] if "class_actions" in self.config else None
        profile = self.config["action_profiles"][key] if key else self.config
        configured = profile["targets"] if key else self.config["profiles"][event["class_id"]]
        active = ACTION_CHANNELS[key] if key else tuple(i for i,v in enumerate(configured) if v != -1)
        if key == 'pinch' and configured[5] == self.config['baseline'][5]:
            active = (3, 4)
        values = [v if i in active else -1 for i,v in enumerate(configured)]
        recovery = "start" if key else profile.get("recovery", "hold")
        result = {"kind": "motion_result", "sequence": event["sequence"], "targets": values,
                  "configured_targets": configured, "hand_action": key, "recovery": recovery, "stages": [],
                  "tolerance": profile['tolerance'],
                  "target_dispatch_attempted": False,
                  "physical_action": True, "execution_status": "completed", "error": ""}
        try:
            return_targets = None
            if recovery == "start":
                if self.cancel.is_set():
                    raise RuntimeError("动作已取消，未发送目标。")
                status = self.hand.get_status()
                fault = self._record_health(status, profile['max_temperature'], 'before_target', result)
                before = status["angles"]
                if not isinstance(before, list) or len(before) != 6:
                    raise RuntimeError("无法读取动作前六路位置，未发送目标。")
                for position in before:
                    integer(position, "动作前位置", 0, 1000)
                if fault:
                    raise RuntimeError(f'动作前检查未通过：{fault}；未发送目标。')
                return_targets = [before[i] if i in active else -1 for i in range(6)]
                result.update(before=list(before), return_targets=return_targets)
            result['target_dispatch_attempted'] = True
            angles = self.hand.move(values, cancel_event=self.cancel,
                **{k:profile[k] for k in ("speed", "force", "timeout", "tolerance", "max_temperature")})
            result["stages"].append(dict(stage="target", angles=angles))
            hold_seconds = profile.get('hold_seconds', 0.)
            if hold_seconds > 0 and not self.cancel.is_set():
                with self.lock:
                    self.state.update(phase='holding', angles=angles)
                self._emit()
                deadline = time.monotonic()+hold_seconds
                try:
                    while time.monotonic() < deadline:
                        if self.cancel.wait(min(.2, max(0., deadline-time.monotonic()))):
                            raise RuntimeError('目标停留已取消，未继续返回。')
                        status = self.hand.get_status()
                        fault = self._record_health(status, profile['max_temperature'], 'holding', result)
                        if fault:
                            raise RuntimeError(f'目标停留期间检查未通过：{fault}')
                except Exception as exc:
                    try:
                        self.hand.hold()
                    except Exception as hold_error:
                        failure = RuntimeError(f'{exc}; 保持当前位置也失败：{hold_error}')
                        failure.hold_failed = True
                        raise failure from exc
                    raise
                result['stages'].append(dict(stage='hold', seconds=hold_seconds))
            if return_targets is not None and not self.cancel.is_set():
                with self.lock:
                    self.state["phase"] = "recovering"
                    self.state["angles"] = angles
                self._emit()
                angles = self.hand.move(return_targets, cancel_event=self.cancel,
                    **{k:profile[k] for k in ("speed", "force", "timeout", "tolerance", "max_temperature")})
                result["stages"].append(dict(stage="return_start", angles=angles))
            # Even a late return after cancellation must not report successful completion.
            result.update(angles=angles, execution_status="cancelled" if self.cancel.is_set() else "completed")
        except Exception as exc:
            if getattr(exc, 'device_status', None):
                self._record_health(exc.device_status, profile['max_temperature'], 'motion_fault', result)
            cancelled = self.cancel.is_set() and not getattr(exc, "hold_failed", False)
            result.update(execution_status="cancelled" if cancelled else "failed", error=str(exc))
        result.update(duration_seconds=time.monotonic()-started, completed_mono=time.monotonic())
        result["read_retry_count"] = getattr(self.hand, "read_retry_count", 0)
        result["write_ack_recovered_count"] = getattr(self.hand, "write_ack_recovered_count", 0)
        result["communication_events"] = copy.deepcopy(getattr(self.hand, "communication_events", []))
        result['feedback'] = motion_feedback(result)
        with self.lock:
            self.state.update(busy=False, phase=result["execution_status"], error=result["error"],
                              last_duration_seconds=result["duration_seconds"],
                              read_retry_count=result["read_retry_count"],
                              write_ack_recovered_count=result["write_ack_recovered_count"],
                              angles=result.get("angles", self.state["angles"]))
        self._emit(result)

    def close(self):
        self.cancel.set()
        if self.thread is not None:
            self.thread.join()
        if self.hand is not None:
            try:
                self.hand.disconnect()
            finally:
                with self.lock:
                    self.state.update(connected=False, busy=False)
                    if self.state["phase"] == "ready":
                        self.state["phase"] = "disconnected"
                self._emit()
