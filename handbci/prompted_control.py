"""Prompted online trials: targets never enter the decoder or hand driver.

The controller owns trial transitions. Serial sending/echo confirmation runs on
one independent worker; receiver callbacks never wait for that worker.
"""
from __future__ import annotations

from collections import deque
import copy
import csv
import json
import math
import queue
import random
import secrets
import threading
import time
import uuid

from .config import write_json
from .triggerbox import validate_port


def validate_settings(raw, mode, classes, gate, output):
    if raw is None:
        raw = {}
    if not isinstance(raw, dict) or not isinstance(raw.get('enabled', False), bool):
        raise ValueError('提示采集设置格式无效。')
    if not raw.get('enabled'):
        return {'enabled': False}
    if mode not in ('six','acp') or gate.settings()['mode'] != 'vote':
        raise ValueError('带提示采集使用真实ACP模型和投票判决参数。')
    if not 2 <= len(classes) <= 6 or sorted(c['id'] for c in classes) != list(range(len(classes))):
        raise ValueError('带提示采集支持2至6类完整模型。')
    if len({c.get('hand_action') for c in classes}) != len(classes):
        raise ValueError('各类须映射到不同的动作。')
    loops, seed = raw.get('loops', 4), raw.get('seed')
    if isinstance(loops, bool) or not isinstance(loops, int) or not 1 <= loops <= 100:
        raise ValueError('提示轮数须为1至100的整数；每轮六类各一次。')
    if seed is None:
        seed = secrets.randbelow(2**31)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError('随机种子须为0至2147483647的整数，或留空自动生成。')
    timings = {}
    for key, label, minimum in [('rest_seconds', '回位后休息', 0.),
                                ('prepare_seconds', '预提示准备', .5)]:
        value = raw.get(key, 2.)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not minimum <= value <= 20):
            raise ValueError(f'{label}时间须为{minimum:g}至20秒。')
        timings[key] = float(value)
    if sum(timings.values()) > 20:
        raise ValueError('休息和预提示准备时间之和不能超过20秒。')
    marking = raw.get('marking', 'serial')
    if marking not in ('serial', 'software'):
        raise ValueError('请选择USB TriggerBox或明确标示的软件标记。')
    port = validate_port(raw.get('trigger_port', '')) if marking == 'serial' else ''
    host = str(raw.get('trigger_host', '')).strip() if marking == 'serial' else ''
    if host and (len(host) > 253 or any(c.isspace() for c in host)):
        raise ValueError('TriggerBox无线数据地址无效。')
    if port and output.get('mode') == 'inspire' and port == output.get('port'):
        raise ValueError('TriggerBox和机械手不能使用同一个串口。')
    return dict(enabled=True, loops=loops, seed=seed, marking=marking,
                trigger_port=port, trigger_host=host, trigger_acp_port=4097, **timings)


class OnlineMarkers:
    def __init__(self, folder, adapter, settings, packet_callback, status_callback):
        from .studio.triggers import TriggerManager
        self.folder, self.adapter, self.settings = folder, adapter, settings
        self.cv = threading.Condition(threading.RLock())
        self.echoes, self.events = deque(maxlen=1024), []
        self.requests = queue.Queue()
        self.error, self.closed = '', False
        self.journal = (folder/'events.jsonl').open('x', encoding='utf-8')
        self.trigger = TriggerManager(None, self.wait_echo, lambda: adapter,
                                      on_packet=packet_callback, on_status=status_callback)
        self.worker = None

    def record(self, row):
        with self.cv:
            if self.closed:
                return
            self.events.append(row)
            self.journal.write(json.dumps(row, ensure_ascii=False)+'\n')
            self.journal.flush()

    def observe(self, p):
        if p.kind not in ('event', 'other'):
            return
        row = dict(kind='device_event', event_id=p.code, device_time_ms=p.device_ms,
                   raw_device_time_ms=p.raw_device_ms, wall_time_ns=p.wall_ns,
                   mono_time_ns=p.mono_ns, detail=p.detail, payload_hex=p.payload.hex(),
                   recognized=p.kind == 'event' and p.code is not None)
        with self.cv:
            self.record(row)
            if row['recognized']:
                self.echoes.append(row)
                self.cv.notify_all()

    def wait_echo(self, code, after, timeout, source=None, before=None):
        deadline = time.perf_counter()+timeout
        with self.cv:
            while True:
                for e in self.echoes:
                    upper = before() if callable(before) else before
                    if (e['event_id'] == code and e['mono_time_ns'] >= after
                            and (upper is None or e['mono_time_ns'] < upper)
                            and (source is None or (e.get('detail') or {}).get('source') == source)):
                        return copy.deepcopy(e)
                left = deadline-time.perf_counter()
                if left <= 0:
                    return None
                self.cv.wait(min(left, .1))

    def connect(self):
        try:
            s = self.settings
            self.trigger.connect(s['marking'], s['trigger_port'], wireless_host=s['trigger_host'],
                                 wireless_port=s['trigger_acp_port'])
            if s['marking'] == 'serial':
                row = self.trigger.test()
                self.record(dict(kind='trigger_self_test', **row))
            write_json(self.folder/'triggerbox.json', self.trigger.state())
            self.worker = threading.Thread(target=self._run, daemon=True, name='online-trial-markers')
            self.worker.start()
        except BaseException:
            self.trigger.disconnect()
            self.close()
            raise

    def submit(self, kind, code, **detail):
        row = dict(kind=kind, event_id=code, wall_time_ns=time.time_ns(),
                   mono_time_ns=time.perf_counter_ns(), verification_pending=True,
                   echo_verified=False, **detail)
        self.record(row)
        self.requests.put(row)
        return row

    def _run(self):
        while True:
            row = self.requests.get()
            if row is None:
                self.requests.task_done()
                return
            try:
                result = self.trigger.send(row['event_id'], require_echo=True)
                row.update(result)
                if result.get('echo_verified'):
                    row.update(device_time_ms=result['observed_device_ms'], marking=result['verification'])
                else:
                    row.update(marking='software_receive_clock', verification_pending=False)
            except Exception as exc:
                row.update(error=str(exc), marking='unverified_hardware_marker', verification_pending=False)
                self.error = 'TriggerBox标记失败：'+str(exc)
            finally:
                self.record(dict(row, kind='marker_result', marker_kind=row['kind']))
                self.requests.task_done()

    def close(self):
        if self.worker is not None:
            self.requests.put(None)
            self.worker.join(8)
            if self.worker.is_alive():
                raise RuntimeError('TriggerBox标记线程未退出，记录尚未封存。')
            self.worker = None
        try:
            self.trigger.disconnect()
        finally:
            with self.cv:
                if not self.closed:
                    self.closed = True
                    self.journal.close()


class PromptedControl:
    def __init__(self, folder, settings, classes, gate, markers, latest_source, fs, physical):
        self.folder, self.settings, self.classes, self.gate = folder, settings, classes, gate
        self.markers, self.latest_source, self.fs, self.physical = markers, latest_source, fs, physical
        rng = random.Random(settings['seed'])
        self.plan = []
        for r in range(settings['loops']):
            order = list(range(len(self.classes)))
            rng.shuffle(order)
            self.plan.extend(dict(round=r+1, label=k) for k in order)
        self.session_id, self.phase = uuid.uuid4().hex, 'waiting_calibration'
        self.trials, self.current, self.prompt_id = [], None, None
        self.cue, self.end, self.cutoff = None, None, None
        self.prompt_at, self.first_prediction, self.completed_at = None, None, None
        self.decode_not_before = None
        self.prepare_mark = None
        self.finalized = False
        write_json(folder/'trial_plan.json', dict(settings=settings, classes=classes,
                   trials=self.plan, decision=gate.settings(), physical_output=physical,
                   display_timing=dict(next_target_lead_seconds=settings['prepare_seconds'],
                       post_return_rest_seconds=settings['rest_seconds'],
                       prepare_duration_from_display_ack=True,
                       preview_display_marker='prepare_20_plus_label', task_start_marker='cue_1_plus_label',
                       repeated_start_prompt=False, task_start_at_cooldown_end=False,
                       first_trial_without_preview=False)))

    def snapshot(self):
        c = self.current or {}
        return dict(enabled=True, session_id=self.session_id, prompt_id=self.prompt_id,
                    phase=self.phase, completed=sum(bool(t.get('complete')) for t in self.trials), total=len(self.plan),
                    round=c.get('round'), trial_id=c.get('trial_id'),
                    target_label=c.get('label'), target_action=c.get('target_action', ''),
                    task_text=c.get('task_text', ''), marking=self.settings['marking'],
                    trigger_error=self.markers.error, settings=self.settings)

    def tick(self, calibration_ready):
        if self.markers.error:
            raise RuntimeError(self.markers.error)
        now = time.monotonic()
        if self.phase == 'waiting_calibration' and calibration_ready:
            self._prompt()
        elif self.phase == 'awaiting_display' and now-self.prompt_at > 5:
            raise RuntimeError('目标提示5秒内未收到前台显示回执，已停止。')
        elif self.phase == 'marking_cue' and not self.cue.get('verification_pending'):
            self._enable_decoding()
        elif self.phase == 'cue_hold' and now >= self.decode_not_before:
            if not self.prepare_mark.get('verification_pending'):
                self._send_task_cue()
        elif self.phase == 'executing' and self.completed_at is not None:
            if self.current.get('execution_status') in ('failed', 'cancelled'):
                return
            display_at = self.completed_at+self.settings['rest_seconds']
            if not self.end.get('verification_pending') and now >= display_at:
                self._finish()

    def _enable_decoding(self):
        self.phase = 'decoding'
        self.current.update(decoding_enabled_wall_ns=time.time_ns(), decoding_enabled_mono_ns=time.perf_counter_ns(),
                            decoding_enabled_source_index=self.latest_source()+1)
        self.markers.record(dict(kind='decoding_enabled', trial_id=self.current['trial_id'],
            label=self.current['label'], wall_time_ns=self.current['decoding_enabled_wall_ns'],
            mono_time_ns=self.current['decoding_enabled_mono_ns'],
            estimated_source_index=self.current['decoding_enabled_source_index']))
        self.gate.reset_candidate()
        self.gate.latched = None
        self.gate.neutral_since = None

    def _prompt(self):
        self.decode_not_before = None
        if len(self.trials) == len(self.plan):
            self.phase, self.prompt_id = 'complete', None
            return
        p = self.plan[len(self.trials)]
        c = next(c for c in self.classes if c['id'] == p['label'])
        self.current = dict(p, trial_id=len(self.trials)+1, target_action=c['hand_action'],
                            target_action_name=c['action_name'], task_text=c.get('task_text', c['name']),
                            physical_output=self.physical, execution_status='pending')
        self.phase, self.prompt_id, self.prompt_at = 'awaiting_display', uuid.uuid4().hex, time.monotonic()
        self.current.update(cue_display_lead_seconds=self.settings['prepare_seconds'],
                            decoding_not_before_mono=None,
                            cue_timing='display_ack_plus_prepare')
        self.cue = self.end = self.first_prediction = self.completed_at = self.cutoff = None
        self.prepare_mark = None

    def acknowledge(self, request):
        if (self.phase != 'awaiting_display' or request.get('session_id') != self.session_id
                or request.get('prompt_id') != self.prompt_id or request.get('visible') is not True):
            raise ValueError('目标提示已失效或页面不在前台。')
        if time.monotonic()-self.prompt_at > 5:
            raise ValueError('目标提示显示回执已超时。')
        audio = request.get('audio')
        if audio is not None:
            names = dict(thumb='拇指',index='食指',middle='中指',ring='无名指',little='小指',pinch='OK手势')
            duration = audio.get('duration_seconds') if isinstance(audio,dict) else None
            if (not isinstance(audio,dict) or audio.get('enabled') is not True
                    or audio.get('text') != names[self.current['target_action']]
                    or audio.get('engine') != 'local_wav_webaudio'
                    or type(duration) not in (int,float) or not math.isfinite(duration) or not 0 < duration <= 5):
                raise ValueError('目标动作语音回执不匹配。')
            self.current['prompt_audio'] = copy.deepcopy(audio)
        display_source_index = self.latest_source()+1
        self.current.update(display_ack_wall_ns=time.time_ns(), display_ack_mono_ns=time.perf_counter_ns(),
                            client_wall_ms=request.get('client_wall_ms'), display_source_index=display_source_index)
        # Every trial, including the first, gets the full visible preparation
        # duration. A delayed display ACK must never shorten that interval.
        self.decode_not_before = time.monotonic()+self.settings['prepare_seconds']
        self.current.update(preview_lead_at_ack_seconds=self.settings['prepare_seconds'],
                            decoding_not_before_mono=self.decode_not_before)
        self.prepare_mark = self.markers.submit('prepare', 20+self.current['label'],
            trial_id=self.current['trial_id'], label=self.current['label'],
            round=self.current['round'], target_action=self.current['target_action'],
            estimated_source_index=display_source_index, display_ack_wall_ns=self.current['display_ack_wall_ns'])
        self.phase = 'cue_hold'
        return {'ok': True}

    def _send_task_cue(self):
        self.cutoff = self.latest_source()+1
        self.current.update(source_cutoff=self.cutoff, task_start_wall_ns=time.time_ns(),
                            task_start_mono_ns=time.perf_counter_ns())
        self.cue = self.markers.submit('cue', self.current['label']+1,
            trial_id=self.current['trial_id'], label=self.current['label'],
            round=self.current['round'], target_action=self.current['target_action'],
            estimated_source_index=self.cutoff, display_ack_wall_ns=self.current['display_ack_wall_ns'])
        self.phase = 'marking_cue'

    def eligible(self, window):
        if self.phase != 'decoding' or window is None:
            return False
        # Same full raw support test as continuous decoding, including FIR history
        # and every window used by the improved models. No pre-cue votes leak in.
        if self.cue.get('echo_verified'):
            boundary = self.cue.get('source_cutoff')
            if boundary is not None:
                self.cutoff = max(self.cutoff, boundary)
        return window['source_start_index'] >= self.cutoff

    def before_prediction(self, window, origin_ms):
        if self.cue and self.cue.get('echo_verified') and origin_ms is not None:
            self.cue['source_cutoff'] = round((self.cue['observed_device_ms']-origin_ms)*self.fs/1000)
        return self.eligible(window)

    def decision(self, event):
        if self.phase != 'decoding':
            raise RuntimeError('当前试次不允许再次下发动作。')
        t = self.current
        event.update(trial_id=t['trial_id'], round=t['round'], target_label=t['label'],
                     target_hand_action=t['target_action'], target_task=t['task_text'])
        t.update(command_sequence=event['sequence'], predicted_label=event['model_class_id'],
                 predicted_action=event['hand_action'], prediction_matches_target=event['model_class_id'] == t['label'],
                 decision_wall_ns=time.time_ns(), decision_mono_ns=time.perf_counter_ns(),
                 decision_reason=event.get('decision_reason'), execution_status=event['execution_status'])
        self.end = self.markers.submit('task_end', 9, trial_id=t['trial_id'], label=t['label'],
                                       estimated_source_index=self.latest_source()+1)
        self.markers.record(dict(kind='decision', **copy.deepcopy(t)))
        self.phase = 'executing'

    def no_output(self):
        t = self.current
        t.update(execution_status='no_output', decision_wall_ns=time.time_ns())
        self.end = self.markers.submit('task_end', 9, trial_id=t['trial_id'], label=t['label'],
                                       estimated_source_index=self.latest_source()+1)
        self.phase, self.completed_at = 'executing', time.monotonic()

    def motion(self, event):
        if self.finalized or self.current is None or event.get('sequence') != self.current.get('command_sequence'):
            return
        self.current.update(execution_status=event['execution_status'], motion_result=copy.deepcopy(event),
                            motion_completed_wall_ns=time.time_ns())
        self.completed_at = time.monotonic()

    def _finish(self):
        self.current['complete'] = True
        self.trials.append(self.current)
        self.markers.record(dict(kind='trial_result', **copy.deepcopy(self.current)))
        write_json(self.folder/'trials.partial.json', self.trials)
        self.current = None
        self.gate.reset_candidate()
        self._prompt()

    def finalize(self, h):
        import h5py
        import numpy as np
        if self.finalized:
            return
        if self.current is not None:
            t = copy.deepcopy(self.current)
            t.update(complete=False, interruption='会话停止或故障，试次未完成')
            self.trials.append(t)
            self.markers.record(dict(kind='trial_interrupted', **t))
        self.finalized = True
        self.markers.close()
        stamps, n = h['device_ms'][:], len(h['device_ms'])
        trigger = np.zeros(n, dtype=np.uint32)
        fs = self.fs
        events = copy.deepcopy(self.markers.events)
        for e in events:
            e['sample_index'] = None
            target = e.get('device_time_ms')
            if target is not None and n and stamps[0] <= target <= stamps[-1]:
                i = int(np.searchsorted(stamps, target))
                if i and stamps[i]-stamps[i-1] > 1000/fs+.501 and stamps[i-1] < target < stamps[i]:
                    e['alignment'] = 'device_clock_gap'
                    continue
                e.update(sample_index=i, alignment='device_clock_lower_bound')
                if e['kind'] == 'device_event' and e.get('recognized'):
                    if trigger[i]:
                        e['trigger_collision'] = True
                    else:
                        trigger[i] = e['event_id']
            elif self.settings['marking'] == 'software' and e.get('kind') in ('prepare', 'cue', 'task_end'):
                i = e.get('estimated_source_index')
                if i is not None and n:
                    k = int(np.searchsorted(h['source_index'][:], i))
                    if k < n:
                        e.update(sample_index=k, alignment='estimated_host_receive_clock')
                        trigger[k] = e['event_id']
            if e['sample_index'] is None:
                e.setdefault('alignment', 'unavailable')
        # Final exports replace the append-only crash journal only after all workers stop.
        with (self.folder/'events.jsonl').open('w', encoding='utf-8') as f:
            for e in events:
                f.write(json.dumps(e, ensure_ascii=False)+'\n')
        for t in self.trials:
            for kind, prefix in [('prepare', 'preview'), ('cue', 'start'), ('task_end', 'end')]:
                e = next((e for e in events if e['kind'] == kind and e.get('trial_id') == t['trial_id']), {})
                t[prefix+'_sample'] = e.get('sample_index')
                t[prefix+'_seconds'] = (float(stamps[e['sample_index']])-float(stamps[0]))/1000 if e.get('sample_index') is not None else None
                t[prefix+'_echo_verified'] = e.get('echo_verified', False)
        for name, rows in [('events', events), ('trials', self.trials)]:
            columns = list(dict.fromkeys(k for row in rows for k in row))
            with (self.folder/(name+'.csv')).open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                                  for k, v in row.items()} for row in rows)
            h.create_dataset(name+'_json', data=[json.dumps(e, ensure_ascii=False) for e in rows],
                             dtype=h5py.string_dtype('utf-8'))
        h.create_dataset('trigger', data=trigger)
        def aligned_hardware(t):
            return bool(t.get('complete') and t.get('start_echo_verified') and t.get('end_echo_verified')
                        and t.get('start_sample') is not None and t.get('end_sample') is not None
                        and t['start_sample'] < t['end_sample'])
        verified = sum(aligned_hardware(t) for t in self.trials)
        summary = dict(complete=self.phase == 'complete', planned=len(self.plan),
                       completed=sum(bool(t.get('complete')) for t in self.trials),
                       hardware_verified_trials=verified, physical_output=self.physical,
                       marking=self.settings['marking'], signal_seconds=n/fs,
                       physical_onset_verified=False, device_timestamp_semantics_verified=False,
                       contains_discontinuities=bool(h.attrs.get('contains_discontinuities', False)),
                       target_counts={str(k):sum(bool(t.get('complete')) and t['label'] == k for t in self.trials) for k in range(len(self.classes))},
                       physical_motion_counts={str(k):sum(t['label'] == k and self.physical and t.get('execution_status') == 'completed' for t in self.trials) for k in range(len(self.classes))},
                       submission_checks=dict(duration_at_least_120s=n/fs >= 120,
                           every_target_at_least_four=all(sum(t.get('complete') and t['label'] == k for t in self.trials) >= 4 for k in range(len(self.classes)))))
        write_json(self.folder/'prompted_report.json', summary)
        write_json(self.folder/'trials.json', self.trials)
        h.attrs['prompted_control_json'] = json.dumps(summary, ensure_ascii=False)
        h.attrs['has_target_labels'] = True
        h.attrs['physical_event_time_verified'] = False
        mapping = '\n'.join(f"| {c['id']+1} | {c.get('task_text', c['name'])} | {c['action_name']} |" for c in self.classes)
        (self.folder/'数据说明.md').write_text(
            '# 带提示在线机械手控制数据\n\n'
            'raw_acp.h5保存物理通道顺序的原始counts、µV、设备时间和trigger。'
            '采样率和通道名称见H5属性，模型及输出配置见model_selection.json。\n\n'
            '| 起始事件码 | 目标运动想象 | 目标机械手动作 |\n|---|---|---|\n'+mapping+
            '\n\n9=想象任务结束（判决产生或没有可用输出）；机械手完成与回位另见hand_events.jsonl。'
            'cue标签是预先安排的目标，预测及实际下发动作另列，错误试次保留。'
            f'动作回位后先休息{self.settings["rest_seconds"]:g}秒，再预提示{self.settings["prepare_seconds"]:g}秒；'
            '回位后间隔包含这两段，不额外叠加。首试次也完整预提示；准备时长从前台显示回执起计；'
            '预显示回执时发送prepare（20加类别编号），间隔结束发送正式cue（1加类别编号），画面保持不变且不重复提醒；'
            '首试次没有前一动作的回位间隔，显示后直接发送cue。decoding_enabled另记开始标记核验后启用判决的时间。'
            'trials.csv的样点从0计数、秒数相对记录起点；events.csv保留设备时间及主机时间。'
            'trigger源自已识别回传事件；软件模式明确使用接收时钟估计。ACK不等于硬件回传。'
            '设备事件时间语义与屏幕物理呈现延迟未独立核验。'
            '模拟机械手不是真实执行证据；驱动报告到位不等于视频或评审确认动作正确。'
            '完成状态、时长及每类次数见prompted_report.json；不自动判定比赛合格。\n', encoding='utf-8')
        return summary
