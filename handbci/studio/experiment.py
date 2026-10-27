"""Monotonic deadlines; preparation and imagery share one static subject view."""
from __future__ import annotations
import math
import random
import threading
import time

DEFAULT_PROTOCOL = {'tasks': [{'text': '左手', 'action': 'left_hand'}, {'text': '右手', 'action': 'right_hand'}],
    'loops': 5, 'prepare_seconds': 2.0, 'task_seconds': 4.0, 'rest_seconds': 3.0,
    'block_rest_seconds': 10.0, 'initial_rest_seconds': 3.0, 'tail_seconds': 1.0,
    'randomize': True, 'seed': 42, 'sound': True, 'condition': '常规', 'condition_notes': ''}


def validate_protocol(value):
    p = {**DEFAULT_PROTOCOL, **value}
    condition, notes = p['condition'], p['condition_notes']
    if not isinstance(condition, str) or not 1 <= len(condition.strip()) <= 80 or any(ord(c) < 32 for c in condition):
        raise ValueError('采集情况须为1至80个字符，不能包含换行或控制字符。')
    if not isinstance(notes, str) or len(notes) > 1000 or any(ord(c) < 32 and c not in '\r\n\t' for c in notes):
        raise ValueError('采集情况备注须为不超过1000字的文字。')
    p['condition'], p['condition_notes'] = condition.strip(), notes.strip()
    if not isinstance(p['tasks'], list) or not 2 <= len(p['tasks']) <= 6:
        raise ValueError('任务类别须为2至6类。')
    p['tasks'] = [{'text': str(t['text']).strip(), 'action': t.get('action', 'custom')} for t in p['tasks']]
    if any(not 1 <= len(t['text']) <= 40 for t in p['tasks']) or len({t['text'] for t in p['tasks']}) != len(p['tasks']):
        raise ValueError('任务名称须不同，且为1至40字。')
    if any(t['action'] not in ('left_hand','right_hand','custom','thumb','index','middle','ring','little','pinch') for t in p['tasks']):
        raise ValueError('任务动作标识无效。')
    if isinstance(p['loops'], bool) or int(p['loops']) != float(p['loops']) or not 1 <= int(p['loops']) <= 1000:
        raise ValueError('轮数须为1至1000的整数。')
    p['loops'], p['seed'] = int(p['loops']), int(p['seed'])
    for k in ('prepare_seconds','task_seconds','rest_seconds','block_rest_seconds','initial_rest_seconds','tail_seconds'):
        p[k] = float(p[k])
        if not math.isfinite(p[k]) or not 0 <= p[k] <= 3600:
            raise ValueError(k+'须为0至3600秒。')
    if p['task_seconds'] < .1 or p['prepare_seconds'] < .1 or p['tail_seconds'] < .1:
        raise ValueError('准备、想象和尾部记录至少0.1秒。')
    p['randomize'], p['sound'] = bool(p['randomize']), bool(p['sound'])
    return p


class Experiment:
    def __init__(self, controller, protocol):
        self.c, self.p = controller, validate_protocol(protocol)
        self.stop = threading.Event()
        self.error = ''
        self.sequence = []
        rng = random.Random(self.p['seed'])
        for loop in range(1, self.p['loops']+1):
            labels = list(range(len(self.p['tasks'])))
            if self.p['randomize']: rng.shuffle(labels)
            base = len(self.sequence)
            self.sequence.extend({'round': loop, 'label': label, 'trial_id': base+i+1} for i, label in enumerate(labels))
        self.thread = threading.Thread(target=self._run, name='studio-experiment', daemon=True)

    def cancel(self, message='操作员中止'):
        self.error = message
        self.stop.set()

    def wait_until(self, deadline):
        while True:
            if self.stop.is_set(): raise RuntimeError(self.error or '会话中止')
            self.c.check_recording()
            left = deadline-time.perf_counter()
            if left <= 0: return
            self.stop.wait(min(left, .025))

    def _run(self):
        p, c, deadline = self.p, self.c, time.perf_counter()
        complete, error = False, ''
        try:
            c.subject('开始', '开始')
            c.mark('session_start', 10)
            deadline += p['initial_rest_seconds']
            self.wait_until(deadline)
            for row in self.sequence:
                task = p['tasks'][row['label']]
                c.subject(task['text'], task['text'])
                c.set_phase('prepare', row)
                c.mark('prepare', 20+row['label'], **row)
                deadline += p['prepare_seconds']
                self.wait_until(deadline)
                # No subject update and no sound at this boundary. Only internal timing/marker.
                c.set_phase('imagery', row)
                c.mark('cue', 1+row['label'], planned_mono_ns=int(deadline*1e9), **row)
                deadline += p['task_seconds']
                self.wait_until(deadline)
                c.mark('task_end', 9, **row)
                last = row['trial_id'] % len(p['tasks']) == 0
                remaining = p['loops']-row['round']
                c.subject('休息', '休息', f'还剩 {remaining} 轮' if last else '')
                c.set_phase('rest', row)
                c.mark('rest', 30, **row)
                deadline += p['rest_seconds']
                self.wait_until(deadline)
                if last and remaining:
                    c.set_phase('block_rest', row)
                    c.mark('block_rest', 31, **row)
                    deadline += p['block_rest_seconds']
                    self.wait_until(deadline)
            c.subject('结束', '结束')
            c.set_phase('finished', {})
            c.mark('session_end', 11)
            deadline += p['tail_seconds']
            self.wait_until(deadline)
            complete = True
        except BaseException as exc:
            error = str(exc)
            c.subject('已中止', '')
            try: c.local_event('abort', error=error)
            except RuntimeError: pass  # A failed disk writer keeps its incomplete manifest and failure record.
        finally:
            c.finish_recording(complete, error)
