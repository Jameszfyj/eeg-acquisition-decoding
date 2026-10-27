"""Optional trial replay. Runs after recording each trial, never on the acquisition thread."""
from __future__ import annotations

from collections import deque
import base64
import copy
import io
import json
from pathlib import Path
import queue
import shutil
import threading
import time
import zipfile

import numpy as np

from ..model_bundle import ONNXDecoder, MAX_UPLOAD, fingerprint, import_bundle
from ..subject_online import resolve_path
from .storage import json_write


def import_validation_model(root, participant, data, filename, manifest=None):
    raw = base64.b64decode(data, validate=True)
    if not 0 < len(raw) <= MAX_UPLOAD:
        raise ValueError('验证模型须为不超过32MB的ONNX包。')
    if str(filename).lower().endswith('.onnx'):
        if not isinstance(manifest, dict):
            raise ValueError('单独ONNX须同时提供manifest.json，以确定通道、预处理和类别。也可直接上传模型ZIP。')
        out = io.BytesIO()
        with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('model.onnx', raw)
            archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False))
        raw = out.getvalue()
    elif not str(filename).lower().endswith('.zip'):
        raise ValueError('请选择模型ZIP，或ONNX及配套manifest.json。')
    entry = import_bundle(raw, participant, registry=Path(root)/'runtime/studio/validation_models.json',
                          import_root=Path(root)/'runtime/studio/validation_models')
    contract = json.loads(resolve_path(entry['path']).read_text(encoding='utf-8'))
    return dict(entry, classes=[c['name'] for c in contract['config']['classes']])


def vote_scores(probabilities):
    """One vote per complete window; tied windows abstain, never favor class zero."""
    ps = np.asarray(probabilities, float)
    counts = np.zeros(ps.shape[1], int)
    for p in ps:
        order = np.argsort(p)
        if p[order[-1]]-p[order[-2]] > 1e-6:
            counts[order[-1]] += 1
    mean = ps.mean(0)
    candidates = np.flatnonzero(counts == counts.max()) if counts.max() else np.array([], int)
    if len(candidates) > 1:
        candidates = candidates[np.abs(mean[candidates]-mean[candidates].max()) <= 1e-6]
    return dict(votes=counts.tolist(), vote_shares=(counts/len(ps)).tolist(),
                mean_probabilities=mean.tolist(), windows=len(ps),
                abstained_windows=int(len(ps)-counts.sum()),
                predicted_class=int(candidates[0]) if len(candidates) == 1 else None)


class TrialValidation:
    def __init__(self, entry, participant, description, protocol):
        folder = resolve_path(entry['path']).parent
        if entry['participant_id'] != participant or fingerprint(folder) != entry['sha256']:
            raise ValueError('验证模型的被试或文件校验不匹配，请重新导入。')
        self.decoder = ONNXDecoder(folder, prediction_stride_seconds=.5)
        cfg = self.decoder.manifest['config']
        self.classes = [c['name'] for c in cfg['classes']]
        tasks = [t['text'] for t in protocol['tasks']]
        if len(set(self.classes)) != len(self.classes) or set(tasks) != set(self.classes):
            raise ValueError('录制任务名称须与验证模型类别一致（顺序可不同）：'+ '、'.join(self.classes))
        if (cfg['device']['sample_rate'] != description['settings']['fs']
                or cfg['device']['unit'] != description['unit']):
            raise ValueError('验证模型采样率或单位与当前设备不一致。')
        channels = cfg['device']['channels']
        labels = description['labels']
        if len(labels) != len(set(labels)) or any(c not in labels for c in channels):
            raise ValueError('当前设备通道无法按名称匹配验证模型。')
        self.order = [labels.index(c) for c in channels]
        self.task_map = [self.classes.index(t) for t in tasks]
        self.fs = float(cfg['device']['sample_rate'])
        self.history = getattr(self.decoder.stream, 'history_source_samples', 0)
        self.warmup = max(self.history, round(cfg['preprocess']['trim_seconds']*self.fs)
                          if self.decoder.manifest['input']['pipeline'] != 'raw' else 0)
        self.retain_samples = round((protocol['prepare_seconds']+protocol['task_seconds']+10)*self.fs)+self.warmup
        self.folder, self.entry = folder, copy.deepcopy(entry)
        self.contract = dict(enabled=True, format='handbci.offline_onnx_validation.v1', model=copy.deepcopy(entry),
            classes=self.classes, stride_seconds=.5, window_seconds=cfg['window']['seconds'],
            anchor='prepare' if self.decoder.prepare_aligned else 'cue_full_raw_support',
            window_end='before_task_end_no_rest_samples', voting='all_complete_windows_one_vote_each',
            tie_break='mean_probability_then_unresolved', feedback_phase='rest',
            task_labels_used_for_prediction=False, path='onnx_validation/summary.json')
        self.lock = threading.RLock()
        self.queue = queue.Queue(128)
        self.batches, self.markers, self.completed = deque(), {}, set()
        self.results = []
        self.state = dict(enabled=True, status='waiting', classes=self.classes, latest=None,
                          model_name=entry['name'], model_id=entry['model_id'],
                          completed_trials=0, scored_trials=0, correct_trials=0, error='')
        self.path = None
        self.closed = threading.Event()
        self.thread = None

    def start(self, session):
        self.path = Path(session)/'onnx_validation'
        self.path.mkdir()
        for name in ('manifest.json', 'model.onnx'):
            shutil.copyfile(self.folder/name, self.path/name)
        json_write(self.path/'metadata.json', self.contract)
        self.thread = threading.Thread(target=self._run, name='studio-onnx-validation', daemon=True)
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state)

    def submit(self, kind, value):
        if self.snapshot()['error'] or self.closed.is_set():
            return
        try:
            self.queue.put_nowait((kind, value))
        except queue.Full:
            with self.lock:
                self.state.update(status='failed', error='验证队列已满，验证停止；原始采集继续保存。')

    def finish(self):
        self.closed.set()
        self.thread.join(10)
        if self.thread.is_alive():
            with self.lock:
                self.state.update(status='failed', error='验证未及时结束，原始采集已独立保存。')
        return self.snapshot()

    def _save(self):
        summary = dict(self.contract, **self.snapshot(), trials=self.results)
        temporary = self.path/'summary.json.tmp'
        json_write(temporary, summary)
        temporary.replace(self.path/'summary.json')

    def _result(self, row, trials):
        self.completed.add(row['trial_id'])
        self.markers.pop(row['trial_id'], None)
        self.results.append(row)
        trials.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        trials.flush()
        with self.lock:
            self.state.update(latest=row, completed_trials=len(self.results),
                scored_trials=sum(r.get('correct') is not None for r in self.results),
                correct_trials=sum(r.get('correct') is True for r in self.results))
        self._save()

    def _run(self):
        try:
            with (self.path/'predictions.jsonl').open('x', encoding='utf-8') as predictions, \
                    (self.path/'trials.jsonl').open('x', encoding='utf-8') as trials:
                while not (self.closed.is_set() and self.queue.empty()):
                    try:
                        kind, value = self.queue.get(timeout=.05)
                    except queue.Empty:
                        continue
                    if self.snapshot()['error']:
                        continue
                    if kind == 'data':
                        self.batches.append(value)
                        end = value[2]+value[0].shape[1]
                        while self.batches and self.batches[0][2]+self.batches[0][0].shape[1] < end-self.retain_samples:
                            self.batches.popleft()
                    elif kind == 'event' and value.get('kind') in ('prepare','cue','task_end'):
                        self.markers.setdefault(value['trial_id'], {})[value['kind']] = value
                    self._ready(predictions, trials)
                if not self.snapshot()['error']:
                    self._ready(predictions, trials, final=True)
                with self.lock:
                    self.state['status'] = 'failed' if self.state['error'] else 'finished'
                self._save()
        except Exception as exc:
            with self.lock:
                self.state.update(status='failed', error=f'验证失败：{type(exc).__name__}: {exc}；原始采集独立保存。')
            try:
                self._save()
            except OSError:
                pass

    def _ready(self, predictions, trials, final=False):
        for trial_id, marks in list(self.markers.items()):
            if trial_id in self.completed:
                continue
            if not all(k in marks for k in ('prepare', 'cue', 'task_end')):
                if final:
                    self._result(dict(trial_id=trial_id, status='skipped', reason='试次未完成或缺少标记', correct=None), trials)
                continue
            if not self.batches or self.batches[-1][1][-1]+1000/self.fs < marks['task_end']['device_time_ms']:
                if final:
                    self._result(dict(trial_id=trial_id, status='skipped', reason='任务结束边界尚无完整数据', correct=None), trials)
                continue
            with self.lock:
                self.state.update(status='evaluating', latest=dict(trial_id=trial_id, status='evaluating', windows=0))
            try:
                row = self.evaluate(marks, predictions)
            except (ValueError, RuntimeError) as exc:
                row = dict(trial_id=trial_id, status='skipped', reason=str(exc), correct=None)
            self._result(row, trials)
            with self.lock:
                self.state['status'] = 'failed' if self.state['error'] else 'waiting'

    def evaluate(self, marks, predictions):
        raw = np.concatenate([b[0] for b in self.batches], axis=1)[self.order]
        stamps = np.concatenate([b[1] for b in self.batches])
        ix = np.concatenate([np.arange(b[2], b[2]+b[0].shape[1]) for b in self.batches])
        if np.any(np.diff(stamps) <= 0):
            raise ValueError('设备时钟倒退，无法验证')
        positions = {k:int(np.searchsorted(stamps, marks[k]['device_time_ms'])) for k in marks}
        prepare, cue, end = (positions[k] for k in ('prepare','cue','task_end'))
        if not 0 <= prepare < cue < end <= len(ix):
            raise ValueError('试次边界无效或数据不足')
        factor = getattr(self.decoder.stream, 'factor', 1)
        anchor = round(int(ix[prepare])/factor)*factor if self.decoder.prepare_aligned else int(ix[cue])+self.history
        anchor = ((anchor+factor-1)//factor)*factor
        start = max(0, int(np.searchsorted(ix, anchor-self.warmup-factor)))
        # The entire prepare/task interval and required FIR support must be continuous.
        check_start = min(start, prepare)
        if (np.any(np.diff(ix[check_start:end]) != 1)
                or np.any(np.abs(np.diff(stamps[check_start:end])-1000/self.fs) > max(.501, .05*1000/self.fs))):
            raise ValueError('试次或滤波历史跨Gap，未拼接、未补点、未投票')
        for k in ('prepare','cue','task_end'):
            j = positions[k]
            if j and j < len(stamps) and stamps[j]-stamps[j-1] > 1000/self.fs+.501:
                raise ValueError('标记位于设备时间缺口')
        if anchor-self.history < ix[start]:
            raise ValueError('滤波历史不足')
        decoder = self.decoder
        decoder.stream.reset(int(ix[start]))
        values, retained = decoder.stream.process(raw[:,start:end], ix[start:end])
        step = round(.5*self.fs)
        ps, bounds = [], []
        trial_id = int(marks['cue']['trial_id'])
        target = self.task_map[int(marks['cue']['label'])]
        verified = all(marks[k].get('echo_verified', False) for k in ('prepare','cue','task_end'))
        for source in range(anchor, int(ix[end-1])+1, step):
            a = int(np.searchsorted(retained, source))
            if a+decoder.n_time > len(retained):
                break
            if retained[a] != source:
                continue
            p = decoder.predict_window(values[decoder.order,a:a+decoder.n_time])
            window = decoder.stream.window_bounds(retained[a:a+decoder.n_time], int(ix[end-1]))
            window['model_start_index'] = int(source)
            ps.append(p);bounds.append(window)
            scores = vote_scores(ps)
            row = dict(trial_id=trial_id, status='evaluating', classes=self.classes,
                       target_class=target, label_verified=verified, **scores)
            with self.lock:
                self.state['latest'] = row
            predictions.write(json.dumps(dict(trial_id=trial_id, window_number=len(ps),
                input_window=window, probabilities=p.tolist(), **scores), ensure_ascii=False, allow_nan=False)+'\n')
        predictions.flush()
        if not ps:
            raise ValueError('完整模型窗口不足，未投票')
        scores = vote_scores(ps)
        return dict(trial_id=trial_id, status='completed', classes=self.classes,
                    target_class=target, label_verified=verified,
                    correct=(scores['predicted_class'] == target) if verified else None,
                    prepare_index=int(ix[prepare]), cue_index=int(ix[cue]), end_index=int(ix[end-1])+1,
                    first_window=bounds[0], last_window=bounds[-1], **scores)
