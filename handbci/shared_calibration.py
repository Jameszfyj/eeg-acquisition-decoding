"""Model-independent labelled calibration EEG; fitted parameters stay model-specific."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
import uuid

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .config import write_json
from .workbench_settings import calibration_protocol, local_settings

FORMAT = 'handbci.shared_calibration.v1'


def imported_protocol(cfg):
    options = dict(rounds=20, imagery_seconds=6., rest_seconds=2.)
    options.update(local_settings().get('shared_calibration', {}))
    protocol = calibration_protocol({'session_metadata': {'brain_calibration': options}})
    protocol['trials'] = len(cfg['classes']) * protocol['rounds']
    return protocol


def signatures(cfg):
    return [(c['name'], c.get('task_text') or c['name']) for c in cfg['classes']]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


class RawCalibrationData:
    def __init__(self, cfg, protocol):
        self.cfg, self.protocol = cfg, protocol
        self.windows, self.labels, self.starts, self.pending = [], [], [], []

    def append(self, raw, indices, start):
        if start is None or not len(indices):
            return
        end = start + round(self.protocol['imagery_seconds']*self.cfg['device']['sample_rate'])
        keep = (indices >= start) & (indices < end)
        if np.any(keep):
            self.pending.append((np.asarray(raw[:, keep], np.float32).copy(), indices[keep].copy()))

    def discard(self):
        self.pending.clear()

    def finish(self, label, start):
        count = round(self.protocol['imagery_seconds']*self.cfg['device']['sample_rate'])
        if not self.pending:
            raise ValueError('校准任务缺少原始脑电，不能登记为完整校准。')
        raw = np.concatenate([x[0] for x in self.pending], axis=1)
        indices = np.concatenate([x[1] for x in self.pending])
        if (raw.shape != (len(self.cfg['device']['channels']), count)
                or not np.array_equal(indices, np.arange(start, start+count)) or not np.isfinite(raw).all()):
            raise ValueError('校准任务脑电不完整或存在缺口，请重新执行当前任务。')
        self.windows.append(raw)
        self.labels.append(int(label))
        self.starts.append(int(start))
        self.pending.clear()

    def save(self, session):
        if len(self.windows) != self.protocol['trials']:
            return
        from .calibration_store import device_contract
        path = Path(session)
        device = self.cfg['device']
        synthetic = device['source'] in ('synthetic', 'synthetic_tcp')
        identity = (dict(synthetic=True, channels=device['channels'], sample_rate=device['sample_rate'], unit=device['unit'])
                    if synthetic else device_contract(json.loads((path/'device.json').read_text(encoding='utf-8'))))
        arrays = path/'calibration_raw.npz'
        np.savez_compressed(arrays, eeg=np.stack(self.windows), labels=np.asarray(self.labels, np.int64),
                            source_starts=np.asarray(self.starts, np.int64))
        write_json(path/'calibration_data.json', dict(format=FORMAT, complete=True,
            participant_id=self.cfg['session_metadata']['participant_id'],
            classes=[list(x) for x in signatures(self.cfg)], channels=device['channels'],
            sample_rate=device['sample_rate'], unit=device['unit'], synthetic=synthetic,
            reference=device.get('reference'),
            protocol=self.protocol, device=identity, arrays_sha256=digest(arrays)))


class SharedCalibrationStore:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def load(self, session_id, cfg, protocol, *, arrays=False):
        if not isinstance(session_id, str) or not re.fullmatch(r'\d{8}_\d{6}_[0-9a-f]{8}_[\w-]+', session_id):
            raise ValueError('共用校准记录编号无效。')
        path = (self.root/session_id).resolve()
        if path.parent != self.root:
            raise ValueError('校准记录不在当前工作台中。')
        metadata = json.loads((path/'calibration_data.json').read_text(encoding='utf-8'))
        device = cfg['device']
        synthetic = device['source'] in ('synthetic', 'synthetic_tcp')
        source_protocol = metadata.get('protocol', {})
        wanted = [x[1] for x in signatures(cfg)]
        available = [x[1] for x in metadata.get('classes', [])]
        external = source_protocol.get('external_acquisition') is True
        duration = source_protocol.get('imagery_seconds',0)
        raw_count = source_protocol.get('raw_samples_per_trial',round(duration*device['sample_rate']))
        timing_matches = (source_protocol.get('rounds') == protocol['rounds'] and
            ((external and 4 <= duration <= 30 and type(raw_count) is int
              and 3.9*device['sample_rate'] <= raw_count <= duration*device['sample_rate'])
             or (not external and all(source_protocol.get(key)==protocol[key] for key in ('imagery_seconds','rest_seconds')))))
        if (metadata.get('format') != FORMAT or metadata.get('complete') is not True
                or metadata.get('participant_id') != cfg['session_metadata']['participant_id']
                or metadata.get('synthetic') != synthetic or metadata.get('sample_rate') != device['sample_rate']
                or metadata.get('unit') != device['unit']
                or metadata.get('reference') != device.get('reference')
                or len(available) != len(set(available)) or len(wanted) != len(set(wanted))
                or not set(wanted).issubset(available)
                or any(c not in metadata['channels'] for c in device['channels'])
                or not timing_matches):
            raise ValueError('共用校准的被试、任务、通道、单位、采样率或校准轮数不匹配。')
        calibration = json.loads((path/'calibration.json').read_text(encoding='utf-8'))
        state = calibration.get('state', {})
        if (calibration.get('complete') is not True or not state.get('ready')
                or state.get('completed') != source_protocol.get('trials')
                or source_protocol.get('trials') != len(available)*protocol['rounds']):
            raise ValueError('共用校准没有完成全部提示任务。')
        if digest(path/'calibration_raw.npz') != metadata['arrays_sha256']:
            raise ValueError('原始校准脑电校验失败。')
        summary = dict(id=session_id, participant_id=metadata['participant_id'],
            saved_at=time.strftime('%Y-%m-%d %H:%M', time.localtime((path/'calibration_data.json').stat().st_mtime)),
            calibration_trials=protocol['trials'], calibration_rounds=protocol['rounds'],
            device=metadata['device'], shared_calibration_input=True,
            arrays_sha256=metadata['arrays_sha256'], metadata_sha256=digest(path/'calibration_data.json'),
            calibration_sha256=digest(path/'calibration.json'), model_independent=True)
        summary.update(external_acquisition=external,imagery_seconds=duration,
                       prepare_seconds=source_protocol.get('prepare_seconds'))
        result = dict(summary=summary, path=path, metadata=metadata)
        if arrays:
            with np.load(path/'calibration_raw.npz', allow_pickle=False) as values:
                raw, labels, starts = values['eeg'].copy(), values['labels'].copy(), values['source_starts'].copy()
            expected = (source_protocol['trials'], len(metadata['channels']),
                        raw_count)
            if (raw.shape != expected or labels.shape != (len(raw),) or starts.shape != labels.shape
                    or not np.issubdtype(labels.dtype, np.integer) or not np.issubdtype(starts.dtype,np.integer)
                    or np.any(starts<0) or not np.isfinite(raw).all()
                    or np.any(labels < 0) or np.any(labels >= len(available))
                    or np.bincount(labels, minlength=len(available)).tolist() != [protocol['rounds']]*len(available)):
                raise ValueError('共用校准数组或每类完整任务数不正确。')
            target_ids = {available.index(signature): i for i, signature in enumerate(wanted)}
            keep = np.isin(labels, list(target_ids))
            order = [metadata['channels'].index(c) for c in device['channels']]
            result.update(raw=raw[keep][:, order], labels=np.asarray([target_ids[int(x)] for x in labels[keep]]),
                          starts=starts[keep])
        return result

    def history(self, cfg, protocol):
        available, unavailable = [], []
        for path in sorted(self.root.glob('*/calibration_data.json'), reverse=True):
            try:
                metadata = json.loads(path.read_text(encoding='utf-8'))
                if metadata.get('participant_id') != cfg['session_metadata']['participant_id']:
                    continue
                available.append(self.load(path.parent.name, cfg, protocol)['summary'])
            except (OSError, ValueError, KeyError, TypeError) as error:
                unavailable.append(dict(id=path.parent.name, reason=str(error)))
        return dict(available=available, unavailable=unavailable)


def reset_decoder(decoder, first):
    if hasattr(decoder, 'recover_stream'):
        decoder.recover_stream(first, 'calibration replay / fresh decoding')
    else:
        decoder.reset(first)


def replay_probabilities(decoder, raw, starts, cfg):
    probabilities = []
    fs = cfg['device']['sample_rate']
    support = cfg['window']['seconds'] + getattr(decoder.stream, 'history_source_samples', 0)/fs
    if support > raw.shape[-1]/fs:
        raise ValueError('模型窗口及滤波历史超过6秒想象时长，无法使用当前校准数据。')
    for trial, start in zip(raw, starts):
        start = int(start)
        reset_decoder(decoder, start)
        # Use up to three completed windows wholly contained in this imagery task.
        ends = sorted(set([min(trial.shape[1], max(math.ceil(support*fs)+4, trial.shape[1]-j*round(fs))) for j in (2,1,0)]))
        cursor, values = 0, []
        for end in ends:
            prediction = decoder.process(trial[:, cursor:end], np.arange(start+cursor, start+end))
            cursor = end
            if prediction is not None:
                values.append(np.asarray(prediction[1], float))
        if not values:
            raise ValueError('校准任务没有足够的完整模型窗口。')
        probabilities.append(np.mean(values, axis=0))
    return np.asarray(probabilities)


def fit_bias(probabilities, labels):
    p, labels = np.asarray(probabilities, float), np.asarray(labels, np.int64)
    if (p.ndim != 2 or len(p) != len(labels) or not np.isfinite(p).all()
            or np.any(p < 0) or not np.allclose(p.sum(1), 1., atol=1e-5)):
        raise ValueError('模型校准概率无效。')
    scores = np.log(np.maximum(p, 1e-12))
    def objective(bias):
        adjusted = scores+bias
        normalizer = logsumexp(adjusted, axis=1)
        loss = np.sum(normalizer-adjusted[np.arange(len(labels)), labels])+.5*np.sum(bias*bias)
        gradient = np.exp(adjusted-normalizer[:,None]).sum(0)-np.bincount(labels,minlength=p.shape[1])+bias
        return loss, gradient
    result = minimize(objective, np.zeros(p.shape[1]), jac=True, method='L-BFGS-B', bounds=[(-2.,2.)]*p.shape[1])
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError('模型校准计算失败，请检查校准信号和任务标签。')
    return result.x


class ImportedCalibrationDecoder:
    def __init__(self, base, cfg, selection):
        self.base, self.stream, self.cfg, self.selection = base, base.stream, cfg, selection
        self.protocol = imported_protocol(cfg)
        support = cfg['window']['seconds']+getattr(base.stream,'history_source_samples',0)/self.stream.fs
        if support > self.protocol['imagery_seconds']:
            raise ValueError('模型窗口及滤波历史超过6秒想象时长，无法按当前设置校准。')
        self.names = [c.get('task_text') or c['name'] for c in cfg['classes']]
        self.total = len(self.names)*self.protocol['rounds']
        rng = np.random.default_rng(42)
        self.tasks = np.concatenate([rng.permutation(len(self.names)) for _ in range(self.protocol['rounds'])]).tolist()
        self.data = RawCalibrationData(cfg, self.protocol)
        self.lock = threading.RLock()
        self.completed, self.latest_index, self.warming_until = 0, -1, round(3*self.stream.fs)
        self.phase, self.prompt_id, self.started_index, self.prompt_created = 'warming', None, None, None
        self.remaining, self.bias, self.reused_from = 0., None, None
        self.events, self.last_window, self.last_details = [], None, None
        self.session_id, self.stream_recovery_count = uuid.uuid4().hex, 0
        self.audio_enabled = bool(cfg['session_metadata'].get('calibration_audio_enabled'))

    def snapshot_calibration(self):
        with self.lock:
            return dict(required=True, completed=self.completed, total=self.total, ready=self.bias is not None,
                round=min(self.completed//len(self.names)+1,self.protocol['rounds']), rounds=self.protocol['rounds'],
                trial=min(self.completed+1,self.total), phase=self.phase,
                task=self.names[self.tasks[min(self.completed,self.total-1)]], remaining_seconds=self.remaining,
                prompt_id=self.prompt_id, awaiting_display=self.prompt_id is not None and self.started_index is None,
                session_id=self.session_id, audio_enabled=self.audio_enabled, labels_used=True,
                imagery_seconds=self.protocol['imagery_seconds'], rest_seconds=self.protocol['rest_seconds'],
                stream_recovery_count=self.stream_recovery_count, reused=self.reused_from is not None,
                reused_from=self.reused_from, physical_event_time_verified=False)

    def _prompt(self, phase):
        self.phase, self.started_index = phase, None
        self.prompt_id, self.prompt_created = uuid.uuid4().hex, time.monotonic()
        self.remaining = self.protocol['imagery_seconds' if phase == 'imagery' else 'rest_seconds']

    def acknowledge(self, prompt_id, session_id, audio=None):
        with self.lock:
            if session_id != self.session_id or prompt_id != self.prompt_id or not self.prompt_id:
                raise ValueError('校准提示已过期。')
            if self.started_index is None:
                expected = '休息' if self.phase == 'rest' else self.names[self.tasks[self.completed]]
                if self.audio_enabled:
                    if (not isinstance(audio, dict) or audio.get('enabled') is not True
                            or audio.get('engine') != 'local_wav_webaudio' or audio.get('text') != expected):
                        raise ValueError('校准语音尚未完成。')
                    values = [audio.get(k) for k in ('scheduled_browser_ms','completed_browser_ms','duration_seconds')]
                    if (any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in values)
                            or not 0 < values[2] <= 10 or values[1]-values[0] < values[2]*1000-100):
                        raise ValueError('校准语音完成时间无效。')
                self.started_index = self.latest_index+1
                self.events.append(dict(kind=self.phase,trial=self.completed+1,task_id=self.tasks[self.completed],
                    source_index=self.started_index,physical_event_time_verified=False))
            return dict(ok=True)

    def _fit(self, raw, labels, starts):
        self.bias = fit_bias(replay_probabilities(self.base,raw,starts,self.cfg),labels)
        reset_decoder(self.base, self.latest_index+1)

    def restore_calibration(self, saved):
        with self.lock:
            self._fit(saved['raw'],saved['labels'],saved['starts'])
            self.tasks = saved['labels'].tolist()
            self.completed, self.phase, self.audio_enabled = self.total, 'complete', False
            self.reused_from = copy.deepcopy(saved['summary'])

    def recover_stream(self, first_source_index, reason):
        with self.lock:
            self.data.discard()
            reset_decoder(self.base, first_source_index)
            self.latest_index, self.warming_until = first_source_index-1, first_source_index+round(3*self.stream.fs)
            self.phase = 'complete' if self.bias is not None else 'warming'
            self.started_index = self.prompt_id = self.prompt_created = None
            self.stream_recovery_count += 1

    def process(self, raw, indices):
        with self.lock:
            self.latest_index = int(indices[-1])
            if self.bias is not None:
                prediction = self.base.process(raw,indices)
                self.last_window = getattr(self.base,'last_window',None)
                if prediction is None:
                    return None
                self.last_details = dict(uncalibrated_probabilities=np.asarray(prediction[1]).tolist(),
                    calibration_method='model_specific_logit_bias_v1',bias=self.bias.tolist())
                scores = np.log(np.maximum(prediction[1],1e-12))+self.bias
                return prediction[0], np.exp(scores-logsumexp(scores))
            if self.phase == 'warming':
                if self.latest_index >= self.warming_until:
                    self._prompt('rest')
                return None
            if self.started_index is None:
                if time.monotonic()-self.prompt_created > (12 if self.audio_enabled else 5):
                    raise RuntimeError('校准提示未收到页面显示或语音完成回执，已停止。')
                return None
            if self.phase == 'imagery':
                self.data.append(raw,indices,self.started_index)
            duration = self.protocol['imagery_seconds' if self.phase == 'imagery' else 'rest_seconds']
            self.remaining = max(0.,duration-(self.latest_index+1-self.started_index)/self.stream.fs)
            if self.remaining > 0:
                return None
            if self.phase == 'rest':
                self._prompt('imagery')
                return None
            self.data.finish(self.tasks[self.completed],self.started_index)
            self.completed += 1
            if self.completed < self.total:
                self._prompt('rest')
            else:
                self._fit(np.stack(self.data.windows),self.data.labels,self.data.starts)
                self.phase, self.prompt_id = 'complete', None
            return None

    def save_calibration(self, session):
        with self.lock:
            complete = self.bias is not None
            write_json(Path(session)/'calibration.json',dict(state=self.snapshot_calibration(),protocol=self.protocol,
                complete=complete,events=self.events,labels_used=True,reused_from=self.reused_from,
                method='model_specific_logit_bias_v1',bias=self.bias.tolist() if complete else None,
                model_sha256=self.selection.get('sha256'),backbone_updated=False,physical_event_time_verified=False))
            if complete and not self.reused_from:
                self.data.save(session)
