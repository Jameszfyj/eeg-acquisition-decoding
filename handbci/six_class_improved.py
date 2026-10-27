"""Session-local EA calibration and streaming adaptation of the saved pair router.

Browser render acknowledgements are software cue times, not physical trigger
verification. The original offline/online-validation flags remain unchanged.
"""
from __future__ import annotations

import copy
import threading
import time
import uuid
import math

import numpy as np
from scipy import signal

from .four_class_online import CHANNELS, SCALP, digest
from .preprocess import StreamPreprocessor
from .workbench_settings import calibration_protocol
from .six_class_online import (ARCH, NAMES, LIMITS, IMPROVED_MODEL, IMPROVED_SHA256,
                              LABELLED_MODEL, LABELLED_SHA256)


def load_checkpoint(path=IMPROVED_MODEL, expected_digest=IMPROVED_SHA256):
    import torch
    from .model import IFNetClassifier
    from pathlib import Path
    labelled = Path(path).resolve() == LABELLED_MODEL.resolve()
    wanted_path, wanted_digest = (LABELLED_MODEL, LABELLED_SHA256) if labelled else (IMPROVED_MODEL, IMPROVED_SHA256)
    if (Path(path).resolve() != wanted_path.resolve() or expected_digest != wanted_digest
            or digest(path) != wanted_digest):
        raise ValueError('改进模型路径或校验值不符。')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if labelled and (payload.get('hand_action_calibration') != dict(ridge=1.,alpha=1.,
            labeled_prefix_rounds=3,labeled_prefix_tasks=18,later_labels_used=False)
            or payload.get('training',{}).get('n_trials') != 629
            or len(payload.get('training',{}).get('source_h5_sha256',{})) != 6):
        raise ValueError('有标签校准模型的来源或规则不符。')
    payload = copy.deepcopy(payload)
    payload['_hand_supervised_calibration'] = labelled
    return load_parent_models(payload)


def load_parent_models(payload):
    """Validate the unchanged six-class pair router inside a trusted checkpoint."""
    from .model import IFNetClassifier
    if (payload.get('format') != 'handbci.six_class_pair_router.full_candidate.v1'
            or payload.get('class_names') != NAMES
            or payload.get('channel_names') != [CHANNELS[i] for i in SCALP]
            or payload.get('router') != dict(pair_classes=[4, 5], hand_classes=[0, 1, 2, 3], alpha=1.)
            or payload.get('calibration') != dict(n_rounds=3, n_trials=18, n_windows_per_trial=3,
                covariance_band=[4., 40.], shrinkage=.1, labels_used=False)
            or any(payload.get(k) is not v for k, v in LIMITS.items())):
        raise ValueError('改进模型的类别、校准或复判约定不一致。')
    models = {}
    for key, bands in [('baseline', [[4., 16.], [16., 40.]]), ('expert', [[8., 13.], [13., 30.]])]:
        component = payload[key]
        prep = component['preprocessing']
        if (component['architecture'] != ARCH or prep['filter_bands'] != bands
                or prep['window_offsets_after_hardware_cue_seconds'] != [.6, 1., 1.4]
                or prep['window_seconds'] != 2. or prep['standardizer'] is not None
                or prep['rereference'] is not None or prep['input_rate'] != 1000
                or prep['output_rate'] != 250 or prep['numtaps'] != 201):
            raise ValueError('改进模型预处理约定不一致。')
        model = IFNetClassifier(**ARCH).eval()
        model.load_state_dict(component['state_dict'], strict=True)
        models[key] = model
    return payload, models


def band_windows(values, bands):
    return np.stack([signal.sosfiltfilt(signal.butter(4, b, fs=250., btype='bandpass',
        output='sos'), values, axis=-1).astype(np.float32) for b in bands], axis=1)


def fit_alignment(values, *, n_trials=18):
    """Original three-window EA estimator, with an explicit complete trial count."""
    values = np.asarray(values)
    if (isinstance(n_trials, bool) or not isinstance(n_trials, int) or not 6 <= n_trials <= 600
            or n_trials % 6 or values.shape != (n_trials*3, 30, 500) or not np.isfinite(values).all()):
        raise ValueError(f'校准需要{n_trials}次完整任务、每次3个窗口，不能用缺失或后续数据补齐。')
    filtered = signal.sosfiltfilt(signal.butter(4, [4., 40.], btype='bandpass',
        fs=250., output='sos'), values, axis=-1)
    centered = filtered-filtered.mean(-1, keepdims=True)
    cov = centered@centered.swapaxes(-1, -2)/499
    mean = cov.reshape(n_trials, 3, 30, 30).mean(1).mean(0)
    shrunk = .9*mean+.1*np.trace(mean)/30*np.eye(30)
    eig, vectors = np.linalg.eigh(shrunk)
    if eig.min() <= 0:
        raise ValueError('校准协方差无效。')
    alignment = (vectors/np.sqrt(eig))@vectors.T
    error = float(abs(alignment@shrunk@alignment.T-np.eye(30)).max())
    if error >= 1e-8:
        raise ValueError('校准矩阵核验失败。')
    return alignment, dict(n_trials=n_trials, n_overlapping_windows=n_trials*3, labels_used=False,
        identity_max_error=error, condition_number=float(eig.max()/eig.min()))


class ImprovedSixClassDecoder:
    def __init__(self, payload, models, cfg):
        self.payload, self.models = payload, models
        self.stream = StreamPreprocessor(cfg)
        self.protocol = calibration_protocol(cfg)
        self.rounds = self.protocol['rounds']
        self.total = self.protocol['trials']
        self.imagery_seconds = self.protocol['imagery_seconds']
        self.rest_seconds = self.protocol['rest_seconds']
        self.lock = threading.RLock()
        self.notch = signal.iirnotch(50., 30., fs=250.)
        self.history = np.empty((30, 0), np.float32)
        self.indices = np.empty(0, np.int64)
        self.latest_index, self.last_prediction = -1, -float('inf')
        rng = np.random.default_rng(42)
        self.tasks = np.concatenate([rng.permutation(6) for _ in range(self.rounds)]).tolist()
        self.completed, self.windows, self.events = 0, [], []
        self.alignment, self.calibration_info = None, None
        self.phase, self.prompt_id = 'warming', None
        self.started_index, self.prompt_created = None, None
        self.remaining, self.last_details = 0., None
        self.last_window = None
        self.session_id = uuid.uuid4().hex
        self.audio_enabled = bool(cfg.get('session_metadata', {}).get('calibration_audio_enabled', False))
        self.reused_from = None
        self.warming_until = 3000
        self.stream_recovery_count = 0
        self.hand_supervised = bool(cfg.get('session_metadata',{}).get('hand_supervised_calibration',False))
        if self.hand_supervised != bool(payload.get('_hand_supervised_calibration',False)):
            raise ValueError('模型与有标签校准模式不一致。')
        self.hand_bias = None

    def recover_stream(self, first_source_index, reason):
        with self.lock:
            self.stream_recovery_count += 1
            self.events.append(dict(kind='stream_interruption', reason=reason,
                first_new_source_index=first_source_index, completed_trials_kept=self.completed,
                repeated_task_id=None if self.alignment is not None else self.tasks[self.completed],
                abandoned_prompt_id=self.prompt_id, physical_event_time_verified=False))
            self.stream.reset(first_source_index)
            self.history = self.history[:, :0]
            self.indices = self.indices[:0]
            self.latest_index = first_source_index-1
            self.last_prediction = -float('inf')
            self.last_details = self.last_window = None
            self.started_index = self.prompt_id = self.prompt_created = None
            self.remaining = 0.
            self.warming_until = first_source_index+3000
            self.phase = 'warming' if self.alignment is None else 'complete'

    def restore_calibration(self, saved):
        # The stream/filter state and EEG history remain fresh for this connection.
        with self.lock:
            if self.latest_index != -1 or self.alignment is not None:
                raise ValueError('只能在新连接接收脑电前载入上次校准。')
            restored_bias = None
            if self.hand_supervised:
                from .labelled_hand_calibration import validate_labels
                labels = validate_labels(saved.get('task_labels'), rounds=self.rounds)
                restored_bias = self._hand_bias_from(saved['windows'],labels)
                if not np.allclose(restored_bias,saved.get('hand_bias'),rtol=1e-9,atol=1e-10):
                    raise ValueError('上次有标签动作校准核验失败，请重新校准。')
            self.alignment = saved['alignment'].copy()
            self.windows = list(saved['windows'].copy())
            self.calibration_info = copy.deepcopy(saved['info'])
            if self.hand_supervised:
                self.tasks = labels.tolist()
                self.hand_bias = restored_bias
            self.reused_from = copy.deepcopy(saved['summary'])
            self.completed, self.phase, self.audio_enabled = self.total, 'complete', False
            self.events.append(dict(kind='calibration_reused', source=saved['summary'],
                                    physical_event_time_verified=False))

    def snapshot_calibration(self):
        with self.lock:
            done = self.alignment is not None and (not self.hand_supervised or self.hand_bias is not None)
            label = self.tasks[min(self.completed, self.total-1)]
            return dict(required=True, completed=self.completed, total=self.total,
                rounds=self.rounds, imagery_seconds=self.imagery_seconds, rest_seconds=self.rest_seconds,
                round=min(self.completed//6+1, self.rounds), trial=min(self.completed+1, self.total),
                phase=self.phase, ready=done, task=NAMES[label],
                remaining_seconds=self.remaining, prompt_id=self.prompt_id,
                awaiting_display=self.prompt_id is not None and self.started_index is None,
                session_id=self.session_id, physical_event_time_verified=False,
                audio_enabled=self.audio_enabled,
                stream_recovery_count=self.stream_recovery_count,
                labels_used=self.hand_supervised,
                reused=self.reused_from is not None, reused_from=self.reused_from,
                instruction=('已载入上次校准，正在准备持续解码' if self.reused_from else
                    '校准完成，正在准备持续解码' if done else
                    f'请想象{NAMES[label]}' if self.phase == 'imagery' else
                    f'放松，下一项：{NAMES[label]}' if self.phase == 'rest' else '正在准备脑电校准'))

    def _prompt(self, phase):
        self.phase, self.started_index = phase, None
        self.prompt_id = uuid.uuid4().hex
        self.prompt_created = time.monotonic()
        self.remaining = self.imagery_seconds if phase == 'imagery' else self.rest_seconds

    def acknowledge(self, prompt_id, session_id, audio=None):
        with self.lock:
            if session_id != self.session_id or prompt_id != self.prompt_id or not self.prompt_id:
                raise ValueError('校准提示已过期，未改变校准状态。')
            if self.started_index is None:
                if self.audio_enabled:
                    expected = '休息' if self.phase == 'rest' else NAMES[self.tasks[self.completed]]
                    if (not isinstance(audio, dict) or audio.get('enabled') is not True
                            or audio.get('engine') != 'local_wav_webaudio' or audio.get('text') != expected):
                        raise ValueError('校准语音未完成，不能开始计时。')
                    values = [audio.get(k) for k in ('scheduled_browser_ms', 'completed_browser_ms', 'duration_seconds')]
                    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)
                            or values[1] < values[0] or not 0 < values[2] <= 10
                            or values[1]-values[0] < values[2]*1000-100):
                        raise ValueError('校准语音播放完成时间无效。')
                self.started_index = self.latest_index+1
                self.events.append(dict(kind=self.phase, trial=self.completed+1,
                    round=self.completed//6+1, task_id=self.tasks[self.completed],
                    source_index=self.started_index, server_ack_mono=time.monotonic(),
                    prompt_id=self.prompt_id, timing_source='browser_audio_completion_ack' if self.audio_enabled else 'browser_render_ack',
                    physical_event_time_verified=False))
                if self.audio_enabled:
                    self.events[-1]['audio_cue'] = {k:audio[k] for k in
                        ('enabled','engine','text','scheduled_browser_ms','completed_browser_ms','duration_seconds')}
            return dict(ok=True)

    def _notched(self, values):
        return signal.filtfilt(*self.notch, values, axis=-1).astype(np.float32)

    def _trial_windows(self):
        # The original FIR has 0.5 s group delay. Logical window times match
        # the offline candidate's retained indices minus 500 raw samples.
        logical = self.indices-self.stream.delay_seconds*self.stream.fs
        result = []
        for offset in (.6, 1., 1.4):
            start = int(np.searchsorted(logical, self.started_index+offset*self.stream.fs))
            if start+500 > len(logical):
                raise ValueError('校准窗口不完整，已停止。')
            # The 1 s causal FIR support must remain within the imagery period.
            if (self.indices[start]-1000 < self.started_index
                    or self.indices[start+499] >= self.started_index+self.imagery_seconds*self.stream.fs):
                raise ValueError('校准滤波支持超出任务边界。')
            result.append(self.history[:, start:start+500])
        return self._notched(np.stack(result))

    def _calibrate_step(self):
        if self.phase == 'warming':
            if self.latest_index >= self.warming_until:
                self._prompt('rest')
            return
        if self.started_index is None:
            timeout = 12 if self.audio_enabled else 5
            if time.monotonic()-self.prompt_created > timeout:
                raise RuntimeError(f'校准提示超过{timeout}秒未收到页面显示回执或语音完成回执，已停止，请保持页面可见。')
            return
        duration = self.imagery_seconds if self.phase == 'imagery' else self.rest_seconds
        self.remaining = max(0., duration-(self.latest_index+1-self.started_index)/self.stream.fs)
        if self.remaining > 0:
            return
        if self.phase == 'rest':
            self._prompt('imagery')
            return
        self.windows.extend(self._trial_windows())
        self.events.append(dict(kind='task_end', trial=self.completed+1,
            source_index=self.latest_index+1, physical_event_time_verified=False))
        self.completed += 1
        if self.completed < self.total:
            self._prompt('rest')
        else:
            self.alignment, self.calibration_info = fit_alignment(np.stack(self.windows), n_trials=self.total)
            self._finish_hand_calibration()
            self.phase, self.prompt_id, self.remaining = 'complete', None, 0.
            self.history = self.history[:, :0]
            self.indices = self.indices[:0]

    def _finish_hand_calibration(self):
        if not self.hand_supervised:
            return
        self.hand_bias = self._hand_bias_from(np.stack(self.windows),self.tasks)

    def _hand_bias_from(self,windows,tasks):
        import torch
        from .labelled_hand_calibration import fit_hand_bias
        values = band_windows(np.asarray(windows), [[4.,16.],[16.,40.]])
        with torch.inference_mode():
            batches = [self.models['baseline'](torch.from_numpy(values[start:start+54])).softmax(1).cpu().numpy()
                       for start in range(0, len(values), 54)]
            p = np.concatenate(batches).reshape(self.total,3,6).mean(1)
        return fit_hand_bias(p,tasks,rounds=self.rounds)

    def _infer(self, key, values):
        import torch
        with torch.inference_mode():
            return self.models[key](torch.from_numpy(values)).softmax(1).cpu().numpy().mean(0)

    def predict_windows(self, values):
        if self.alignment is None:
            raise RuntimeError('本次会话尚未完成校准。')
        p = self._infer('baseline', band_windows(values, [[4., 16.], [16., 40.]]))
        expert = band_windows(values, [[8., 13.], [13., 30.]])
        aligned = np.einsum('cd,nbdt->nbct', self.alignment, expert, optimize=True).astype(np.float32)
        q = self._infer('expert', aligned)
        gated = int(p.argmax()) >= 4
        routed = p.copy()
        if gated:
            # Native candidate hard-routes only feet/tongue. Conditional pair
            # probabilities keep that decision; they are explicitly identified
            # in model_selection.json, not presented as original six-class p.
            mass = float(q[4:6].sum())
            routed[:4] = 0
            routed[4:6] = q[4:6]/mass if mass > 0 else .5
        elif self.hand_supervised:
            if self.hand_bias is None:
                raise RuntimeError('有标签动作校准尚未完成。')
            from .labelled_hand_calibration import route_hand
            routed = route_hand(p,self.hand_bias)
        self.last_details = dict(baseline_probabilities=p.tolist(), expert_probabilities=q.tolist(),
            pair_routed=gated, hand_pair_routed=bool(self.hand_supervised and not gated),
            hand_bias=self.hand_bias.tolist() if self.hand_bias is not None else None,
            probability_semantics='hard_route_pair_conditional' if gated else
                'labelled_hand_pair_conditional' if self.hand_supervised else 'baseline')
        return routed

    def process(self, raw, indices):
        with self.lock:
            values, retained = self.stream.process(raw, indices)
            self.latest_index = int(indices[-1])
            keep = max(2000, int((self.imagery_seconds+2)*self.stream.output_fs))
            self.history = np.concatenate((self.history, values[SCALP]), axis=1)[:, -keep:]
            self.indices = np.concatenate((self.indices, retained))[-keep:]
            if self.alignment is None:
                self._calibrate_step()
                return None
            now = self.latest_index/self.stream.fs
            if len(self.indices) < 700 or now-self.last_prediction < .1-1e-9:
                return None
            self.last_window = self.stream.window_bounds(self.indices[-700:], indices[-1])
            values = self._notched(np.stack([self.history[:, -700:-200],
                self.history[:, -600:-100], self.history[:, -500:]]))
            self.last_prediction = now
            return now, self.predict_windows(values)

    def save_calibration(self, session):
        import json
        with self.lock:
            complete = self.alignment is not None and (not self.hand_supervised or self.hand_bias is not None)
            metadata = dict(state=self.snapshot_calibration(), events=copy.deepcopy(self.events),
                protocol=self.protocol,
                info=self.calibration_info, complete=complete,
                labels_used=self.hand_supervised, physical_event_time_verified=False,
                timing_source='reused_calibration' if self.reused_from else 'browser_render_ack_at_latest_received_source_index',
                audio_enabled=self.audio_enabled,
                stream_adaptation_validated=False, reusable_in_another_session=complete,
                reuse_requires_user_selection=True, reuse_requires_same_setup=True,
                reused_from=copy.deepcopy(self.reused_from))
            if self.hand_supervised:
                metadata['labelled_hand_calibration'] = dict(method='pair_logodds_bias',ridge=1.,alpha=1.,
                    task_labels=self.tasks[:self.completed],
                    bias=self.hand_bias.tolist() if self.hand_bias is not None else None,
                    labels_used=True,later_labels_used=False,independent_online_validation=False)
            (session/'calibration.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
            if self.windows:
                np.savez_compressed(session/'calibration_arrays.npz', windows=np.stack(self.windows),
                    alignment=self.alignment if self.alignment is not None else np.empty((0, 0)),
                    task_labels=np.asarray(self.tasks[:self.completed],dtype=np.int64))
