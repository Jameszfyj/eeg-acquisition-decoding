"""Explicit reuse of complete EA calibration, bound to model and EEG setup."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import copy
import hashlib
import json
from pathlib import Path
import re
from zipfile import BadZipFile

import numpy as np

from .four_class_online import digest
from .workbench_settings import calibration_protocol


def input_contract(cfg):
    device = cfg['device']
    return dict(channels=device['channels'], sample_rate=device['sample_rate'],
                unit=device['unit'], reference=device.get('reference'),
                preprocess=cfg['preprocess'], window_seconds=cfg['window']['seconds'],
                calibration_protocol=calibration_protocol(cfg))


def device_contract(device):
    info = device.get('info', {})
    result = {key: str(info.get(key, '')) for key in ('type', 'serial', 'cap')}
    if not all(result.values()) or not info.get('labels_verified'):
        raise ValueError('原校准的设备身份不完整，请重新校准。')
    result.update(channels=device['labels'], sample_rate=device['settings']['fs'], unit=device['unit'])
    return result


def verify_device(expected, current):
    actual = device_contract(current)
    if (any(actual[key] != expected[key] for key in ('type', 'serial', 'cap', 'sample_rate', 'unit'))
            or sorted(actual['channels']) != sorted(expected['channels'])):
        raise ValueError('当前脑电设备、脑电帽或通道配置与上次校准不同，请选择重新校准。')


class CalibrationStore:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def load(self, session_id, selection, cfg):
        from .six_class_improved import fit_alignment
        protocol = calibration_protocol(cfg)
        n_trials = protocol['trials']
        if (not isinstance(session_id, str)
                or not re.fullmatch(r'\d{8}_\d{6}_[0-9a-f]{8}_[\w-]+', session_id)):
            raise ValueError('上次校准记录无效，请重新选择。')
        source = (self.root/session_id).resolve()
        if source.parent != self.root:
            raise ValueError('校准记录不在本工作台目录中。')
        if (source/'calibration_data.json').is_file():
            from .shared_calibration import SharedCalibrationStore
            shared = SharedCalibrationStore(self.root).load(session_id, cfg, protocol)
            return dict(summary=dict(shared['summary'], model_id=selection['model_id'],
                        model_sha256=selection['sha256']), data_summary=shared['summary'],
                        shared_raw=True, root=str(self.root))
        try:
            record = json.loads((source/'model_selection.json').read_text(encoding='utf-8'))
            calibration = json.loads((source/'calibration.json').read_text(encoding='utf-8'))
            device = json.loads((source/'device.json').read_text(encoding='utf-8'))
            if record['model_id'] != selection['model_id']:
                return self._shared(session_id, record, selection, cfg)
            if (record['participant_id'] != selection['participant_id']
                    or record['model_id'] != selection['model_id']
                    or record['sha256'] != selection['sha256']
                    or input_contract(record['config']) != input_contract(cfg)):
                raise ValueError('上次校准的被试、模型或输入配置不匹配，请重新校准。')
            if (calibration.get('complete') is not True or calibration.get('reused_from')
                    or calibration.get('state', {}).get('ready') is not True
                    or calibration['state'].get('completed') != n_trials
                    or calibration['state'].get('total') != n_trials):
                raise ValueError(f'该记录没有完整的{n_trials}次校准，请重新校准。')
            with np.load(source/'calibration_arrays.npz', allow_pickle=False) as arrays:
                windows = arrays['windows'].copy()
                alignment = arrays['alignment'].copy()
                task_labels = arrays['task_labels'].copy() if 'task_labels' in arrays else None
            computed, info = fit_alignment(windows, n_trials=n_trials)
            if (alignment.shape != (30, 30) or not np.isfinite(alignment).all()
                    or not np.allclose(alignment, computed, rtol=1e-9, atol=1e-10)):
                raise ValueError('保存的校准矩阵核验失败，请重新校准。')
            hand_bias = None
            if selection.get('hand_supervised_calibration'):
                from .labelled_hand_calibration import validate_labels
                task_labels = validate_labels(task_labels, rounds=protocol['rounds'])
                hand = calibration.get('labelled_hand_calibration',{})
                hand_bias = np.asarray(hand.get('bias'),dtype=float)
                if (calibration.get('labels_used') is not True or hand.get('method') != 'pair_logodds_bias'
                        or hand.get('ridge') != 1. or hand.get('alpha') != 1.
                        or hand.get('labels_used') is not True or hand.get('later_labels_used') is not False
                        or not np.array_equal(task_labels,np.asarray(hand.get('task_labels')))
                        or hand_bias.shape != (2,) or not np.isfinite(hand_bias).all() or np.max(abs(hand_bias))>2.):
                    raise ValueError('上次记录缺少完整的有标签动作校准，请重新校准。')
            right_metadata = None
            if selection.get('right_hand_review'):
                from .right_hand_calibration import validate_saved
                right_metadata = calibration.get('right_hand_review_calibration')
                state = validate_saved(right_metadata,selection['right_hand_review_config'],task_labels)
                if not np.allclose(state['hand_bias'],hand_bias,rtol=1e-9,atol=1e-10):
                    raise ValueError('右手复核与手部校准参数不一致，请重新校准。')
            contract = device_contract(device)
            summary = dict(id=session_id, participant_id=record['participant_id'],
                model_id=record['model_id'], model_sha256=record['sha256'],
                saved_at=datetime.fromtimestamp((source/'calibration.json').stat().st_mtime,
                    timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M'),
                calibration_session_id=calibration['state']['session_id'],
                calibration_trials=n_trials, calibration_rounds=protocol['rounds'],
                device=contract, calibration_sha256=digest(source/'calibration.json'),
                arrays_sha256=digest(source/'calibration_arrays.npz'))
            return dict(summary=summary, windows=windows, alignment=alignment, info=info,
                        task_labels=task_labels,hand_bias=hand_bias,right_hand_review_calibration=right_metadata)
        except (OSError, KeyError, TypeError, ValueError, EOFError, BadZipFile) as exc:
            raise ValueError(f'无法使用该校准记录：{exc}') from exc

    def _shared(self, session_id, record, selection, cfg):
        """Share verified calibration EEG, never another model's fitted offsets."""
        from .six_class_online import LABELLED_ID, RIGHT_REVIEW_ID, RIGHT_COMBINED_ID, prepare
        family = {LABELLED_ID, RIGHT_REVIEW_ID, RIGHT_COMBINED_ID}
        if (record['model_id'] not in family or selection['model_id'] not in family
                or record['participant_id'] != selection['participant_id']
                or input_contract(record['config']) != input_contract(cfg)):
            raise ValueError('上次校准的被试、模型或输入配置不匹配，请重新校准。')
        source_selection, source_cfg = prepare(record['participant_id'],record['model_id'])
        # Recursion enters the strict same-model branch, validating the original
        # weights, labels, EA matrix and every original fitted parameter first.
        original = self.load(session_id,source_selection,source_cfg)
        classes = lambda c: [(x['id'],x['name'],x.get('task_text')) for x in c['classes']]
        if (input_contract(source_cfg) != input_contract(cfg) or classes(source_cfg) != classes(cfg)
                or classes(record['config']) != classes(source_cfg)):
            raise ValueError('校准任务、通道、单位、采样率或预处理不一致，不能共用校准脑电。')
        from .six_class_improved import ImprovedSixClassDecoder, load_checkpoint
        if selection.get('right_hand_review'):
            from .six_class_right_review import RightHandSixClassDecoder as Decoder, load_checkpoint
        else:
            Decoder = ImprovedSixClassDecoder
        payload,models = load_checkpoint(selection['model'],selection['sha256'])
        decoder = Decoder(payload,models,cfg)
        decoder.windows = list(original['windows'].copy())
        decoder.tasks = original['task_labels'].tolist()
        decoder.alignment = original['alignment'].copy()
        decoder.calibration_info = copy.deepcopy(original['info'])
        decoder._finish_hand_calibration()
        right = None
        if selection.get('right_hand_review'):
            from .right_hand_calibration import serialize_state
            right = dict(method='right_hand_review_v1',config=copy.deepcopy(selection['right_hand_review_config']),
                         task_labels=decoder.tasks,offsets=serialize_state(decoder.right_state),
                         labels_used=True,later_labels_used=False)
        parameters = dict(hand_bias=decoder.hand_bias.tolist(),right_hand_review_calibration=right)
        parameter_hash = hashlib.sha256(json.dumps(parameters,sort_keys=True,ensure_ascii=False,
                                                    allow_nan=False).encode()).hexdigest()
        summary = dict(original['summary'],source_model_id=record['model_id'],
                       source_model_sha256=record['sha256'],model_id=selection['model_id'],
                       model_sha256=selection['sha256'],shared_calibration_input=True,
                       target_parameters_sha256=parameter_hash)
        return dict(summary=summary,windows=original['windows'],alignment=original['alignment'],
                    info=original['info'],task_labels=original['task_labels'],hand_bias=decoder.hand_bias.copy(),
                    right_hand_review_calibration=right)

    def history(self, selection, cfg):
        available, unavailable = [], []
        for path in sorted(self.root.glob('*/calibration.json'),reverse=True):
            try:
                record = json.loads((path.parent/'model_selection.json').read_text(encoding='utf-8'))
                if record.get('participant_id') != selection['participant_id']:
                    continue
                calibration = json.loads(path.read_text(encoding='utf-8'))
                if calibration.get('reused_from'):
                    continue  # Show the original once, not every reuse session.
                expected = calibration_protocol(cfg)['trials']
                if calibration.get('complete') is not True or calibration.get('state',{}).get('completed') != expected:
                    raise ValueError(f"校准未完成或轮数不匹配（{calibration.get('state',{}).get('completed',0)}/{expected}）")
                available.append(self.load(path.parent.name,selection,cfg)['summary'])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                unavailable.append(dict(id=path.parent.name,reason=str(exc)))
        return dict(available=available,unavailable=unavailable)

    def latest(self, selection, cfg):
        for path in sorted(self.root.glob('*/calibration.json'), reverse=True):
            try:
                return self.load(path.parent.name, selection, cfg)['summary']
            except ValueError:
                continue
        return None
