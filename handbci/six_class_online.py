"""Online adapter for the 2026-10-23 P01 offline IFNet prototype.

Never converts this research checkpoint into a production/validated checkpoint.
ACP observation reuses the existing stream path, with the original window notch.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from scipy import signal

from .four_class_online import CHANNELS, SCALP, FourClassDecoder, digest
from .preprocess import filter_bank
from .workbench_settings import local_path

ROOT = Path(__file__).resolve().parents[1]
BATCH = local_path('research_data_root', ROOT/'六分类数据_20261023_常规_三组60轮')
BASE = BATCH/'六分类训练'
MODEL = BASE/'models/six_class_ifnet_final.pt'
MODEL_ID = 'p01-six-20261023-offline-observation'
MODEL_SHA256 = 'ef42056d4c021b88c932a51145d54f5731a90d17dcddbdec028897d5c2ada3eb'
IMPROVED_ID = 'p01-six-20261023-pair-router'
IMPROVED_MODEL = BATCH/'六分类改进对照/完整六分类候选/six_class_pair_router_candidate.pt'
IMPROVED_SHA256 = '1883a8b69dd37a1fcd678a7aa83c8fe25e08d0342f33491464df560e97a116f4'
LABELLED_ID = 'p01-six-20261024-labelled-calibration-experiment'
LABELLED_MODEL = local_path('labelled_checkpoint', BATCH/'握拳旋臂专项_20261024/有标签动作校准/small_gain_calibrated_candidate.pt')
LABELLED_SHA256 = 'c196b4c27b81567f6888ef1e8ed9d46338301b2dae6d146eabc547ba0a4433b6'
RIGHT_REVIEW_ID = 'p01-six-20261024-right-hand-review-experiment'
RIGHT_COMBINED_ID = 'p01-six-20261024-right-hand-combined-experiment'
SIX_MODEL_IDS = (MODEL_ID, IMPROVED_ID, LABELLED_ID, RIGHT_REVIEW_ID, RIGHT_COMBINED_ID)
PLAN_SHA256 = 'a99888b61c761f743268528886628df31d9c6f8a6b837b9e975c9a4ce1c6bdc6'
NAMES = ['左握拳', '右握拳', '左旋臂', '右旋臂', '双脚', '舌头']
SESSIONS = ['20261023_165123_6d75003f_P01', '20261023_172914_41c4436c_P01',
            '20261023_180058_bd101ff2_P01']
ARCH = dict(n_channels=30, n_times=500, n_classes=6, radix=2, out_planes=64,
            kernel_size=63, patch_size=125)
LIMITS = dict(offline_prototype_only=True, online_control_validated=False,
              physical_event_time_verified=False)
NOTE = '2026-10-23真实ACP数据训练的离线原型；支持网页模拟或机械手实时输出，在线控制效果及物理事件时间仍待验证。'


def catalog():
    if not MODEL.is_file():
        return []
    items = [dict(participant_id='P01', model_id=MODEL_ID, name='基础模型 · 六分类 IFNet',
                 source='p01_six_acp_observation', n_classes=6, n_channels=30,
                 requires_calibration=False, sha256=MODEL_SHA256, **LIMITS)]
    if IMPROVED_MODEL.is_file():
        items.append(dict(participant_id='P01', model_id=IMPROVED_ID,
            name='改进模型 · IFNet＋脚舌复判（需校准）', source='p01_six_acp_observation',
            n_classes=6, n_channels=30, requires_calibration=True, sha256=IMPROVED_SHA256, **LIMITS))
    if LABELLED_MODEL.is_file():
        from .six_class_labelled import TITLE
        items.append(dict(participant_id='P01',model_id=LABELLED_ID,name=TITLE,
            source='p01_six_acp_observation',n_classes=6,n_channels=30,requires_calibration=True,
            hand_supervised_calibration=True,experimental=True,sha256=LABELLED_SHA256,**LIMITS))
    from .six_class_right_review import catalog as right_catalog
    items.extend(right_catalog())
    return items


def provenance():
    path = BASE/'experiment_plan.json'
    if digest(path) != PLAN_SHA256:
        raise ValueError('六类训练方案发生变化，拒绝使用未核对的来源。')
    plan = json.loads(path.read_text(encoding='utf-8'))
    if plan['sessions'] != SESSIONS or plan['class_names'] != NAMES or plan['architecture'] != ARCH:
        raise ValueError('六类训练方案的类别、会话或结构不一致。')
    metadata = []
    for session in SESSIONS:
        for filename in ('session.json', 'data.h5'):
            relative = f'{session}/{filename}'
            if digest(BATCH/relative) != plan['source_sha256'].get(relative):
                raise ValueError('六类原采集文件校验失败：'+relative)
        meta = json.loads((BATCH/session/'session.json').read_text(encoding='utf-8'))
        if (meta['participant_id'] != 'P01' or meta['device']['synthetic'] is not False
                or meta['source'] != 'neuracle_acp' or meta['device']['unit'] != 'uV'
                or meta['config']['device']['sample_rate'] != 1000
                or [t['text'] for t in meta['protocol']['tasks']] != NAMES):
            raise ValueError('六类原采集的被试、来源或类别不匹配。')
        metadata.append(meta)
    return plan, metadata


def prepare(participant, model_id):
    if model_id in (RIGHT_REVIEW_ID, RIGHT_COMBINED_ID):
        from .six_class_right_review import prepare as prepare_right
        return prepare_right(participant, model_id)
    if model_id == LABELLED_ID:
        from .six_class_labelled import prepare as prepare_labelled
        return prepare_labelled(participant)
    if participant != 'P01' or model_id not in (MODEL_ID, IMPROVED_ID):
        raise ValueError('昨天六类权重归属P01，不能替换为其它被试。')
    if digest(MODEL) != MODEL_SHA256:
        raise ValueError('昨天六类权重校验失败，不能自动改用其它模型。')
    plan, metadata = provenance()
    cfg = copy.deepcopy(metadata[0]['config'])
    # Preserve original class IDs/names/action=custom and all unresolved quality flags.
    if [c['id'] for c in cfg['classes']] != list(range(6)) or [c['name'] for c in cfg['classes']] != NAMES:
        raise ValueError('原采集类别编号与六类权重顺序不一致。')
    cfg['description'] = NOTE
    cfg['device'].update(source='neuracle_acp', host='', port=4097, channels=list(CHANNELS))
    cfg['preprocess'] = dict(output_rate=250., lowpass_hz=100., highpass_hz=4., numtaps=201, trim_seconds=1.)
    cfg['window'].update(seconds=2., stride_seconds=.1, task_offset_seconds=.6)
    source_metadata = cfg['session_metadata']
    cfg['session_metadata'] = dict(participant_id='P01', source_session_metadata=source_metadata,
        presentation='continuous_six_class_observation', unprompted=True,
        input_contract='ACP_uV_by_electrode_name; original FIR, 50Hz window notch and filter bank',
        source_quality=[m['quality'] for m in metadata], **LIMITS)
    selection = dict(participant_id='P01', model_id=MODEL_ID, name=catalog()[0]['name'],
        source='p01_six_acp_observation', model=str(MODEL), sha256=MODEL_SHA256,
        plan_sha256=PLAN_SHA256, synthetic_model=False, n_classes=6, n_channels=30,
        sample_rate=1000., driver='mock', physical_action=False,
        independent_test_of_this_refit=False, input_note=NOTE, **LIMITS)
    selection['requires_calibration'] = model_id == IMPROVED_ID
    if model_id == IMPROVED_ID:
        if digest(IMPROVED_MODEL) != IMPROVED_SHA256:
            raise ValueError('改进模型校验失败。')
        selection.update(model_id=IMPROVED_ID, name=catalog()[1]['name'], model=str(IMPROVED_MODEL),
            sha256=IMPROVED_SHA256, calibration_trials=18, calibration_rounds=3,
            input_note='改进模型先完成3轮18次任务的本次会话校准，再持续解码；在线适配效果待实测。',
            probability_semantics='hand: baseline; feet/tongue: expert conditional after hard routing')
        cfg['session_metadata'].update(unprompted=False, calibration_required=True,
            calibration_timing='browser_render_ack_to_received_source_index; not hardware verified')
    return selection, cfg


def load_checkpoint(path=MODEL, expected_digest=MODEL_SHA256):
    import torch
    from .model import IFNetClassifier
    if Path(path).resolve() != MODEL.resolve() or expected_digest != MODEL_SHA256 or digest(path) != MODEL_SHA256:
        raise ValueError('六类权重路径或校验值不符。')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload.get('format') != 'handbci.six_class_ifnet.offline.v1'
            or payload.get('architecture') != ARCH or payload.get('class_names') != NAMES
            or payload.get('channel_names') != [CHANNELS[i] for i in SCALP]
            or payload.get('plan_sha256') != PLAN_SHA256
            or any(payload.get(k) is not v for k,v in LIMITS.items())
            or payload.get('training', {}).get('training_sessions') != SESSIONS
            or payload.get('training', {}).get('independent_test_of_this_refit') is not False):
        raise ValueError('六类权重的原始约定或限制标记不符。')
    plan = json.loads((BASE/'experiment_plan.json').read_text(encoding='utf-8'))
    if digest(BASE/'experiment_plan.json') != PLAN_SHA256 or payload['preprocessing'] != plan['preprocessing']:
        raise ValueError('六类预处理与训练方案不一致。')
    model = IFNetClassifier(**ARCH).eval()
    model.load_state_dict(payload['state_dict'], strict=True)
    return payload, model


class SixClassDecoder(FourClassDecoder):
    def __init__(self, payload, model, cfg):
        super().__init__(payload, model, stream_config=cfg)
        self.notch = signal.iirnotch(50., 30., fs=250.)

    def window_inputs(self):
        values = signal.filtfilt(*self.notch, self.history[None], axis=-1).astype(np.float32)
        return filter_bank(values, 250.)
