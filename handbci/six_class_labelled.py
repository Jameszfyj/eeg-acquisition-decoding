"""Explicit experimental entry for all-six-session weights plus labeled prefix calibration."""
from __future__ import annotations
import copy,json
from .four_class_online import CHANNELS,digest
from .six_class_online import (BATCH,NAMES,LIMITS,LABELLED_ID,LABELLED_MODEL,LABELLED_SHA256)

PLAN=BATCH/'六组联合训练_20261024/experiment_plan.json'
PLAN_SHA256='1e5609d7ba720443f7b4d8072a88b3a29a2b9ab94f7ce79cadc0a3d784f7cdda'
TITLE='六组模型 · 3轮有标签校准（实验）'


def prepare(participant):
    if participant!='P01':raise ValueError('六组权重归属P01，不能替换为其它被试。')
    if digest(LABELLED_MODEL)!=LABELLED_SHA256 or digest(PLAN)!=PLAN_SHA256:
        raise ValueError('六组有标签校准模型或来源方案校验失败。')
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    if len(plan['sessions'])!=6 or plan['class_names']!=NAMES:
        raise ValueError('六组模型的来源或类别不符。')
    metadata=[]
    for sid in plan['sessions']:
        for filename in ('session.json','data.h5'):
            relative=sid+'/'+filename
            if digest(BATCH/relative)!=plan['source_sha256'][relative]:
                raise ValueError('六组模型原始来源校验失败：'+relative)
        meta=json.loads((BATCH/sid/'session.json').read_text(encoding='utf-8'))
        if (not meta['complete'] or meta['participant_id']!='P01' or meta['device']['synthetic']
                or [t['text'] for t in meta['protocol']['tasks']]!=NAMES):
            raise ValueError('六组来源记录与模型约定不一致。')
        metadata.append(meta)
    cfg=copy.deepcopy(metadata[0]['config'])
    cfg['description']='六组离线模型的有标签动作校准实验；新会话和在线效果尚待独立验证。'
    cfg['device'].update(source='neuracle_acp',host='',port=4097,channels=list(CHANNELS))
    cfg['preprocess']=dict(output_rate=250.,lowpass_hz=100.,highpass_hz=4.,numtaps=201,trim_seconds=1.)
    cfg['window'].update(seconds=2.,stride_seconds=.1,task_offset_seconds=.6)
    cfg['session_metadata']=dict(participant_id='P01',presentation='continuous_six_class_observation',
        unprompted=False,calibration_required=True,calibration_timing='browser_render_ack_to_received_source_index; not hardware verified',
        source_quality=[m['quality'] for m in metadata],hand_supervised_calibration=True,**LIMITS)
    selection=dict(participant_id='P01',model_id=LABELLED_ID,name=TITLE,source='p01_six_acp_observation',
        model=str(LABELLED_MODEL),sha256=LABELLED_SHA256,plan_sha256=PLAN_SHA256,
        requires_calibration=True,hand_supervised_calibration=True,calibration_labels_used=True,
        calibration_trials=18,calibration_rounds=3,experimental=True,synthetic_model=False,
        n_classes=6,n_channels=30,sample_rate=1000.,driver='mock',physical_action=False,
        independent_test_of_this_refit=False,input_note=cfg['description'],
        probability_semantics='conditional within original inferred hand side after labeled bias; original feet/tongue expert conditional',**LIMITS)
    return selection,cfg
