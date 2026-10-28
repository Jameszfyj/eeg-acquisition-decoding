"""Trusted model entries and streaming adapter for the right-hand research models."""
from __future__ import annotations
import copy, json
from pathlib import Path
import numpy as np
from .four_class_online import digest
from .six_class_online import BATCH, NAMES, ARCH, LIMITS, RIGHT_REVIEW_ID, RIGHT_COMBINED_ID
from .six_class_improved import ImprovedSixClassDecoder, load_parent_models, band_windows
from .right_hand_calibration import calibrate, predict, serialize_state, validate_saved, state_equal

OUT = BATCH/'右手专项复核_20261024'
PLAN_SHA256 = 'e6db60d3140173efcc99ac063dac0d1ff3cafca487d96766f467b0377f0e9b8a'
SELECTION_SHA256 = '1970a516067d26e61a5664f058507501906b46dafbe06fafcc377ddb6a19fc15'
GATE_ROI = ['FC1','FC2','FC5','FC6','C3','Cz','C4','CP1','CP2','CP5','CP6']
LEFT_ROI = ['FC1','FC5','C3','Cz','CP1','CP5']
ENTRIES = {
    RIGHT_REVIEW_ID: dict(title='六组模型 · 右手复核（实验）',file='right_hand_review_research.pt',
        sha256='cb19530a292c33ddba3bda571fd173a5d08172b8bc30fe310f913dc2afbc7d25',
        config=dict(name='gate_0.7',threshold=.7,action='none',alpha=0.,family='gate')),
    RIGHT_COMBINED_ID: dict(title='六组模型 · 右手复核＋动作专属（实验）',file='right_hand_combined_research.pt',
        sha256='720469e424643f3a4259c294fa0f19e8ecd954d0991f9cc4f7492a70d6562c42',
        config=dict(name='both_wide_0.5_0.7',threshold=.7,action='wide',alpha=.5,family='both')),
}


def catalog():
    return [dict(participant_id='P01',model_id=key,name=e['title'],source='p01_six_acp_observation',
        n_classes=6,n_channels=30,requires_calibration=True,hand_supervised_calibration=True,
        right_hand_review=True,experimental=True,sha256=e['sha256'],**LIMITS)
        for key,e in ENTRIES.items() if (OUT/e['file']).is_file()]


def prepare(participant, model_id):
    from .six_class_labelled import prepare as prepare_labelled
    if model_id not in ENTRIES:
        raise ValueError('未知的右手专项模型。')
    entry = ENTRIES[model_id]
    if digest(OUT/entry['file']) != entry['sha256'] or digest(OUT/'experiment_plan.json') != PLAN_SHA256:
        raise ValueError('右手专项模型或训练方案校验失败。')
    selection,cfg = prepare_labelled(participant)
    cfg['description'] = entry['title']+'；3轮提示校准后持续解码，在线改善尚待独立验证。'
    cfg['session_metadata'].update(right_hand_review=True,right_hand_review_config=copy.deepcopy(entry['config']))
    selection.update(model_id=model_id,name=entry['title'],model=str(OUT/entry['file']),sha256=entry['sha256'],
        parent_plan_sha256=selection['plan_sha256'],plan_sha256=PLAN_SHA256,right_hand_review=True,
        right_hand_review_config=copy.deepcopy(entry['config']),input_note=cfg['description'],
        probability_semantics='conditional within reviewed hand side; original feet/tongue expert conditional; right review score recorded separately')
    return selection,cfg


def load_checkpoint(path, expected_digest):
    import torch
    from .model import IFNetClassifier
    resolved = Path(path).resolve()
    matched = [(key,e) for key,e in ENTRIES.items() if resolved == (OUT/e['file']).resolve()]
    if len(matched) != 1:
        raise ValueError('右手专项模型路径不符。')
    model_id,entry = matched[0]
    if expected_digest != entry['sha256'] or digest(path) != entry['sha256']:
        raise ValueError('右手专项模型校验失败。')
    if digest(OUT/'experiment_plan.json') != PLAN_SHA256 or digest(OUT/'locked_selection.json') != SELECTION_SHA256:
        raise ValueError('右手专项训练或选择方案不符。')
    raw = torch.load(path,map_location='cpu',weights_only=True)
    if (raw.get('format') != 'handbci.right_hand_review.offline.v1' or raw.get('config') != entry['config']
            or raw.get('plan_sha256') != PLAN_SHA256 or raw.get('selection_sha256') != SELECTION_SHA256
            or raw.get('primary_source_accepted') is not True or raw.get('independent_test_of_this_refit') is not False
            or any(raw.get(k) is not v for k,v in LIMITS.items())):
        raise ValueError('右手专项模型的校准规则或限制标记不符。')
    parent = copy.deepcopy(raw['parent'])
    from .six_class_labelled import PLAN, PLAN_SHA256 as PARENT_PLAN_SHA256
    if digest(PLAN) != PARENT_PLAN_SHA256:
        raise ValueError('六组来源方案校验失败。')
    source = json.loads(PLAN.read_text(encoding='utf-8'))
    expected_h5 = {s:source['source_sha256'][s+'/data.h5'] for s in source['sessions']}
    if (parent.get('plan_sha256') != PARENT_PLAN_SHA256 or parent.get('training',{}).get('n_trials') != 629
            or parent['training'].get('source_h5_sha256') != expected_h5
            or raw.get('training_sessions') != source['sessions']):
        raise ValueError('右手专项的六组基础权重来源不符。')
    parent['_hand_supervised_calibration'] = True
    parent['_right_hand_review_config'] = copy.deepcopy(entry['config'])
    parent['_right_hand_review_model_id'] = model_id
    parent,models = load_parent_models(parent)
    needed = [('review_gate',raw['gate'],len(GATE_ROI),'gate',629)]
    if entry['config']['action'] == 'wide':
        needed.append(('review_action',raw['actions']['wide'],len(LEFT_ROI),'wide',210))
    for key,c,n,recipe,count in needed:
        architecture = dict(ARCH,n_channels=n,n_classes=2)
        if (c.get('architecture') != architecture or c.get('recipe') != recipe
                or c.get('plan_sha256') != PLAN_SHA256 or c.get('training',{}).get('n_tasks') != count
                or c['training'].get('epochs') != 88 or c['training'].get('seed') != 42):
            raise ValueError('右手专项分支的结构或训练来源不符。')
        model = IFNetClassifier(**architecture).eval()
        model.load_state_dict(c['state_dict'],strict=True)
        models[key] = model
    return parent,models


class RightHandSixClassDecoder(ImprovedSixClassDecoder):
    def __init__(self,payload,models,cfg):
        super().__init__(payload,models,cfg)
        self.review_config = copy.deepcopy(payload.get('_right_hand_review_config'))
        metadata = cfg.get('session_metadata',{})
        if (not metadata.get('right_hand_review') or not self.hand_supervised
                or self.review_config != metadata.get('right_hand_review_config')):
            raise ValueError('右手专项模型与运行模式不一致。')
        self.right_state = None

    def _window_probabilities(self,key,values):
        import torch
        with torch.inference_mode():
            return self.models[key](torch.from_numpy(values)).softmax(1).cpu().numpy()

    def _state_from(self,windows,labels):
        windows = np.asarray(windows)
        if windows.shape != (54,30,500) or not np.isfinite(windows).all():
            raise ValueError('右手专项校准需要18次完整任务的54个窗口。')
        wide = band_windows(windows,[[4.,16.],[16.,40.]])
        names = self.payload['channel_names']
        p = self._window_probabilities('baseline',wide).reshape(18,3,6).mean(1)
        gate = self._window_probabilities('review_gate',wide[:,:,[names.index(c) for c in GATE_ROI]]).reshape(18,3,2).mean(1)[:,1]
        action = {}
        if self.review_config['action'] == 'wide':
            action['wide'] = self._window_probabilities('review_action',wide[:,:,[names.index(c) for c in LEFT_ROI]]).reshape(18,3,2).mean(1)[:,1]
        return calibrate(p,gate,action,labels)

    def _finish_hand_calibration(self):
        state = self._state_from(np.stack(self.windows),self.tasks)
        self.hand_bias = state['hand_bias'].copy()
        self.right_state = state

    def snapshot_calibration(self):
        result = super().snapshot_calibration()
        result['ready'] = result['ready'] and (self.right_state is not None or self.calibration_disabled)
        result['right_hand_review'] = True
        return result

    def predict_windows(self,values):
        if self.calibration_disabled and self.right_state is None:
            result = super().predict_windows(values)
            self.last_details.update(right_hand_review=False, right_hand_review_skipped='online_calibration_disabled')
            return result
        if self.right_state is None:
            raise RuntimeError('右手专项校准尚未完成。')
        routed = super().predict_windows(values)
        p = np.asarray(self.last_details['baseline_probabilities'])
        if int(p.argmax()) >= 4:
            self.last_details.update(right_hand_review=True,right_hand_promoted=False,right_hand_review_score=None)
            return routed
        names = self.payload['channel_names']
        wide = band_windows(values,[[4.,16.],[16.,40.]])
        gate = self._infer('review_gate',wide[:,:,[names.index(c) for c in GATE_ROI]])[1]
        action = {}
        if self.review_config['action'] == 'wide':
            action['wide'] = np.array([self._infer('review_action',wide[:,:,[names.index(c) for c in LEFT_ROI]])[1]])
        pred,info = predict(p[None],np.array([gate]),action,np.array([int(p.argmax())]),self.right_state,self.review_config)
        output = routed.copy()
        if info['inferred_right'][0]:
            probability = float(info['right_rotation_probability'][0])
            output[:] = 0.
            output[1],output[3] = 1-probability,probability
        if int(output.argmax()) != int(pred[0]):
            raise RuntimeError('右手专项概率与动作决策不一致。')
        self.last_details.update(right_hand_review=True,right_hand_promoted=bool(info['promoted'][0]),
            right_hand_review_score=float(info['review_score'][0]),
            before_right_review=int(info['baseline'][0]),right_hand_gate_probability=float(gate),
            probability_semantics='reviewed_hand_pair_conditional')
        return output

    def save_calibration(self,session):
        with self.lock:
            super().save_calibration(session)
            path = session/'calibration.json'
            metadata = json.loads(path.read_text(encoding='utf-8'))
            metadata['complete'] = metadata['complete'] and self.right_state is not None
            metadata['right_hand_review_calibration'] = dict(method='right_hand_review_v1',config=self.review_config,
                task_labels=self.tasks[:self.completed],offsets=serialize_state(self.right_state) if self.right_state is not None else None,
                labels_used=True,later_labels_used=False)
            path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')

    def restore_calibration(self,saved):
        if self.calibration_disabled:
            raise ValueError('在线调试期间不从历史脑电重新拟合校准参数。')
        with self.lock:
            # Validate and recompute everything before the parent activates EA.
            if self.latest_index != -1 or self.alignment is not None:
                raise ValueError('只能在新连接接收脑电前载入上次校准。')
            stored = validate_saved(saved.get('right_hand_review_calibration'),self.review_config,saved.get('task_labels'))
            state = self._state_from(saved['windows'],saved['task_labels'])
            if not state_equal(state,stored):
                raise ValueError('右手复核校准重算核验失败，请重新校准。')
            super().restore_calibration(saved)
            self.right_state = state
