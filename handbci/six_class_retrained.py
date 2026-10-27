"""Inference/calibration math for the known two-IFNet checkpoint format.

The stream and catalog integration lives in retrained_runtime. Inputs retain the original FIR,
30-channel ordering and three notched 2s windows per task.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
import torch

from .model import IFNetClassifier
from .four_class_online import digest, CHANNELS, SCALP
from .six_class_online import ARCH, NAMES
from .six_class_improved import band_windows
from .feet_tongue_adaptation import (fit_alignment, fit_residual,
                                     residual_probabilities, validate_trials, metrics)

FORMAT='handbci.p01_six_retrained_20261027.v1'
METHODS=('ea_only','head_ridge10','head_ridge1')


class RetrainedSixClassModel:
    def __init__(self, path, expected_sha256=None, device='cpu'):
        self.path=Path(path).resolve();self.sha256=digest(self.path)
        if expected_sha256 is not None and self.sha256!=expected_sha256:
            raise ValueError('新模型哈希不匹配。')
        p=torch.load(self.path,map_location='cpu',weights_only=True)
        if (p.get('format')!=FORMAT or p.get('class_names')!=NAMES
                or p.get('channel_names')!=[CHANNELS[i] for i in SCALP]
                or p.get('online_control_validated') is not False
                or p.get('future_calibration')!=dict(rounds=20,tasks=120,labels_required=True)):
            raise ValueError('模型格式、类别或20轮校准约定不一致。')
        self.models={};self.device=device;self.payload=p
        for key in ('baseline','expert'):
            if p[key]['architecture']!=ARCH:raise ValueError('新模型网络结构不一致。')
            model=IFNetClassifier(**ARCH).to(device)
            model.load_state_dict(p[key]['state_dict'],strict=True)
            model.eval();model.requires_grad_(False);self.models[key]=model
        self.alignment=None;self.state=None;self.calibration=None

    def features(self, windows, expert=False):
        x=np.asarray(windows,np.float32)
        if x.ndim!=4 or x.shape[1:]!=(3,30,500) or not len(x) or not np.isfinite(x).all():
            raise ValueError('需要有限的[任务数,3,30,500]原预处理和陷波窗口。')
        bands=[[8.,13.],[13.,30.]] if expert else [[4.,16.],[16.,40.]]
        values=band_windows(x.reshape(-1,30,500),bands)
        if expert:
            if self.alignment is None:raise RuntimeError('尚未完成新会话20轮校准。')
            values=np.einsum('cd,nbdt->nbct',self.alignment,values,optimize=True).astype(np.float32)
        model=self.models['expert' if expert else 'baseline'];features=[];logits=[]
        with torch.inference_mode():
            for start in range(0,len(values),128):
                t=torch.from_numpy(values[start:start+128]).to(self.device)
                f=model.stem(t.flatten(1,2)).flatten(1)
                logits.append(model.classifier(f).cpu().numpy());features.append(f.cpu().numpy())
        return np.concatenate(features).reshape(-1,3,features[0].shape[-1]),np.concatenate(logits).reshape(-1,3,6)

    def predict(self, windows):
        if self.calibration is None:raise RuntimeError('新模型必须先完成独立的新会话20轮校准。')
        _,z=self.features(windows);p=softmax(z,axis=-1).mean(1)
        f,z=self.features(windows,expert=True);pair=residual_probabilities(f,z,self.state)
        gate=p.argmax(1)>=4;out=p.copy();out[gate]=0;out[gate,4:6]=pair[gate]
        return dict(probabilities=out,pair_probabilities=pair,
                    baseline_probabilities=p,feet_tongue_gate=gate)

    def calibrate(self, windows, labels, rounds, source_provenance):
        x,y,r=validate_trials(windows,labels,rounds)
        if (source_provenance.get('participant_id')!='P01'
                or source_provenance.get('class_names')!=NAMES
                or source_provenance.get('channel_names')!=self.payload['channel_names']
                or not source_provenance.get('source_sha256')):
            raise ValueError('需要P01原六类、原通道顺序及校准源文件哈希。')
        training_hashes=set(self.payload['training']['source_sha256'].values())
        if set(source_provenance['source_sha256'].values()) & training_hashes:
            raise ValueError('校准记录已参与新模型训练，需要另采的新会话。')
        # Work on a separate wrapper so failed calibration never partially
        # replaces an existing ready state. Model weights remain frozen.
        worker=object.__new__(type(self));worker.__dict__=self.__dict__.copy()
        predictions={m:[] for m in METHODS};targets=[]
        for low in (1,6,11,16):
            held=(r>=low)&(r<low+5);train=~held;pair=(y[train]>=4)
            worker.alignment=fit_alignment(x[train]);f,z=worker.features(x[train][pair],expert=True)
            states={'ea_only':None}
            for method,ridge in [('head_ridge10',10.),('head_ridge1',1.)]:
                states[method]=fit_residual(f,z[:,:,5]-z[:,:,4],y[train][pair],ridge)
            worker.calibration={'temporary_fold':True}
            for method in METHODS:
                worker.state=states[method];predictions[method].append(worker.predict(x[held]))
            targets.append(y[held])
        scores={m:metrics(np.concatenate(targets),{k:np.concatenate([o[k] for o in predictions[m]])
                                                  for k in predictions[m][0]}) for m in METHODS}
        selected=max(METHODS,key=lambda m:(scores[m]['pair_end_to_end_accuracy'],m=='ea_only',-scores[m]['expert_nll']))
        alignment=fit_alignment(x);worker.alignment=alignment;state=None
        if selected!='ea_only':
            pair=y>=4;f,z=worker.features(x[pair],expert=True)
            state=fit_residual(f,z[:,:,5]-z[:,:,4],y[pair],10. if selected=='head_ridge10' else 1.)
        self.alignment=alignment;self.state=state
        self.calibration=dict(method=selected,rounds=20,n_tasks=120,n_supervised_pair_tasks=40,
            CV=scores,model_sha256=self.sha256,source_provenance=source_provenance,
            calibration_data_accuracy_is_not_independent_test=True,online_control_validated=False)
        return self.calibration

    def save_calibration(self,path):
        if self.calibration is None:raise RuntimeError('尚未校准。')
        arrays=dict(metadata=np.asarray(json.dumps(self.calibration,ensure_ascii=False)),alignment=self.alignment)
        if self.state is not None:arrays.update({k:self.state[k] for k in ('mean','scale','delta')})
        with Path(path).open('xb') as stream:np.savez_compressed(stream,**arrays)

    def restore_calibration(self,path):
        with np.load(path,allow_pickle=False) as a:
            meta=json.loads(str(a['metadata']))
            if meta.get('model_sha256')!=self.sha256 or meta.get('method') not in METHODS or meta.get('n_tasks')!=120:
                raise ValueError('校准记录不属于这份新模型。')
            alignment=a['alignment'].copy();state=None
            if (alignment.shape!=(30,30) or not np.isfinite(alignment).all()
                    or not np.allclose(alignment,alignment.T,atol=1e-8)
                    or np.linalg.eigvalsh(alignment).min()<=0):raise ValueError('校准对齐矩阵无效。')
            if meta['method']!='ea_only':
                state={k:a[k].copy() for k in ('mean','scale','delta')}
                d=self.models['expert'].classifier[0].in_features
                if (state['mean'].shape!=(d,) or state['scale'].shape!=(d,) or state['delta'].shape!=(d+1,)
                        or any(not np.isfinite(v).all() for v in state.values()) or np.any(state['scale']<=0)):
                    raise ValueError('脚舌校准分类层无效。')
        self.alignment,self.state,self.calibration=alignment,state,meta
