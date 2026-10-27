"""Small session-local action-bias calibration; backbone and EA remain frozen."""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

PAIRS=((0,2),(1,3))


def validate_labels(labels):
    raw=np.asarray(labels)
    if (raw.shape!=(18,) or not np.issubdtype(raw.dtype,np.integer)
            or np.any(raw<0) or np.any(raw>5) or np.bincount(raw,minlength=6).tolist()!=[3]*6):
        raise ValueError('有标签校准需要18次完整提示任务，每类3次。')
    return raw.astype(np.int64)


def fit_hand_bias(probabilities,labels):
    labels=validate_labels(labels)
    p=np.asarray(probabilities)
    if p.shape!=(18,6) or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1.,atol=1e-5):
        raise ValueError('有标签校准的任务概率无效。')
    biases=[]
    for fist,rotation in PAIRS:
        select=np.isin(labels,[fist,rotation])
        logit=np.log(np.maximum(p[select,rotation],1e-12))-np.log(np.maximum(p[select,fist],1e-12))
        target=(labels[select]==rotation).astype(float)
        bias=brentq(lambda b:np.sum(expit(logit+b)-target)+b,-20.,20.)
        biases.append(float(np.clip(bias,-2.,2.)))
    return np.array(biases)


def route_hand(p,bias):
    """Return a conditional pair probability only for the parent's inferred hand side."""
    p=np.asarray(p);bias=np.asarray(bias)
    if p.shape!=(6,) or bias.shape!=(2,) or not np.isfinite(p).all() or not np.isfinite(bias).all():
        raise ValueError('动作校准参数无效。')
    parent=int(p.argmax())
    if parent>=4:
        raise ValueError('手部校准不能处理脚舌分支。')
    side=parent%2
    fist,rotation=PAIRS[side]
    logit=np.log(max(float(p[rotation]),1e-12))-np.log(max(float(p[fist]),1e-12))
    probability=float(expit(logit+bias[side]))
    routed=np.zeros(6,dtype=p.dtype)
    routed[fist],routed[rotation]=1-probability,probability
    return routed
