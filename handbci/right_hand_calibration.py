"""Session-local right-hand review rules, matched to the frozen offline study."""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from .labelled_hand_calibration import fit_hand_bias, validate_labels


def logit(p):
    return np.log(np.maximum(p, 1e-12))-np.log(np.maximum(1-p, 1e-12))


def fit_offset(values, targets):
    values, targets = np.asarray(values), np.asarray(targets)
    if (len(values) != len(targets) or not np.isfinite(values).all()
            or not np.any(targets == 0) or not np.any(targets == 1)):
        raise ValueError('右手校准的标签或概率不完整。')
    return float(np.clip(brentq(lambda b: np.sum(expit(values+b)-targets)+b, -30., 30.), -2., 2.))


def calibrate(p, gate, action, labels):
    labels = validate_labels(labels)
    p, gate = np.asarray(p), np.asarray(gate)
    if (p.shape != (18, 6) or gate.shape != (18,)
            or not np.isfinite(p).all() or not np.isfinite(gate).all()
            or np.any(gate < 0) or np.any(gate > 1)
            or any(np.shape(v) != (18,) or not np.isfinite(v).all()
                   or np.any(v < 0) or np.any(v > 1) for v in action.values())):
        raise ValueError('右手校准需要完整18次任务概率。')
    right = np.isin(labels, [1, 3])
    return dict(hand_bias=fit_hand_bias(p, labels),
        mass_bias=fit_offset(logit(p[:, [1, 3]].sum(1)), right.astype(float)),
        gate_bias=fit_offset(logit(gate), right.astype(float)),
        action_bias={k: fit_offset(logit(v[right]), (labels[right] == 3).astype(float)) for k, v in action.items()},
        n_prefix_tasks=18, class_counts=np.bincount(labels, minlength=6).tolist())


def predict(p, gate, action, current, state, config):
    p, gate, current = np.asarray(p), np.asarray(gate), np.asarray(current)
    base = current.copy()
    for side, (a, b) in enumerate([(0, 2), (1, 3)]):
        selected = np.isin(current, [a, b])
        z = np.log(np.maximum(p[:, b], 1e-12))-np.log(np.maximum(p[:, a], 1e-12))
        base[selected] = np.where(z[selected]+state['hand_bias'][side] > 0., b, a)
    output = base.copy()
    right = np.isin(base, [1, 3])
    review = np.zeros(len(p), bool)
    score = .25*expit(logit(p[:, [1, 3]].sum(1))+state['mass_bias']) + .75*expit(logit(gate)+state['gate_bias'])
    if config['threshold'] is not None:
        review = np.isin(base, [0, 2]) & (p.max(1) < .8) & (score >= config['threshold'])
        right |= review
    z = np.log(np.maximum(p[:, 3], 1e-12))-np.log(np.maximum(p[:, 1], 1e-12))
    qr = expit(z+state['hand_bias'][1])
    if config['action'] != 'none':
        key, alpha = config['action'], config['alpha']
        qa = expit(logit(action[key])+state['action_bias'][key])
        qr = (1-alpha)*qr+alpha*qa
    output[right] = np.where(qr[right] > .5, 3, 1)
    return output, dict(promoted=review, baseline=base, right_rotation_probability=qr,
                        review_score=score, inferred_right=right)


def serialize_state(state):
    return dict(hand_bias=np.asarray(state['hand_bias']).tolist(), mass_bias=float(state['mass_bias']),
        gate_bias=float(state['gate_bias']), action_bias={k: float(v) for k, v in state['action_bias'].items()},
        n_prefix_tasks=state['n_prefix_tasks'], class_counts=state['class_counts'])


def validate_saved(metadata, config, labels):
    labels = validate_labels(labels)
    if (not isinstance(metadata, dict) or metadata.get('method') != 'right_hand_review_v1'
            or metadata.get('config') != config or metadata.get('labels_used') is not True
            or metadata.get('later_labels_used') is not False
            or not np.array_equal(np.asarray(metadata.get('task_labels')), labels)):
        raise ValueError('右手复核校准记录不完整或与模型不一致，请重新校准。')
    state = metadata.get('offsets', {})
    actions = {} if config['action'] == 'none' else {config['action']}
    if (not isinstance(state, dict) or state.get('n_prefix_tasks') != 18 or state.get('class_counts') != [3]*6
            or not isinstance(state.get('action_bias'), dict) or set(state['action_bias']) != set(actions)):
        raise ValueError('右手复核校准参数不完整，请重新校准。')
    bias = np.asarray(state.get('hand_bias'), dtype=float)
    scalars = np.asarray([state.get('mass_bias'), state.get('gate_bias'), *state['action_bias'].values()], dtype=float)
    if (bias.shape != (2,) or not np.isfinite(bias).all() or np.max(abs(bias)) > 2.
            or not np.isfinite(scalars).all() or np.max(abs(scalars)) > 2.):
        raise ValueError('右手复核校准参数无效，请重新校准。')
    return state


def state_equal(a, b):
    if set(a['action_bias']) != set(b['action_bias']):
        return False
    x = [*a['hand_bias'], a['mass_bias'], a['gate_bias'], *[a['action_bias'][k] for k in sorted(a['action_bias'])]]
    y = [*b['hand_bias'], b['mass_bias'], b['gate_bias'], *[b['action_bias'][k] for k in sorted(b['action_bias'])]]
    return bool(np.allclose(x, y, rtol=1e-9, atol=1e-10))
