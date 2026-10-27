"""Task-grouped EA and regularized feet/tongue calibration math.

Contains no personal weights, data paths, device access or model registration.
"""
import numpy as np
from scipy import signal
from scipy.optimize import minimize
from scipy.special import expit, softmax

def validate_trials(windows, labels, rounds, expected_rounds=20):
    x, y, r = np.asarray(windows), np.asarray(labels), np.asarray(rounds)
    if (x.shape != (expected_rounds * 6, 3, 30, 500)
            or not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all()):
        raise ValueError('需要完整任务窗口：[轮数×6,3,30,500]，且全部为有限浮点数。')
    if (y.shape != (len(x),) or r.shape != y.shape
            or not np.issubdtype(y.dtype, np.integer)
            or not np.issubdtype(r.dtype, np.integer)
            or not np.array_equal(np.unique(r), np.arange(1, expected_rounds + 1))
            or np.any(np.diff(r) < 0)):
        raise ValueError('任务标签和轮次必须为整数，轮次按时间排列且连续。')
    for rr in range(1, expected_rounds + 1):
        if sorted(y[r == rr].tolist()) != list(range(6)):
            raise ValueError('每轮必须恰好包含原六类各一次；不能混入双手新范式。')
    return x, y.astype(np.int64), r.astype(np.int64)

def fit_alignment(windows):
    """Same broadband/shrinkage formula as parent; any complete training rounds."""
    values = np.asarray(windows).reshape(-1, 30, 500)
    if len(values) < 18 or not np.isfinite(values).all():
        raise ValueError('信号对齐数据不足或无效。')
    filtered = signal.sosfiltfilt(signal.butter(4, [4., 40.], fs=250.,
                                              btype='bandpass', output='sos'), values, axis=-1)
    centered = filtered - filtered.mean(-1, keepdims=True)
    covariance = np.einsum('nct,ndt->cd', centered, centered) / (len(values) * 499)
    covariance = .9 * covariance + .1 * np.trace(covariance) / 30 * np.eye(30)
    eig, vectors = np.linalg.eigh(covariance)
    if eig.min() <= 0:
        raise ValueError('校准协方差不是正定矩阵。')
    return (vectors / np.sqrt(eig)) @ vectors.T

def fit_residual(features, logits, labels, ridge):
    """Convex binary residual, one task weight shared by its three windows.

    Label 4 -> 0 (feet), label 5 -> 1 (tongue). Feature moments use only
    calibration feet/tongue tasks. The fixed parent logit is the anchor.
    """
    f, z, y = np.asarray(features, float), np.asarray(logits, float), np.asarray(labels)
    if (f.ndim != 3 or f.shape[1] != 3 or z.shape != f.shape[:2]
            or y.shape != (len(f),) or not set(y.tolist()) == {4, 5}
            or not np.isfinite(f).all() or not np.isfinite(z).all()
            or not np.isfinite(ridge) or ridge <= 0):
        raise ValueError('脚舌分类层需要成组特征、原始logit，以及脚舌两类标签。')
    mean = f.mean((0, 1))
    scale = np.maximum(f.std((0, 1)), 1e-5) * np.sqrt(f.shape[-1])
    design = np.concatenate([(f - mean) / scale,
                             np.ones((*f.shape[:2], 1))], axis=-1).reshape(-1, f.shape[-1] + 1)
    target = np.repeat(y == 5, 3).astype(float)
    counts = np.bincount(y - 4, minlength=2)
    # Three overlapping windows collectively contribute one balanced task.
    weights = np.repeat(len(y) / (2 * counts[y - 4]) / 3, 3)
    parent = z.ravel()
    penalty = np.r_[np.full(f.shape[-1], ridge), 1.]

    def objective(delta):
        score = parent + design @ delta
        loss = np.sum(weights * (np.logaddexp(0, score) - target * score))
        loss += .5 * np.sum(penalty * delta ** 2)
        grad = design.T @ (weights * (expit(score) - target)) + penalty * delta
        return float(loss), grad

    result = minimize(objective, np.zeros(design.shape[1]), jac=True,
                      method='L-BFGS-B', options=dict(maxiter=300, ftol=1e-10, gtol=1e-7))
    if not result.success or not np.isfinite(result.fun):
        raise ValueError('脚舌分类层优化未收敛：' + str(result.message))
    return dict(mean=mean, scale=scale, delta=result.x,
                iterations=int(result.nit), objective=float(result.fun))

def residual_probabilities(features, logits, state):
    score = np.asarray(logits, float).copy()
    if state is not None:
        normalized = (np.asarray(features) - state['mean']) / state['scale']
        correction = normalized @ state['delta'][:-1] + state['delta'][-1]
        score[:, :, 4] -= .5 * correction
        score[:, :, 5] += .5 * correction
    # Match parent: average SIX-class softmax, then normalize the pair. A zero
    # residual must reproduce the parent even when other-class mass varies.
    pair = softmax(score, axis=-1).mean(1)[:, 4:6]
    return pair / pair.sum(1, keepdims=True)

def metrics(labels, output):
    y = np.asarray(labels)
    pred = output['probabilities'].argmax(1)
    pair_mask = np.isin(y, [4, 5])
    pair_y = y[pair_mask] - 4
    pair_p = output['pair_probabilities'][pair_mask]
    pair_pred = pair_p.argmax(1)
    matrix = np.zeros((2, 2), int)
    np.add.at(matrix, (pair_y, pair_pred), 1)
    support = matrix.sum(1)
    recall = matrix.diagonal() / support
    return dict(n_tasks=len(y), six_class_accuracy=float(np.mean(y == pred)),
                n_feet_tongue=int(pair_mask.sum()), expert_balanced_accuracy=float(recall.mean()),
                expert_nll=float(-np.log(np.maximum(pair_p[np.arange(len(pair_y)), pair_y], 1e-12)).mean()),
                expert_confusion_matrix=matrix.tolist(),
                feet_as_tongue=int(matrix[0, 1]), tongue_as_feet=int(matrix[1, 0]),
                feet_recall=float(recall[0]), tongue_recall=float(recall[1]),
                pair_end_to_end_accuracy=float(np.mean(pred[pair_mask] == y[pair_mask])),
                pair_routed_to_hand=int((~output['feet_tongue_gate'][pair_mask]).sum()))
