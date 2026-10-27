# -*- coding: utf-8 -*-
"""Six-class source training on IFNetRTG-L, with the project's staged recipe.

The architecture and the three training stages are the same as
``jixieshou_training/mi_bench/reg_calib6h.py``:

    pretrain CE (early stop on inner-valid trial bACC)
    -> SVM-initialised d-dimensional bottleneck
    -> freeze the backbone, train the head
    -> full fine-tune for a fixed number of epochs

The studio keeps it self-contained so it can run from a plain checkout, but
the recipe (width 16, d=8, batches, LRs) is unchanged.  Outer evaluation and
the final deployment fit are separate: the reported metrics always come from
a model that never saw the held-out sessions, while the saved final model is
refit on every selected session.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import contracts as C
from .preprocess import WindowBank

WIDTH = 16


# ---------------------------------------------------------------------------
# cargame / torch plumbing
# ---------------------------------------------------------------------------
def cargame_root() -> Path:
    candidates = [
        os.environ.get("CARGAME_ROOT"),
        "F:/project/cargame",
        "/root/autodl-fs/cargame",
        str(Path(__file__).resolve().parents[2] / "cargame"),
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "app/training/trainer_protocol.py").is_file():
            return Path(candidate)
    raise RuntimeError("找不到 cargame 源码；请设置环境变量 CARGAME_ROOT")


def _torch():
    import torch
    return torch


def device_auto() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def create_model(fs: float, n_p: int, n_c: int, n_class: int, *,
                 seed: int, device: str, hidden: int = 8):
    import torch
    root = cargame_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from app.training.trainer_protocol import create_protocol, _set_seed

    _set_seed(seed)
    proto = create_protocol(
        "IFNetRTG-L", fs=float(fs), n_p=int(n_p), n_c=int(n_c),
        n_class=int(n_class), checkpoint_name="mi_studio", seed=int(seed),
        batch_size=128, device=torch.device(device), compact_width=WIDTH,
    )
    proto.base_preprocessor = None
    proto.base_standardlizer_name = "None"
    proto.label_names = {i: f"class {i}" for i in range(n_class)}
    proto.studio_hidden = int(hidden)
    return proto


def _aggregate_proba(proba: np.ndarray, trial_idx: np.ndarray, n_trials: int) -> np.ndarray:
    out = np.zeros((n_trials, proba.shape[1]), np.float64)
    count = np.zeros(n_trials, np.float64)
    np.add.at(out, trial_idx, proba)
    np.add.at(count, trial_idx, 1.0)
    return out / np.maximum(count, 1.0)[:, None]


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, np.float64)
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return (z / z.sum(axis=1, keepdims=True)).astype(np.float64)


def trial_metrics(y_true: np.ndarray, pred: np.ndarray, n_class: int = C.OUTPUT_COUNT) -> dict[str, Any]:
    y_true = np.asarray(y_true, np.int64)
    pred = np.asarray(pred, np.int64)
    support = [int((y_true == c).sum()) for c in range(n_class)]
    recall = []
    for c in range(n_class):
        mask = y_true == c
        recall.append(float((pred[mask] == c).mean()) if mask.any() else None)
    present = [c for c in range(n_class) if support[c] > 0]
    bacc = float(np.mean([recall[c] for c in present])) if present else None
    acc = float((pred == y_true).mean()) if len(y_true) else None
    confusion = np.zeros((n_class, n_class), np.int64)
    for t, p in zip(y_true, pred):
        confusion[int(t), int(p)] += 1
    return {
        "n_trials": int(len(y_true)),
        "support": support,
        "evaluated_classes": present,
        "missing_classes": [c for c in range(n_class) if support[c] == 0],
        "balanced_accuracy": bacc,
        "accuracy": acc,
        "per_class_recall": recall,
        "confusion": confusion.tolist(),
        "note": ("balanced_accuracy is over the classes present in this split; "
                 "missing classes are N/A, not zero"),
    }


# ---------------------------------------------------------------------------
# training stages (same recipe as reg_calib6h, parameterised)
# ---------------------------------------------------------------------------
def fit_ce(proto, X: np.ndarray, y: np.ndarray, X_va: np.ndarray, y_va: np.ndarray,
           trial_idx_va: np.ndarray, n_trials_va: int, *, max_epochs: int,
           patience: int, batch: int, seed: int, log: Callable[[str], None] = print) -> dict:
    import torch
    from sklearn.metrics import balanced_accuracy_score

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_input = proto._model_fit_transform(X, y)
    proto.model_input_shape = tuple(int(x) for x in train_input.shape[1:])
    proto.model = proto._build_model(proto.model_input_shape).to(proto.device)
    x = torch.from_numpy(np.ascontiguousarray(train_input, np.float32)).float()
    yt = torch.from_numpy(np.asarray(y, np.int64)).long()
    ds = torch.utils.data.TensorDataset(x, yt)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)
    opt = proto._make_optimizer(proto.model, proto.effective_lr, proto.weight_decay)
    va_input = np.ascontiguousarray(proto._model_transform(X_va), np.float32)

    best = {"score": -np.inf, "epoch": None, "state": None}
    patience_cnt = 0
    history = []
    stop = "max_epochs"
    for epoch in range(1, max_epochs + 1):
        proto.model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(proto.device)
            yb = yb.to(proto.device)
            xb, _, _ = proto._augment_batch(xb, None, yb)
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(proto.model(xb), yb)
            loss.backward()
            opt.step()
            proto._on_optimizer_step_end(proto.model)
            losses.append(float(loss.detach().cpu()))
        proto._on_train_epoch_end(epoch)
        proto.model.eval()
        with torch.no_grad():
            va_out = proto._eval_forward(
                torch.from_numpy(va_input).float().to(proto.device)).detach().cpu().numpy()
        pred = _aggregate_proba(softmax(va_out), trial_idx_va, n_trials_va).argmax(1)
        y_trial = np.zeros(n_trials_va, np.int64)
        for w_i, t_i in enumerate(trial_idx_va):
            y_trial[int(t_i)] = int(y_va[w_i])
        score = float(balanced_accuracy_score(y_trial, pred))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_bacc": score})
        if score > best["score"]:
            best = {"score": score, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in proto.model.state_dict().items()}}
            patience_cnt = 0
        else:
            patience_cnt += 1
        if epoch % max(1, max_epochs // 10) == 0 or epoch == max_epochs:
            log(f"      pretrain {epoch}/{max_epochs}: loss={np.mean(losses):.3f} "
                f"valid_bacc={score:.3f} best={best['score']:.3f}@{best['epoch']}")
        if patience_cnt >= patience:
            stop = "early_stopping"
            break
    if best["state"] is None:
        raise RuntimeError("pretrain 没有产生任何检查点")
    proto.model.load_state_dict(best["state"])
    return {"history": history,
            "stop_reason": stop, "epochs_completed": len(history),
            "best_epoch": best["epoch"], "best_valid_bacc": best["score"],
            "n_train_windows": int(len(X)), "n_valid_windows": int(len(X_va))}


def svm_init_d(model, z_tr: np.ndarray, y_tr: np.ndarray, d: int, seed: int = 0) -> dict:
    import torch
    from sklearn.svm import LinearSVC
    from app.training.trainer_models import LinearWithConstraint

    present = sorted({int(v) for v in np.unique(y_tr)})
    svm = LinearSVC(C=1.0, max_iter=5000, class_weight="balanced").fit(z_tr, y_tr)
    _u, _sv, vt = np.linalg.svd(svm.coef_.astype(np.float64), full_matrices=False)
    r = min(d, vt.shape[0])
    in_features = int(z_tr.shape[1])
    B = np.zeros((in_features, d), np.float64)
    B[:, :r] = vt[:r].T
    z2 = z_tr @ B[:, :r]
    A = np.column_stack([z2, np.ones(len(z2))])
    Y = np.zeros((len(z2), C.OUTPUT_COUNT), np.float64)
    Y[np.arange(len(y_tr)), y_tr] = 1.0
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    W2 = np.zeros((C.OUTPUT_COUNT, d), np.float64)
    W2[:, :r] = coef[:r].T
    b2 = coef[r].copy()
    for c in range(C.OUTPUT_COUNT):
        if c not in present:
            W2[c, :] = 0.0
            b2[c] = -10.0

    l1 = LinearWithConstraint(in_features, d, max_norm=0.5)
    l2 = torch.nn.Linear(d, C.OUTPUT_COUNT, bias=True)
    with torch.no_grad():
        w1 = torch.from_numpy(B.T.copy()).float()
        nrm = w1.norm(dim=1, keepdim=True).clamp_min(1e-8)
        l1.weight.copy_(w1 * (0.5 / nrm).clamp(max=1.0))
        l2.weight.copy_(torch.from_numpy(W2).float())
        l2.bias.copy_(torch.from_numpy(b2).float())
    dev = next(model.parameters()).device
    model.classifier = torch.nn.Sequential(l1, l2).to(dev)
    return {"in_features": in_features, "hidden": d, "svm_rank": int(r),
            "present_classes": present,
            "absent_classes": [c for c in range(C.OUTPUT_COUNT) if c not in present]}


def head_only(proto, X: np.ndarray, y: np.ndarray, epochs: int, batch: int,
              lr: float, seed: int) -> None:
    import torch
    model = proto.model
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("classifier.")
    x = torch.from_numpy(np.ascontiguousarray(proto._model_transform(X), np.float32)).float().to(proto.device)
    yt = torch.from_numpy(np.asarray(y, np.int64)).long().to(proto.device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    torch.manual_seed(seed)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(x), device=x.device)
        for i in range(0, len(x), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(model(x[idx]), yt[idx]).backward()
            opt.step()
    for p in model.parameters():
        p.requires_grad = True


def finetune_fixed(proto, X: np.ndarray, y: np.ndarray, epochs: int, batch: int,
                   seed: int) -> None:
    import torch
    model = proto.model
    x = torch.from_numpy(np.ascontiguousarray(proto._model_transform(X), np.float32)).float().to(proto.device)
    yt = torch.from_numpy(np.asarray(y, np.int64)).long().to(proto.device)
    opt = proto._make_optimizer(model, proto.effective_lr, proto.weight_decay)
    torch.manual_seed(seed * 7919 + 13)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x), device=x.device)
        for i in range(0, len(x), batch):
            idx = perm[i:i + batch]
            xb, yb = x[idx], yt[idx]
            xb, _, _ = proto._augment_batch(xb, None, yb)
            opt.zero_grad()
            torch.nn.functional.cross_entropy(model(xb), yb).backward()
            opt.step()
            proto._on_optimizer_step_end(model)
        proto._on_train_epoch_end(epoch)


def pooled_features(proto, X: np.ndarray, batch: int = 512) -> np.ndarray:
    import torch
    model = proto.model
    model.eval()
    band = np.ascontiguousarray(proto._model_transform(X), np.float32)
    parts = []
    with torch.no_grad():
        for start in range(0, len(band), batch):
            xb = torch.from_numpy(band[start:start + batch]).float().to(proto.device)
            parts.append(model.forward_features(xb).detach().cpu().numpy())
    return np.concatenate(parts, 0).astype(np.float64)


def latent_features(proto, X: np.ndarray) -> np.ndarray:
    """Pooled features projected through the constrained bottleneck."""
    pooled = pooled_features(proto, X)
    head = _normalise_head(extract_head(proto))
    return pooled @ head["W1"].T + head["b1"]


def staged_train(proto, X_tr, y_tr, X_va, y_va, trial_idx_va, n_trials_va, *,
                 pre_epochs: int, head_epochs: int, ft_epochs: int, patience: int,
                 batch: int, seed: int, hidden: int,
                 log: Callable[[str], None] = print) -> dict:
    started = time.time()
    log(f"    [1/4] pretrain CE {pre_epochs} epochs (patience {patience})")
    fit = fit_ce(proto, X_tr, y_tr, X_va, y_va, trial_idx_va, n_trials_va,
                 max_epochs=pre_epochs, patience=patience, batch=batch, seed=seed, log=log)
    log(f"    [2/4] SVM init d={hidden}")
    z_tr = pooled_features(proto, X_tr)
    init = svm_init_d(proto.model, z_tr, y_tr, hidden, seed)
    log(f"    [3/4] head-only {head_epochs} epochs (in={init['in_features']}, "
        f"rank={init['svm_rank']})")
    head_only(proto, X_tr, y_tr, head_epochs, batch, 1e-2, seed)
    log(f"    [4/4] fine-tune {ft_epochs} epochs")
    finetune_fixed(proto, X_tr, y_tr, ft_epochs, batch, seed)
    return {"seconds": round(time.time() - started, 1), "head_init": init,
            "pretrain": {k: fit[k] for k in ("stop_reason", "epochs_completed",
                                             "best_epoch", "best_valid_bacc")},
            "n_train_windows": int(len(X_tr)), "n_train_trials": int(len(np.unique(trial_idx_va)))}


# ---------------------------------------------------------------------------
# artifact handling
# ---------------------------------------------------------------------------
def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalise_head(head: dict[str, Any]) -> dict[str, np.ndarray]:
    W1 = np.asarray(head["W1"], np.float64)
    nrm = np.linalg.norm(W1, axis=1, keepdims=True)
    W1 = W1 * np.minimum(1.0, 0.5 / np.maximum(nrm, 1e-8))
    return {"W1": W1, "b1": np.asarray(head["b1"], np.float64),
            "W2": np.asarray(head["W2"], np.float64),
            "b2": np.asarray(head["b2"], np.float64)}


def extract_head(proto) -> dict[str, np.ndarray]:
    model = proto.model
    # a forward pass applies the constraint in place
    import torch
    with torch.no_grad():
        dummy = torch.zeros((1, *proto.model_input_shape), dtype=torch.float32).to(proto.device)
        model(dummy)
    return {"W1": model.classifier[0].weight.detach().cpu().numpy().astype(np.float64),
            "b1": model.classifier[0].bias.detach().cpu().numpy().astype(np.float64),
            "W2": model.classifier[-1].weight.detach().cpu().numpy().astype(np.float64),
            "b2": model.classifier[-1].bias.detach().cpu().numpy().astype(np.float64)}


def build_payload(proto, *, outputs: list[dict[str, Any]], channels: list[str],
                  arch: dict[str, Any], scaler: dict[str, Any],
                  studio: dict[str, Any], head: dict[str, Any] | None = None,
                  calibration: dict[str, Any] | None = None) -> dict[str, Any]:
    head = _normalise_head(head or extract_head(proto))
    return {
        "format": C.MODEL_FORMAT,
        "model_labels": [int(o["label_id"]) for o in outputs],
        "class_names": [str(o["name"]) for o in outputs],
        "outputs": outputs,
        "channels": list(channels),
        "arch": dict(arch),
        "scaler": dict(scaler),
        "head": {k: np.asarray(v).tolist() for k, v in head.items()},
        "state_dict": {k: v.detach().cpu() for k, v in proto.model.state_dict().items()},
        "calibration": calibration,
        "studio": studio,
    }


def save_model(payload: dict[str, Any], path: str | Path) -> Path:
    import torch
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    sidecar = out.with_suffix(".json")
    meta = {k: v for k, v in payload.items() if k not in ("state_dict",)}
    meta["checkpoint"] = out.name
    meta["sha256"] = sha256_file(out)
    sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_payload(path: str | Path) -> dict[str, Any]:
    import torch
    p = Path(path)
    if p.is_dir():
        for candidate in ("model.pt", "six_platform_d8_s0.pt", "checkpoint.pt"):
            if (p / candidate).is_file():
                p = p / candidate
                break
        else:
            spec = p / "spec.json"
            if spec.is_file():
                data = json.loads(spec.read_text(encoding="utf-8"))
                if data.get("checkpoint"):
                    p = p / data["checkpoint"]
    if p.suffix.lower() == ".onnx":
        raise C.ContractError(
            "校准需要能导出潜特征的 PyTorch 源模型；任意 logits-only ONNX 不满足本配方。")
    if not p.is_file():
        raise FileNotFoundError(f"找不到源模型 {p}")
    with torch.serialization.safe_globals([np.core.multiarray._reconstruct,np.ndarray,np.dtype,
            type(np.dtype(np.float64)),type(np.dtype(np.float32)),type(np.dtype(np.int64))]):
        payload = torch.load(p, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload or "head" not in payload:
        raise C.ContractError(f"{p} 不是 studio/六类离线模型（缺少 state_dict/head）")
    return payload


def rebuild_model(payload: dict[str, Any], device: str = "cpu"):
    import torch
    from .network import LinearWithConstraint, IFNetGClassifier, band_split_array
    from types import SimpleNamespace

    arch = payload["arch"]
    n_class = int(arch["n_class"])
    if n_class != C.OUTPUT_COUNT:
        raise C.ContractError(f"源模型有 {n_class} 维输出，studio 只支持 6 维")
    if arch.get('protocol')!='IFNetRTG-L':
        raise C.ContractError('本机兼容加载器只处理已核对的IFNetRTG-L源模型。')
    # Rebuild the same inference network without importing unrelated Python 3.11 trainers.
    proto=SimpleNamespace(device=torch.device(device))
    proto._model_transform=lambda values:band_split_array(values,float(arch['fs']),[[4.,16.],[16.,40.]],mode='concat')
    proto.model=IFNetGClassifier(n_channels=int(arch['n_c']),n_times=int(arch['n_p']),
        n_classes=n_class,radix=2,out_planes=int(arch.get('compact_width',16)),
        kernel_size=63,zero_padding=True,pool_mode='logvar_mean').to(proto.device)
    head = payload["head"]
    d = int(np.asarray(head["W1"]).shape[0])
    proj = LinearWithConstraint(int(np.asarray(head["W1"]).shape[1]), d, max_norm=0.5)
    out = torch.nn.Linear(d, n_class, bias=True)
    proto.model.classifier = torch.nn.Sequential(proj, out)
    proto.model.load_state_dict(payload["state_dict"], strict=True)
    proto.model.eval()
    return proto



def resolve_model_profile(payload: dict[str, Any], model_path: str | Path | None = None
                          ) -> tuple[dict[str, Any] | None, str]:
    """Resolve a model's preprocessing profile from validated metadata only.

    Order: studio sidecar -> payload.preprocess_profile -> sibling spec.json.
    A checkpoint with none of these is 'unknown'; the caller must ask the
    operator to confirm a migration explicitly instead of guessing a default.
    """
    studio = payload.get("studio") or {}
    prof = studio.get("profile")
    if isinstance(prof, dict) and prof.get("id") in C.PROFILES:
        return C.profile(str(prof["id"])), "studio.profile"
    if isinstance(prof, str) and prof in C.PROFILES:
        return C.profile(prof), "studio.profile"
    sid = studio.get("profile_id")
    if isinstance(sid, str) and sid in C.PROFILES:
        return C.profile(sid), "studio.profile_id"
    pp = payload.get("preprocess_profile")
    if isinstance(pp, dict) and pp.get("id") in C.PROFILES:
        return C.profile(str(pp["id"])), "preprocess_profile"
    if isinstance(pp, str) and pp in C.PROFILES:
        return C.profile(pp), "preprocess_profile"
    if model_path is not None:
        spec_path = Path(model_path).parent / "spec.json"
        if spec_path.is_file():
            try:
                spec = json.loads(spec_path.read_text(encoding="utf-8"))
            except ValueError:
                spec = {}
            prov = spec.get("provenance") or {}
            inp = spec.get("input") or {}
            pipeline = str(prov.get("pipeline") or "")
            if pipeline == "platform_replay":
                return C.profile("platform_replay_v1"), "spec.json.provenance"
            if inp.get("pipeline") == "fir" and (prov.get("pipeline") in ("", None)):
                return C.profile("platform_replay_v1"), "spec.json.input.pipeline=fir"
            if str(prov.get("pipeline")) == "l100d250h4":
                return C.profile("l100d250h4_v1"), "spec.json.provenance"
    return None, "unknown"


def model_outputs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    declared = payload.get("outputs")
    labels = payload.get("model_labels")
    names = payload.get("class_names")
    if declared is not None:
        outputs = C.validate_outputs(declared)
        if labels is not None and [int(v) for v in labels] != [o["label_id"] for o in outputs]:
            raise C.ContractError("payload.outputs 与 model_labels 冲突，拒绝猜测列顺序")
        if names is not None and [str(v) for v in names] != [o["name"] for o in outputs]:
            raise C.ContractError("payload.outputs 与 class_names 冲突，拒绝猜测列顺序")
        return outputs
    if labels is None or names is None:
        raise C.ContractError("模型缺少 6 维输出声明（outputs 或 model_labels/class_names）")
    if len(labels) != C.OUTPUT_COUNT or len(names) != C.OUTPUT_COUNT:
        raise C.ContractError("模型不是 6 维输出")
    return C.validate_outputs([
        {"index": i, "label_id": int(labels[i]), "name": str(names[i]),
         "description": ""} for i in range(C.OUTPUT_COUNT)
    ])


class SourceModel:
    """Inference + frozen features + a six-column linear calibration head."""

    def __init__(self, payload: dict[str, Any], device: str = "cpu",
                 source_path: str | Path | None = None):
        import torch
        self.payload = payload
        self.source_path = source_path
        self.torch = torch
        self.device = device
        self.outputs = model_outputs(payload)
        self.model_labels = [int(o["label_id"]) for o in self.outputs]
        self.class_names = [str(o["name"]) for o in self.outputs]
        self.channels = [str(c) for c in payload["channels"]]
        self.arch = payload["arch"]
        self.scaler = {"mean": np.asarray(payload["scaler"]["mean"], np.float64),
                       "scale": np.asarray(payload["scaler"]["scale"], np.float64)}
        self.head = _normalise_head(payload["head"])
        self._head_n = self.head
        self.proto = rebuild_model(payload, device)
        self.model = self.proto.model
        self.calibration = payload.get("calibration")

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "SourceModel":
        return cls(load_payload(path), device, source_path=path)

    def profile(self) -> tuple[dict[str, Any] | None, str]:
        return resolve_model_profile(self.payload, self.source_path)

    # ------------------------------------------------------------------ features
    def pooled(self, X: np.ndarray) -> np.ndarray:
        return pooled_features(self.proto, np.asarray(X, np.float32))

    def latent(self, X: np.ndarray) -> np.ndarray:
        pooled = self.pooled(X)
        head = self._head_n
        return pooled @ head["W1"].T + head["b1"]

    def scaled_latent(self, X: np.ndarray) -> np.ndarray:
        return (self.latent(X) - self.scaler["mean"]) / self.scaler["scale"]

    # ------------------------------------------------------------------ logits
    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        head = self._head_n
        if self.calibration is not None:
            coef = np.asarray(self.calibration["coef"], np.float64)
            intercept = np.asarray(self.calibration["intercept"], np.float64)
            if coef.shape != (C.OUTPUT_COUNT, head["W1"].shape[0]):
                raise C.ContractError(f"校准头形状 {coef.shape} 与 6 维模型不匹配")
            return (self.scaled_latent(X) @ coef.T + intercept).astype(np.float32)
        z = self.latent(X)
        return (z @ head["W2"].T + head["b2"]).astype(np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.predict_logits(X)).astype(np.float32)

    # ------------------------------------------------------------------ calibration
    def fit_calibration(self, X: np.ndarray, y: np.ndarray, *, C_value: float = 1.0) -> dict[str, Any]:
        from sklearn.linear_model import LogisticRegression

        Z = self.scaled_latent(np.asarray(X, np.float32))
        y = np.asarray(y, np.int64)
        missing = sorted(set(range(C.OUTPUT_COUNT)) - set(int(v) for v in np.unique(y)))
        if missing:
            raise C.ContractError(
                f"校准数据缺少输出列 {missing}；六路重拟合要求六类都有标签（不允许任意补零或混用 logits）")
        clf = LogisticRegression(solver="lbfgs", C=float(C_value), max_iter=5000,
                                 tol=1e-6, fit_intercept=True)
        clf.fit(Z, y, sample_weight=np.full(len(Z), 1.0 / 3.0))
        classes = [int(v) for v in clf.classes_]
        if classes != list(range(C.OUTPUT_COUNT)):
            raise C.ContractError(f"校准分类器的类别顺序异常：{classes}")
        coef = np.asarray(clf.coef_, np.float64)
        intercept = np.asarray(clf.intercept_, np.float64)
        if coef.shape != (C.OUTPUT_COUNT, Z.shape[1]):
            raise C.ContractError(f"校准系数形状 {coef.shape} 不是六列")
        return {"kind": "linear-latent-readout", "d": int(Z.shape[1]),
                "coef": coef.tolist(), "intercept": intercept.tolist(),
                "classes": classes,
                "n_windows": int(len(Z)), "n_trials": int(len(Z) // 3),
                "sample_weight": "1/3 per window",
                "C": float(C_value),
                "scaler_fit_on": self.payload["scaler"].get("fit_on", "source training features")}

    def with_calibration(self, calibration: dict[str, Any]) -> "SourceModel":
        payload = dict(self.payload)
        payload["calibration"] = calibration
        return SourceModel(payload, self.device)

    def save(self, path: str | Path, *, studio: dict[str, Any] | None = None) -> Path:
        payload = dict(self.payload)
        payload["outputs"] = self.outputs
        payload["calibration"] = self.calibration
        if studio is not None:
            payload["studio"] = studio
        payload["state_dict"] = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        payload["head"] = {k: np.asarray(v).tolist() for k, v in _normalise_head(self.head).items()}
        return save_model(payload, path)


# ---------------------------------------------------------------------------
# source training driver
# ---------------------------------------------------------------------------
def _trial_table(bank: WindowBank) -> dict[str, np.ndarray]:
    trial_keys = list(bank.meta.get("trial_keys") or [])
    if not trial_keys:
        raise C.ContractError("bank 缺少 trial_keys")
    n = len(trial_keys)
    dims = np.full(n, -1, np.int64)
    sessions = np.empty(n, dtype=object)
    for w, key in enumerate(bank.trial_key):
        idx = int(bank.trial_index[w])
        dims[idx] = int(bank.y[w])
        sessions[idx] = str(bank.session[w])
    if np.any(dims < 0):
        raise C.ContractError("bank 里有试次没有任何窗口")
    return {"key": np.asarray(trial_keys, dtype=object), "dim": dims,
            "session": np.asarray([str(s) for s in sessions], dtype=object)}


def _windows_for(bank: WindowBank, trial_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    selected = set(int(i) for i in trial_indices)
    mask = np.asarray([int(t) in selected for t in bank.trial_index], bool)
    X = bank.X[mask]
    y = bank.y[mask]
    tidx_raw = bank.trial_index[mask]
    remap = {int(v): i for i, v in enumerate(sorted(selected))}
    tidx = np.asarray([remap[int(v)] for v in tidx_raw], np.int64)
    return np.ascontiguousarray(X, np.float32), y.astype(np.int64), tidx, len(selected)


def _stratified_trial_split(dims: np.ndarray, frac: float, seed: int,
                            n_class: int = C.OUTPUT_COUNT) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    valid: list[int] = []
    for c in range(n_class):
        pool = np.flatnonzero(dims == c)
        if len(pool) < 2:
            continue
        n_valid = min(max(1, int(round(frac * len(pool)))), len(pool) - 1)
        valid.extend(rng.choice(pool, n_valid, replace=False).tolist())
    valid = np.asarray(sorted(set(valid)), np.int64)
    train = np.asarray(sorted(set(np.arange(len(dims)).tolist()) - set(valid.tolist())), np.int64)
    return train, valid


def _ensure_coverage(dims: np.ndarray, idx: np.ndarray, role: str) -> None:
    present = set(int(v) for v in np.unique(dims[idx]))
    missing = [c for c in range(C.OUTPUT_COUNT) if c not in present]
    if missing:
        raise C.ContractError(f"{role} 缺少输出列 {missing}，无法训练六类源模型")


def train_source(bank: WindowBank, *, outputs: list[dict[str, Any]], recipe: dict[str, Any],
                 eval_mode: str, holdout: list[str] | None, seed: int, device: str,
                 final_fit: bool, out_dir: str | Path,
                 log: Callable[[str], None] = print) -> dict[str, Any]:
    outputs = C.validate_outputs(outputs)
    table = _trial_table(bank)
    dims, sessions = table["dim"], table["session"]
    all_sessions = sorted(set(str(s) for s in sessions))
    if eval_mode not in ("none", "holdout", "loso"):
        raise C.ContractError(f"未知评估模式 {eval_mode}")
    if eval_mode != "none" and len(all_sessions) < 2:
        log("  [eval] 只有一份会话：不能做跨会话留出，改为会话内试次留出（仅供诊断）")
        eval_mode = "trial"
    folds: list[dict[str, Any]] = []
    if eval_mode == "loso":
        holdouts = all_sessions
    elif eval_mode == "holdout":
        holdouts = [str(h) for h in (holdout or all_sessions[:1])]
        unknown = [h for h in holdouts if h not in all_sessions]
        if unknown:
            raise C.ContractError(f"留出会话不存在：{unknown}；可选 {all_sessions}")
    else:
        holdouts = []

    def fit_one(train_idx: np.ndarray, valid_idx: np.ndarray):
        _ensure_coverage(dims, train_idx, "训练集")
        X_tr, y_tr, tidx_tr, n_tr = _windows_for(bank, train_idx)
        X_va, y_va, tidx_va, n_va = _windows_for(bank, valid_idx)
        proto = create_model(float(bank.meta["fs"]), int(bank.meta["n_time"]),
                             bank.X.shape[1], C.OUTPUT_COUNT, seed=seed,
                             device=device, hidden=int(recipe.get("hidden", 8)))
        meta = staged_train(
            proto, X_tr, y_tr, X_va, y_va, tidx_va, n_va,
            pre_epochs=int(recipe["pre_epochs"]), head_epochs=int(recipe["head_epochs"]),
            ft_epochs=int(recipe["ft_epochs"]), patience=int(recipe["patience"]),
            batch=int(recipe["batch"]), seed=seed, hidden=int(recipe.get("hidden", 8)),
            log=log)
        return proto, meta, (X_tr, y_tr, tidx_tr, n_tr), (X_va, y_va)

    trial_test_idx = np.empty(0, np.int64)
    if eval_mode == "trial":
        initial_train, trial_test_idx = _stratified_trial_split(dims, 0.30, seed + 3)
    else:
        initial_train = np.arange(len(dims), dtype=np.int64)

    eval_specs: list[tuple[str, np.ndarray, np.ndarray]] = []
    if eval_mode == "trial":
        eval_specs.append(("trial-holdout", initial_train, trial_test_idx))
    else:
        for held in holdouts:
            test_idx = np.flatnonzero(np.asarray([str(s) == held for s in sessions]))
            train_pool = np.asarray(sorted(set(range(len(dims))) - set(test_idx.tolist())), np.int64)
            eval_specs.append((held, train_pool, test_idx))

    for held, train_pool, test_idx in eval_specs:
        inner_train, inner_valid = _stratified_trial_split(dims[train_pool], 0.15, seed + 1)
        train_idx = train_pool[inner_train]
        valid_idx = train_pool[inner_valid]
        log(f"  [fold] held-out={held}: train {len(train_idx)} 试次 / inner-valid {len(valid_idx)} "
            f"/ test {len(test_idx)} 试次")
        proto, meta, _tr, _va = fit_one(train_idx, valid_idx)
        X_te, y_te, tidx_te, n_te = _windows_for(bank, test_idx)
        logits = np.asarray(proto.predict_logits_windows(X_te), np.float64)
        pred = _aggregate_proba(softmax(logits), tidx_te, n_te).argmax(1)
        metrics = trial_metrics(dims[test_idx], pred)
        log(f"        held-out bacc={metrics['balanced_accuracy']} "
            f"acc={metrics['accuracy']} 支持={metrics['support']}")
        fold_dir = Path(out_dir) / "folds"
        fold_dir.mkdir(parents=True, exist_ok=True)
        payload = build_payload(
            proto, outputs=outputs, channels=bank.channels,
            arch={"protocol": "IFNetRTG-L", "fs": float(bank.meta["fs"]),
                  "n_p": int(bank.meta["n_time"]), "n_c": int(bank.X.shape[1]),
                  "n_class": C.OUTPUT_COUNT, "compact_width": WIDTH,
                  "hidden": int(recipe.get("hidden", 8))},
            scaler={"mean": [0.0] * int(recipe.get("hidden", 8)),
                    "scale": [1.0] * int(recipe.get("hidden", 8)),
                    "fit_on": "fold (diagnostic only)"},
            studio={"kind": "source_fold", "held_out": held,
                    "created_at": datetime.now().isoformat(timespec="seconds")})
        fold_path = fold_dir / f"source_fold_{str(held).replace('/', '_')}.pt"
        save_model(payload, fold_path)
        folds.append({"held_out": held, "n_train_trials": int(len(train_idx)),
                      "n_inner_valid_trials": int(len(valid_idx)),
                      "n_test_trials": int(len(test_idx)), "metrics": metrics,
                      "train": {k: meta[k] for k in ("seconds", "pretrain", "head_init")},
                      "fold_model": str(fold_path.name)})

    evaluation = None
    if folds:
        baccs = [f["metrics"]["balanced_accuracy"] for f in folds
                 if f["metrics"]["balanced_accuracy"] is not None]
        note = ("outer held-out sessions are test-only; inner-valid is a trial-level "
                "split of the training sessions and is not a cross-session estimate")
        if eval_mode == "trial":
            note = ("single-session trial-level holdout: diagnostic only, not evidence of "
                    "cross-session generalisation")
        evaluation = {"mode": eval_mode, "folds": folds,
                      "mean_balanced_accuracy": float(np.mean(baccs)) if baccs else None,
                      "note": note}

    final_path = None
    if not final_fit:
        log("  [final] skipped by request (evaluation artifacts only)")
    else:
        train_idx, valid_idx = _stratified_trial_split(dims, 0.15, seed + 2)
        _ensure_coverage(dims, train_idx, "最终训练集")
        log(f"  [final] internally validated fit: {len(train_idx)} fitted / "
            f"{len(valid_idx)} inner-valid excluded (selected {len(dims)} trials)")
        proto, meta, _tr, _va = fit_one(train_idx, valid_idx)
        # scaler is fitted on the SOURCE training features, never on calibration data
        z = latent_features(proto, bank.X)
        scaler = {"mean": z.mean(0).tolist(),
                  "scale": np.maximum(z.std(0), 1e-6).tolist(),
                  "fit_on": "all selected source windows"}
        studio = {
            "format": C.SIDECAR_FORMAT, "kind": "source",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "studio_version": C.STUDIO_VERSION,
            "outputs": outputs,
            "model_labels": [int(o["label_id"]) for o in outputs],
            "profile": bank.meta["profile"],
            "channels": bank.channels,
            "recipe": dict(recipe),
            "data": {"sources": bank.meta.get("sources", []),
                     "classes": bank.summary()["classes"],
                     "rejected": bank.meta.get("rejected", {})},
            "evaluation": evaluation,
            "final_fit": {"n_selected_trials": int(len(dims)),
                          "n_fitted_trials": int(len(train_idx)),
                          "n_inner_valid_excluded": int(len(valid_idx)),
                          "n_windows": int(len(bank.y)),
                          "note": ("internally validated fit: the inner-valid trials are used "
                                   "for epoch selection and are excluded from every weight update; "
                                   "this is not a refit on all selected trials, and the metrics "
                                   "above belong to the held-out folds")},
            "fit": {k: meta[k] for k in ("seconds", "pretrain", "head_init")},
        }
        payload = build_payload(
            proto, outputs=outputs, channels=bank.channels,
            arch={"protocol": "IFNetRTG-L", "fs": float(bank.meta["fs"]),
                  "n_p": int(bank.meta["n_time"]), "n_c": int(bank.X.shape[1]),
                  "n_class": C.OUTPUT_COUNT, "compact_width": WIDTH,
                  "hidden": int(recipe.get("hidden", 8))},
            scaler=scaler, studio=studio)
        model_path = Path(out_dir) / "model.pt"
        final_path = str(save_model(payload, model_path))
        log(f"  [saved] {final_path}")

    report = {
        "kind": "source",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "profile": bank.meta["profile"],
        "outputs": outputs,
        "recipe": dict(recipe),
        "data": bank.summary(),
        "evaluation": evaluation,
        "model": final_path,
        "eval_mode": eval_mode,
        "final_fit": (None if not final_fit else {
            "n_selected_trials": int(len(dims)),
            "n_fitted_trials": int(len(train_idx)),
            "n_inner_valid_excluded": int(len(valid_idx)),
        }),
    }
    (Path(out_dir) / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
