# -*- coding: utf-8 -*-
"""Source calibration: a six-column linear read-out on frozen source features.

The output contract is inherited from the source model and asserted to be
byte-identical.  The latent scaler is the one fitted on the source training
windows; it is never refitted here.  Metrics come from a pre-declared
trial-level holdout that is disjoint from the read-out fit, while the saved
artifact is refit on every calibration trial afterwards (the report says so).
"""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import contracts as C
from .preprocess import WindowBank
from .training import (SourceModel, _aggregate_proba, _trial_table, _windows_for,
                       sha256_file, softmax, trial_metrics)


def _split_calibration(dims: np.ndarray, holdout_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    fit: list[int] = []
    hold: list[int] = []
    for c in range(C.OUTPUT_COUNT):
        pool = np.flatnonzero(dims == c)
        if len(pool) == 0:
            raise C.ContractError(f"校准数据缺少输出列 {c}")
        if len(pool) <= holdout_per_class:
            fit.extend(pool.tolist())
        else:
            chosen = rng.choice(pool, int(holdout_per_class), replace=False)
            hold.extend(chosen.tolist())
            fit.extend(sorted(set(pool.tolist()) - set(int(v) for v in chosen)))
    return np.asarray(sorted(fit), np.int64), np.asarray(sorted(hold), np.int64)


def _tensor_digest(model: SourceModel) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.model.state_dict().items()):
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def _scaler_digest(model: SourceModel) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(model.scaler["mean"], np.float64).tobytes())
    h.update(np.ascontiguousarray(model.scaler["scale"], np.float64).tobytes())
    return h.hexdigest()


def _trial_predictions(model: SourceModel, X: np.ndarray, trial_index: np.ndarray,
                       n_trials: int) -> np.ndarray:
    logits = np.asarray(model.predict_logits(X), np.float64)
    return _aggregate_proba(softmax(logits), trial_index, n_trials)


def calibrate_model(source_path: str | Path, bank: WindowBank, *,
                    C_value: float = 1.0, holdout_per_class: int = 1,
                    seed: int = 0, device: str = "cpu",
                    out_dir: str | Path = "runs/studio_calibration",
                    log: Callable[[str], None] = print) -> dict[str, Any]:
    source = SourceModel.load(source_path, device=device)
    before_tensors = _tensor_digest(source)
    before_scaler = _scaler_digest(source)
    outputs = copy.deepcopy(source.outputs)
    outputs = C.validate_outputs(outputs)
    if list(bank.channels) != list(source.channels):
        raise C.ContractError("源模型通道顺序与校准数据的 30 路输入不一致")
    if int(bank.meta.get("n_time", -1)) != int(source.arch["n_p"]):
        raise C.ContractError("源模型窗口长度与校准数据不一致")
    if int(bank.X.shape[1]) != int(source.arch["n_c"]):
        raise C.ContractError("源模型通道数与校准数据不一致")
    if bank.meta.get("outputs_fingerprint") != C.outputs_fingerprint(outputs):
        raise C.ContractError("校准数据的输出列顺序与源模型不一致")
    src_profile, _src = source.profile()
    bank_profile = bank.meta.get("profile")
    if src_profile is not None and isinstance(bank_profile, dict):
        if C.profile_fingerprint(src_profile) != C.profile_fingerprint(C.profile(str(bank_profile.get("id")))):
            raise C.ContractError("校准数据的预处理口径与源模型不一致")
    table = _trial_table(bank)
    dims = table["dim"]
    missing = [c for c in range(C.OUTPUT_COUNT) if not np.any(dims == c)]
    if missing:
        raise C.ContractError(f"校准数据缺少输出列 {missing}；六路校准要求六类都有标签")
    fit_idx, hold_idx = _split_calibration(dims, int(holdout_per_class), seed)
    log(f"  [split] fit {len(fit_idx)} 试次 / held-out {len(hold_idx)} 试次 "
        f"(每类留 {holdout_per_class})")
    X_fit, y_fit, tidx_fit, n_fit = _windows_for(bank, fit_idx)
    calibration = source.fit_calibration(X_fit, y_fit, C_value=C_value)
    calibrated = source.with_calibration(calibration)

    metrics: dict[str, Any]
    if len(hold_idx) and len(np.unique(dims[hold_idx])) == C.OUTPUT_COUNT:
        X_hold, y_hold, tidx_hold, n_hold = _windows_for(bank, hold_idx)
        before = _trial_predictions(source, X_hold, tidx_hold, n_hold)
        after = _trial_predictions(calibrated, X_hold, tidx_hold, n_hold)
        y_hold_trial = dims[hold_idx]
        metrics = {
            "kind": "trial_holdout",
            "holdout_per_class": int(holdout_per_class),
            "n_fit_trials": int(len(fit_idx)),
            "n_holdout_trials": int(len(hold_idx)),
            "before": trial_metrics(y_hold_trial, before.argmax(1)),
            "after": trial_metrics(y_hold_trial, after.argmax(1)),
            "note": ("before/after are measured on exactly the same untouched trials; "
                     "the saved head is refit on all calibration trials afterwards"),
        }
        log(f"  [holdout] before bacc={metrics['before']['balanced_accuracy']} "
            f"-> after bacc={metrics['after']['balanced_accuracy']}")
    else:
        X_all, y_all, tidx_all, n_all = _windows_for(bank, np.arange(len(dims)))
        before = _trial_predictions(source, X_all, tidx_all, n_all)
        after = _trial_predictions(calibrated, X_all, tidx_all, n_all)
        metrics = {
            "kind": "in_sample_diagnostic",
            "n_fit_trials": int(len(fit_idx)),
            "n_holdout_trials": int(len(hold_idx)),
            "before": trial_metrics(dims, before.argmax(1)),
            "after": trial_metrics(dims, after.argmax(1)),
            "note": ("每类试次不足以做留出（至少需要 2 个/类）；这里的数字是拟合集内诊断，"
                     "不能当作泛化性能"),
        }
        log("  [holdout] 每类试次不足，改用拟合集内诊断（不可当泛化性能）")

    # deployment refit on every calibration trial
    X_all, y_all, tidx_all, n_all = _windows_for(bank, np.arange(len(dims)))
    final_cal = source.fit_calibration(X_all, y_all, C_value=C_value)
    final_model = source.with_calibration(final_cal)

    source_sha = sha256_file(source_path) if Path(source_path).is_file() else None
    studio = {
        "format": C.SIDECAR_FORMAT,
        "kind": "calibration",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "studio_version": C.STUDIO_VERSION,
        "outputs": outputs,
        "model_labels": [int(o["label_id"]) for o in outputs],
        "output_fingerprint": C.outputs_fingerprint(outputs),
        "profile": bank.meta["profile"],
        "channels": bank.channels,
        "data": {"sources": bank.meta.get("sources", []),
                 "classes": bank.summary()["classes"],
                 "rejected": bank.meta.get("rejected", {})},
        "source_model": {"path": str(source_path), "sha256": source_sha},
        "calibration": {"C": float(C_value), "holdout_per_class": int(holdout_per_class),
                        "seed": int(seed), "n_trials": int(len(dims)),
                        "n_windows": int(len(bank.y))},
        "metrics": metrics,
        "final_fit": {"n_trials": int(len(dims)),
                      "note": "保存的是全部校准试次上重拟合的头；上面的指标来自拟合集与留出的对照"},
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "calibrated.pt"
    final_model.save(model_path, studio=studio)

    # ---- integrity checks (output contract + reload + six columns) ----
    reloaded = SourceModel.load(model_path, device=device)
    if C.outputs_fingerprint(reloaded.outputs) != C.outputs_fingerprint(outputs):
        raise AssertionError("校准后 6 维输出语义发生变化")
    if reloaded.model_labels != [int(o["label_id"]) for o in outputs]:
        raise AssertionError("校准后 model_labels 发生变化")
    probe = np.ascontiguousarray(bank.X[: min(6, len(bank.X))], np.float32)
    a = np.asarray(final_model.predict_logits(probe), np.float64)
    b = np.asarray(reloaded.predict_logits(probe), np.float64)
    if a.shape != (len(probe), C.OUTPUT_COUNT):
        raise AssertionError(f"校准模型输出列数不是 6：{a.shape}")
    if not np.allclose(a, b, atol=1e-5):
        raise AssertionError(f"保存-重载不一致：max|Δ|={np.abs(a - b).max()}")
    if _tensor_digest(reloaded) != before_tensors:
        raise AssertionError("校准改动了骨干/投影张量")
    if _scaler_digest(reloaded) != before_scaler:
        raise AssertionError("校准改动了源模型拟合的 latent scaler")
    if _tensor_digest(final_model) != before_tensors:
        raise AssertionError("校准头拟合改动了骨干/投影张量")
    backbone_sha = sha256_file(model_path)

    report = {
        "kind": "calibration",
        "created_at": studio["created_at"],
        "source_model": str(source_path),
        "source_sha256": source_sha,
        "model": str(model_path),
        "model_sha256": backbone_sha,
        "outputs": outputs,
        "output_fingerprint": studio["output_fingerprint"],
        "profile": bank.meta["profile"],
        "data": bank.summary(),
        "metrics": metrics,
        "reload_check": {"max_abs_logit_diff": float(np.abs(a - b).max()),
                         "columns": int(a.shape[1])},
        "integrity": {"tensor_digest": before_tensors,
                      "scaler_digest": before_scaler,
                      "backbone_and_scaler_unchanged": True},
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    log(f"  [saved] {model_path}（输出契约未变，重载检查通过）")
    return report
