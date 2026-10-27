# -*- coding: utf-8 -*-
"""Export a studio model to the platform's ``handbci.onnx.v1`` contract.

Input  ``eeg``   [1, 2, 30, 500] float32  (4–16 / 16–40 Hz dual-band tensor)
Output ``scores``[1, 6]        float32  logits, columns in the model's declared order

The graph starts at the already-split tensor; the platform owns FIR,
decimation and the dual-band split.  The exported ZIP contains **exactly**
``manifest.json`` and ``model.onnx`` — the platform importer rejects any extra
entry — while the calibration-capable ``.pt`` and the Studio report stay
outside that ZIP.
"""
from __future__ import annotations

import copy
import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from . import contracts as C
from .training import SourceModel, _normalise_head

PLATFORM_CANDIDATES = (
    os.environ.get("HANDBCI_ROOT"),
    str(Path(__file__).resolve().parents[1]),
)


def platform_root() -> Path | None:
    for candidate in PLATFORM_CANDIDATES:
        if candidate and (Path(candidate) / "handbci/model_bundle.py").is_file():
            return Path(candidate)
    return None


def _effective_head(model: SourceModel) -> tuple[np.ndarray, np.ndarray]:
    head = _normalise_head(model.head)
    W1, b1, W2, b2 = head["W1"], head["b1"], head["W2"], head["b2"]
    if model.calibration is None:
        return W2 @ W1, W2 @ b1 + b2
    coef = np.asarray(model.calibration["coef"], np.float64)
    intercept = np.asarray(model.calibration["intercept"], np.float64)
    mean, scale = model.scaler["mean"], model.scaler["scale"]
    W = (coef / scale[None, :]) @ W1
    b = coef @ ((b1 - mean) / scale) + intercept
    return W, b


def band_stack(X: np.ndarray, fs: float = 250.0) -> np.ndarray:
    from .network import band_split_array
    return band_split_array(np.asarray(X, np.float32), fs,
                            [[4.0, 16.0], [16.0, 40.0]], mode="stack")


def _first_offset(profile: dict[str, Any]) -> float:
    offsets = [float(x) for x in (profile.get("offsets") or [])]
    return min(offsets) if offsets else 0.0


def build_config(model: SourceModel, profile: dict[str, Any]) -> dict[str, Any]:
    """A complete configuration accepted by ``handbci.config.validate_config``.

    ``stream_channels`` is the 32 EEG channels **plus** the trailing trigger;
    ``collection`` is required by the platform and is not optional metadata.
    """
    collection = {
        "loops": 20,
        "initial_rest_seconds": 4.0,
        "prepare_seconds": 2.4,
        "task_seconds": max(4.0, float(profile["window_seconds"]) + 2.0),
        "rest_seconds": 3.0,
        "block_rest_seconds": 6.0,
        "tail_seconds": 1.0,
    }
    return {
        "paradigm_id": "mi_studio_six_class",
        "device": {
            "source": "neuracle_acp",
            "unit": "uV",
            "sample_rate": int(profile["fs_in"]),
            "stream_channels": len(C.DEVICE_CHANNELS) + 1,
            "channels": list(C.DEVICE_CHANNELS),
            "reference": "A1/A2 冰点参考（记录时固定）",
            "host": None,
            "port": None,
        },
        "preprocess": {
            "highpass_hz": float(profile["highpass_hz"]),
            "lowpass_hz": float(profile["lowpass_hz"]),
            "numtaps": int(profile["numtaps"]),
            "trim_seconds": float(profile["trim_seconds"]),
            "input_rate": int(profile["fs_in"]),
            "output_rate": int(profile["fs_out"]),
        },
        "window": {
            "seconds": float(profile["window_seconds"]),
            "stride_seconds": float(profile.get("stride_seconds") or 0.5),
            # v2（prepare 锚点）的 offsets 由 stride 现算，部署侧要按 anchor 解释
            "task_offset_seconds": float(_first_offset(profile)),
            "anchor": str(profile.get("anchor") or "cue"),
        },
        "classes": [
            {"id": i, "name": str(o["name"]), "task_text": str(o["name"]),
             "action": "custom"}
            for i, o in enumerate(model.outputs)
        ],
        "collection": collection,
        "decision": {"confidence": 0.7, "margin": 0.15, "stable_seconds": 0.3,
                     "update_seconds": 0.1, "timeout_seconds": 10.0, "max_attempts": 2},
        "session_metadata": {
            "participant_id": "P01",
            "recording_context": {"condition": "studio_export",
                                  "note": "studio ONNX export"},
        },
    }


def build_manifest(model: SourceModel, profile: dict[str, Any],
                   name: str = "MI studio six-class model") -> dict[str, Any]:
    return {
        "format": "handbci.onnx.v1",
        "participant_id": "P01",
        "name": name[:100],
        "config": build_config(model, profile),
        "input": {"name": "eeg", "pipeline": "ifnet_filterbank",
                  "channels": list(model.channels)},
        "output": {"name": "scores", "kind": "logits"},
        "model_labels": list(model.model_labels),
        "outputs": model.outputs,
        "studio": {"profile": profile["id"],
                   "profile_fingerprint": C.profile_fingerprint(profile),
                   "note": ("Only platform_replay_v1 is window-compatible with the "
                            "platform replay pipeline; other profiles are portable graphs, "
                            "not a claim of replay parity.")},
    }


def _platform_check(folder: Path, manifest: dict[str, Any], participant: str) -> dict[str, Any]:
    root = platform_root()
    if root is None:
        return {"ok": False, "error": "platform checkout not found (set HANDBCI_ROOT)"}
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from handbci import model_bundle

    validated = model_bundle.validate_manifest(manifest)
    (folder / "manifest.json").write_text(
        json.dumps(validated, ensure_ascii=False, indent=2), encoding="utf-8")
    graph = (folder / "model.onnx").read_bytes()
    model_bundle.verify_graph(graph)
    decoder = model_bundle.ONNXDecoder(folder, validated)
    zap = io.BytesIO()
    with zipfile.ZipFile(zap, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(folder / "manifest.json", "manifest.json")
        zf.write(folder / "model.onnx", "model.onnx")
    with tempfile.TemporaryDirectory(prefix="mi_studio_import_") as tmp:
        entry = model_bundle.import_bundle(
            zap.getvalue(), participant, registry=Path(tmp) / "registry.json",
            import_root=Path(tmp) / "imported", name=manifest["name"])
        imported_root = Path(tmp) / "imported"
        candidates = [p for p in imported_root.glob("*") if p.is_dir()]
        imported_files = sorted(p.name for p in candidates[0].iterdir()) if candidates else []
    return {"ok": True, "model_id": entry["model_id"], "sha256": entry["sha256"],
            "n_channels": entry["n_channels"], "n_classes": entry["n_classes"],
            "imported_files": imported_files,
            "decoder_classes": len(decoder.manifest["config"]["classes"])}


def export_onnx(model_path: str | Path, out_dir: str | Path, *,
                sample: np.ndarray | None = None,
                name: str = "MI studio six-class model",
                require_platform: bool = True) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    model = SourceModel.load(model_path)
    profile, provenance = model.profile()
    if profile is None:
        raise C.ContractError(
            "模型没有可验证的预处理口径，拒绝导出可被误读的部署包；"
            "请先在配置里显式确认 profile")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    W, b = _effective_head(model)
    n_c = len(model.channels)
    n_p = int(model.arch["n_p"])

    class Export(torch.nn.Module):
        def __init__(self, backbone, W, b):
            super().__init__()
            self.backbone = backbone
            self.register_buffer("W", torch.from_numpy(W.astype(np.float32)))
            self.register_buffer("b", torch.from_numpy(b.astype(np.float32)))

        def forward(self, x):
            pooled = self.backbone.forward_features(x)
            return F.linear(pooled, self.W, self.b)

    backbone = copy.deepcopy(model.model).eval()
    net = Export(backbone, W, b).eval()
    onnx_path = out / "model.onnx"
    torch.onnx.export(net, torch.zeros(1, 2, n_c, n_p, dtype=torch.float32),
                      str(onnx_path), input_names=["eeg"], output_names=["scores"],
                      opset_version=17, dynamo=False)

    if sample is None:
        sample = np.zeros((2, n_c, n_p), np.float32)
    sample = np.ascontiguousarray(sample[: max(2, min(len(sample), 16))], np.float32)
    bands = np.ascontiguousarray(band_stack(sample, float(model.arch["fs"])))
    ref = np.asarray(model.predict_logits(sample), np.float64)
    with torch.no_grad():
        torch_out = net(torch.from_numpy(bands)).numpy()
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = np.concatenate([session.run(["scores"], {"eeg": bands[i:i + 1]})[0]
                               for i in range(len(bands))], 0)
    diff_t = float(np.abs(torch_out - ref).max())
    diff_o = float(np.abs(onnx_out - ref).max())
    agree = float((onnx_out.argmax(1) == ref.argmax(1)).mean())
    numerical_ok = bool(diff_t < 1e-4 and diff_o < 1e-4 and agree == 1.0)

    manifest = build_manifest(model, profile, name)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    platform = {"ok": False, "error": "not attempted"}
    if numerical_ok:
        try:
            platform = _platform_check(out, manifest, "P01")
        except Exception as exc:
            platform = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    zip_path = out / "deployment.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out / "manifest.json", "manifest.json")
        zf.write(onnx_path, "model.onnx")
    zip_entries = sorted(zipfile.ZipFile(zip_path).namelist())

    verify = {
        "n_samples": int(len(sample)),
        "torch_vs_numpy_max_abs_logit": diff_t,
        "onnx_vs_numpy_max_abs_logit": diff_o,
        "onnx_argmax_agreement": agree,
        "numerical_ok": numerical_ok,
        "platform": platform,
        "zip_entries": zip_entries,
        "zip_exact_two_files": zip_entries == ["manifest.json", "model.onnx"],
        "onnx_inputs": [{"name": i.name, "shape": i.shape} for i in session.get_inputs()],
        "onnx_outputs": [{"name": o.name, "shape": o.shape} for o in session.get_outputs()],
        "profile": profile["id"], "profile_source": provenance,
        "platform_compatible": profile["id"] == "platform_replay_v1",
    }
    verify["pass_all"] = bool(numerical_ok and verify["zip_exact_two_files"]
                              and platform.get("ok"))
    (out / "verify_onnx.json").write_text(json.dumps(verify, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    if require_platform and not verify["pass_all"]:
        raise C.ContractError(
            f"ONNX 部署校验失败：{platform.get('error') or 'numerical/zip check'}")
    return {"onnx": str(onnx_path), "manifest": str(out / "manifest.json"),
            "zip": str(zip_path), "verify": verify}
