"""IFNet训练、独立评价与有标签校准；模型输入约定随权重保存。"""
from __future__ import annotations

import copy
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import require_compatible, write_json
from .data import WindowDataset, build_dataset, require_disjoint, split_dataset
from .model import IFNetClassifier
from .preprocess import TrainStandardizer, filter_bank

MODEL_FORMAT = "handbci.ifnet_v2.v1"


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(y, predicted, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (np.asarray(y, int), np.asarray(predicted, int)), 1)
    total = matrix.sum()
    support = matrix.sum(axis=1)
    recall = np.divide(matrix.diagonal(), support, out=np.zeros(n_classes, float), where=support > 0)
    denom = matrix.sum(axis=0) + support
    f1 = np.divide(2 * matrix.diagonal(), denom, out=np.zeros(n_classes, float), where=denom > 0)
    return {"n_windows": int(total), "accuracy": float(np.trace(matrix) / total) if total else 0.0,
            "balanced_accuracy": float(recall[support > 0].mean()) if total else 0.0,
            "macro_f1": float(f1.mean()), "confusion_matrix": matrix.tolist(), "support": support.tolist()}


def transform_windows(payload: dict, windows: np.ndarray) -> np.ndarray:
    mean = payload["standardizer"]["mean"].cpu().numpy()
    std = payload["standardizer"]["std"].cpu().numpy()
    standardized = TrainStandardizer(mean, std).transform(windows)
    return filter_bank(standardized, payload["config"]["preprocess"]["output_rate"])


def predict_inputs(model: nn.Module, inputs: np.ndarray, device: str = "cpu", batch_size: int = 64):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            logits = model(torch.from_numpy(inputs[start:start + batch_size]).to(device))
            outputs.append(logits.softmax(dim=1).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty((0, model.classifier[0].out_features), np.float32)


def load_model(path: str | Path, device: str = "cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != MODEL_FORMAT:
        raise ValueError("仅支持整理版模型；不自动加载原工程的自定义pickle文件。")
    model = IFNetClassifier(**payload["architecture"])
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return payload, model


def save_model(path: str | Path, payload: dict, model: nn.Module):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"为保留模型来源，不覆盖已有模型：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = copy.deepcopy(payload)
    payload["state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def score_dataset(payload: dict, model: nn.Module, ds: WindowDataset, device="cpu"):
    require_compatible(payload["config"], ds.config)
    probabilities = predict_inputs(model, transform_windows(payload, ds.X), device)
    predicted = probabilities.argmax(axis=1)
    return metrics(ds.y, predicted, len(ds.config["classes"])), probabilities


def train(paths, output: str | Path, *, epochs=100, patience=20, batch_size=32,
          lr=1e-3, seed=42, split="session", device="cpu", progress=print):
    if min(epochs, patience, batch_size) <= 0 or lr <= 0:
        raise ValueError("训练轮数、早停耐心、batch及学习率必须为正。")
    seed_all(seed)
    ds = build_dataset(paths)
    train_ds, valid_ds, test_ds = split_dataset(ds, split, seed)
    cfg = ds.config
    arch = {"n_channels": ds.X.shape[1], "n_times": ds.X.shape[2],
            "n_classes": len(cfg["classes"]), "radix": 2,
            "out_planes": int(cfg.get("network", {}).get("out_planes", 64)),
            "patch_size": int(round(0.5 * cfg["preprocess"]["output_rate"]))}
    standardizer = TrainStandardizer.fit(train_ds.X)
    payload = {"format": MODEL_FORMAT, "config": cfg, "architecture": arch,
               "standardizer": {"mean": torch.from_numpy(standardizer.mean),
                                "std": torch.from_numpy(standardizer.std)},
               "training": {"seed": seed, "split": split,
                            "train_groups": sorted({r["group"] for r in train_ds.rows}),
                            "valid_groups": sorted({r["group"] for r in valid_ds.rows}),
                            "test_groups": sorted({r["group"] for r in test_ds.rows})}}
    x_train = torch.from_numpy(transform_windows(payload, train_ds.X))
    y_train = torch.from_numpy(train_ds.y)
    x_valid = transform_windows(payload, valid_ds.X)
    model = IFNetClassifier(**arch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                          batch_size=batch_size, shuffle=True,
                                          generator=torch.Generator().manual_seed(seed))
    counts = np.bincount(train_ds.y, minlength=len(cfg["classes"]))
    weights = torch.tensor(len(train_ds) / (len(counts) * counts), dtype=torch.float32, device=device)
    best_score, best_state, stale = -1.0, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb.to(device)), yb.to(device), weight=weights)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        valid_pred = predict_inputs(model, x_valid, device).argmax(axis=1)
        score = metrics(valid_ds.y, valid_pred, len(cfg["classes"]))["balanced_accuracy"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_balanced_accuracy": score})
        progress(f"epoch {epoch:3d}: train_loss={np.mean(losses):.4f}, valid_bacc={score:.3f}")
        if score > best_score:
            best_score, stale = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            payload["training"]["best_epoch"] = epoch
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise ValueError("至少训练1个epoch。")
    model.load_state_dict(best_state)
    payload["training"]["history"] = history
    # 只有在验证集选定权重后才评价测试集，测试集不参与标准化和早停。
    test_metrics, _ = score_dataset(payload, model, test_ds, device)
    report = {"format": MODEL_FORMAT, "data_source_types": sorted({r["source"] for r in ds.rows}),
              "synthetic_only": all(r["source"] in {"synthetic", "synthetic_tcp"} for r in ds.rows),
              "split": split, "n_train": len(train_ds), "n_valid": len(valid_ds), "n_test": len(test_ds),
              "best_epoch": payload["training"]["best_epoch"], "test": test_metrics,
              "history": history, "groups": {k: v for k, v in payload["training"].items() if k.endswith("_groups")}}
    save_model(output, payload, model)
    write_json(Path(output).with_suffix(".report.json"), report)
    return report


def tune_arrays(payload: dict, model: nn.Module, target: WindowDataset, *, mode="head",
                epochs=5, lr=1e-4, batch_size=32, device="cpu", replay: WindowDataset | None = None,
                seed=42):
    """头部微调冻结骨干及BN状态；全网络微调也默认冻结BN统计。"""
    require_compatible(payload["config"], target.config)
    require_learning_data(payload, target)
    if min(epochs, batch_size) <= 0 or lr <= 0:
        raise ValueError("校准轮数、batch及学习率必须为正。")
    seed_all(seed)
    target_x = transform_windows(payload, target.X)
    replay_x = None if replay is None else transform_windows(payload, replay.X)
    if replay is not None:
        require_compatible(target.config, replay.config)
        require_disjoint(target, replay)
        require_learning_data(payload, replay)
    if mode not in {"head", "all", "adabn"}:
        raise ValueError("校准模式为head/all/adabn。")
    for parameter in model.parameters():
        parameter.requires_grad_(mode == "all")
    if mode == "head":
        for parameter in model.classifier.parameters():
            parameter.requires_grad_(True)
    selected = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(selected, lr=lr) if selected else None
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(epochs):
        model.eval()
        if mode == "all":
            model.train()
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        elif mode == "head":
            model.classifier.train()
        else:
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.train()
        permutation = rng.permutation(len(target))
        for start in range(0, len(target), batch_size):
            rows = permutation[start:start + batch_size]
            xb, yb = target_x[rows], target.y[rows]
            if replay_x is not None:
                chosen = rng.integers(0, len(replay), size=len(rows))
                xb, yb = np.concatenate([xb, replay_x[chosen]]), np.concatenate([yb, replay.y[chosen]])
            x = torch.from_numpy(xb).to(device)
            if mode == "adabn":
                with torch.no_grad():
                    model(x)
            else:
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x), torch.from_numpy(yb).to(device))
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
    model.eval()
    return {"mode": mode, "epochs": epochs, "lr": lr, "bn_statistics": "update" if mode == "adabn" else "frozen",
            "n_target": len(target), "n_replay": 0 if replay is None else len(replay),
            "mean_loss": float(np.mean(losses)) if losses else None}


def require_learning_data(payload: dict, dataset: WindowDataset):
    heldout = set(payload.get("training", {}).get("valid_groups", [])) | set(payload.get("training", {}).get("test_groups", []))
    if heldout & {r["group"] for r in dataset.rows}:
        raise ValueError("不能把原模型的验证/测试试次用于梯度、BN更新或历史训练回放。")


def calibrate(model_path, paths, output, *, valid_paths=None, replay_paths=None, **kwargs):
    device = kwargs.get("device", "cpu")
    payload, model = load_model(model_path, device)
    target = build_dataset(paths, payload["config"])
    valid = build_dataset(valid_paths, payload["config"]) if valid_paths else None
    replay = build_dataset(replay_paths, payload["config"]) if replay_paths else None
    if valid is not None:
        require_disjoint(target, valid, *([replay] if replay is not None else []))
    # 原模型用于选权重的验证和测试试次，不允许再作为校准材料。
    heldout = set(payload.get("training", {}).get("valid_groups", [])) | set(payload.get("training", {}).get("test_groups", []))
    if heldout & {r["group"] for r in target.rows}:
        raise ValueError("校准材料包含原模型的验证/测试试次；请使用新会话或另行建立实验协议。")
    before = None if valid is None else score_dataset(payload, model, valid, device)[0]
    report = tune_arrays(payload, model, target, replay=replay, **kwargs)
    report.update({"parent": str(Path(model_path).resolve()), "target_groups": sorted({r["group"] for r in target.rows}),
                   "validation_before": before,
                   "validation_after": None if valid is None else score_dataset(payload, model, valid, device)[0]})
    payload["calibration"] = report
    save_model(output, payload, model)
    write_json(Path(output).with_suffix(".report.json"), report)
    return report
