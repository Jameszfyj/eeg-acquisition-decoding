# -*- coding: utf-8 -*-
"""Command-line entry point for the studio (the web server is only a client).

    python -m mi_studio scan  <session-or-parent> [--json]
    python -m mi_studio run   --config job.json
    python -m mi_studio train --config job.json
    python -m mi_studio calibrate --config job.json
    python -m mi_studio inspect --model model.pt
    python -m mi_studio export-onnx --model model.pt --out out/ [--sample-npz bank.npz]
    python -m mi_studio serve --host 127.0.0.1 --port 57500
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from . import contracts as C
from .adapters import scan
from .preprocess import build_bank

ROOT = Path(os.environ.get('MI_STUDIO_ROOT',str(Path(__file__).resolve().parent.parent)))


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load_config(path: str | None) -> dict[str, Any]:
    if not path or path == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise C.ContractError("配置必须是 JSON 对象")
    return data


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def _job_dir(config: dict[str, Any], kind: str) -> Path:
    if config.get("out_dir"):
        out = _resolve(config["out_dir"])
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = ROOT / "runs" / "studio" / f"{stamp}_{kind}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_status(out: Path, **fields: Any) -> None:
    data = {"updated_at": _now()}
    path = out / "status.json"
    if path.is_file():
        try:
            data.update(json.loads(path.read_text(encoding="utf-8")))
        except ValueError:
            pass
    data.update(fields)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _prepare_sources(config: dict[str, Any], outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise C.ContractError("配置里没有 sources")
    prepared = []
    seen_paths = set()
    for item in sources:
        if not isinstance(item, dict) or not item.get("path"):
            raise C.ContractError(f"数据源缺少 path：{item!r}")
        path = str(_resolve(item["path"]))
        if path in seen_paths:
            raise C.ContractError(f"同一路径重复加载：{path}")
        seen_paths.add(path)
        mapping = item.get("mapping")
        if not isinstance(mapping, dict) or not mapping.get("rows"):
            raise C.ContractError(f"{path}: 缺少显式 mapping.rows")
        prepared.append({"path": path, "mapping": mapping, "_outputs": outputs})
    return prepared


def _print_report(report: dict[str, Any]) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_scan(args) -> int:
    rows = scan(args.path)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        for row in rows:
            if row.get("kind") == "invalid":
                print(f"[invalid] {row['path']}: {row.get('error')}")
                continue
            labels = ", ".join(
                f"{item['id']}:{item.get('hint') or '?'}×{item['count']}"
                for item in row["labels"])
            print(f"[{row['kind_label']}] {row['session_id']}  {row['n_trials']} 试次  "
                  f"{row['duration_s']:.1f}s  {row['date']}  {labels}")
    return 0


def _load_bank(path: str | Path):
    from .preprocess import WindowBank
    return WindowBank.open(_resolve(path))


def _run_bank(config: dict[str, Any], out: Path) -> dict[str, Any]:
    """只构建窗口库并落盘（低内存：X 走 memmap）。"""
    from .preprocess import WindowBank

    outputs = C.validate_outputs(config.get("outputs") or C.default_outputs())
    profile = str(config.get("profile") or C.DEFAULT_PROFILE)
    target = _resolve(config.get("bank_out") or (out / "bank"))
    _write_status(out, state="running", phase="build_bank", started_at=_now(), config=config)
    print(f"[studio] bank profile={profile} -> {target}")
    bank = build_bank(_prepare_sources(config, outputs), profile, log=print, out_dir=target)
    summary = bank.summary()
    report = {"kind": "bank", "profile": bank.meta["profile"], "bank": str(target),
              "n_windows": bank.n_windows, "n_trials": bank.n_trials,
              "data": summary, "anchor_kinds": bank.meta.get("anchor_kinds"),
              "rejected": bank.meta.get("rejected")}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    _write_status(out, state="done", phase="done", finished_at=_now(), result=report)
    print(f"[studio] bank: {bank.n_windows} 窗口 / {bank.n_trials} 试次 -> {target}")
    return report


def _run_train(config: dict[str, Any], out: Path, bank_path: str | Path | None = None) -> dict[str, Any]:
    from . import training
    from .onnx_export import export_onnx

    outputs = C.validate_outputs(config.get("outputs") or C.default_outputs())
    profile = str(config.get("profile") or C.DEFAULT_PROFILE)
    _write_status(out, state="running", phase="build_bank", started_at=_now(),
                  config=config)
    if bank_path:
        bank = _load_bank(bank_path)
        profile = str(bank.meta["profile"]["id"])
        if C.outputs_fingerprint(bank.meta["outputs"]) != C.outputs_fingerprint(outputs):
            raise C.ContractError("窗口库的 6 维输出列顺序与本次任务的 declarations 不一致")
        requested = config.get("profile")
        if requested and C.profile_fingerprint(C.profile(str(requested))) != C.profile_fingerprint(bank.meta["profile"]):
            raise C.ContractError(f"窗口库口径 {profile} 与任务要求的 {requested} 不一致")
        print(f"[studio] profile={profile} bank={bank_path} "
              f"windows={bank.n_windows} trials={bank.n_trials} "
              f"outputs={[o['name'] for o in outputs]}")
    else:
        print(f"[studio] profile={profile} sources={len(config.get('sources') or [])} "
              f"outputs={[o['name'] for o in outputs]}")
        bank = build_bank(_prepare_sources(config, outputs), profile, log=print)
    if not bank.n_windows:
        raise C.ContractError("没有窗口")
    print(f"[studio] bank: {bank.n_windows} 窗口 / {bank.n_trials} 试次")
    _write_status(out, phase="train", bank=bank.summary())
    train_cfg = dict(config.get("train") or {})
    recipe = {
        "pre_epochs": int(train_cfg.get("pre_epochs", 200)),
        "head_epochs": int(train_cfg.get("head_epochs", 30)),
        "ft_epochs": int(train_cfg.get("ft_epochs", 200)),
        "patience": int(train_cfg.get("patience", 40)),
        "batch": int(train_cfg.get("batch", 128)),
        "hidden": int(train_cfg.get("hidden", 8)),
    }
    eval_mode = str(train_cfg.get("eval", "holdout"))
    holdout = train_cfg.get("holdout") or []
    device = str(train_cfg.get("device", "auto"))
    if device == "auto":
        device = training.device_auto()
    report = training.train_source(
        bank, outputs=outputs, recipe=recipe, eval_mode=eval_mode,
        holdout=holdout, seed=int(train_cfg.get("seed", 0)), device=device,
        final_fit=bool(train_cfg.get("final_fit", True)), out_dir=out, log=print)
    if config.get("export_onnx") and report.get("model"):
        _write_status(out, phase="export", message="导出 ONNX")
        sample = bank.X[: min(12, len(bank.X))]
        onnx_info = export_onnx(report["model"], out / "onnx", sample=sample,
                                name=str(config.get("name") or "MI studio six-class model"))
        report["onnx"] = onnx_info
        print(f"[studio] ONNX: {onnx_info['onnx']} pass={onnx_info['verify']['pass_all']}")
    (out / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    _write_status(out, state="done", phase="done", finished_at=_now(),
                  result=report, model=report.get("model"))
    _print_report(report)
    return report


def _run_calibrate(config: dict[str, Any], out: Path) -> dict[str, Any]:
    from .calibrate import calibrate_model
    from .onnx_export import export_onnx
    from .training import SourceModel

    source_path = config.get("source_model")
    if not source_path:
        raise C.ContractError("校准配置缺少 source_model")
    source_path = _resolve(source_path)
    source = SourceModel.load(source_path)
    outputs = C.validate_outputs(config.get("outputs") or source.outputs)
    if C.outputs_fingerprint(outputs) != C.outputs_fingerprint(source.outputs):
        raise C.ContractError(
            "校准配置的 6 维输出语义与源模型不一致；校准必须原样继承源模型 metadata")
    resolved, provenance = source.profile()
    requested = config.get("profile")
    if resolved is None:
        if not bool(config.get("legacy_profile_confirmed")):
            raise C.ContractError(
                "源模型没有可验证的预处理口径（studio/preprocess_profile/spec.json 均缺失）。"
                "请显式选择 profile 并设置 legacy_profile_confirmed=true，不要静默默认。")
        if not requested:
            raise C.ContractError("legacy 模型必须显式给出 profile")
        profile = C.profile(str(requested))
        print(f"[studio] legacy 模型口径由操作者显式确认：{profile['id']}")
    else:
        profile = resolved
        if requested and C.profile_fingerprint(C.profile(str(requested))) != C.profile_fingerprint(profile):
            raise C.ContractError(
                f"校准配置口径 {requested} 与源模型口径 {profile['id']}（来自 {provenance}）不一致")
    _write_status(out, state="running", phase="build_bank", started_at=_now(),
                  config=config, outputs=outputs, profile=profile["id"])
    print(f"[studio] calibrate source={source_path.name} profile={profile} "
          f"outputs={[o['name'] for o in outputs]}")
    bank = build_bank(_prepare_sources(config, outputs), profile["id"], log=print)
    if list(bank.channels) != list(source.channels):
        raise C.ContractError("源模型通道顺序与 studio 的 30 路输入不一致")
    if int(bank.meta["n_time"]) != int(source.arch["n_p"]) or int(bank.X.shape[1]) != int(source.arch["n_c"]):
        raise C.ContractError("源模型的窗口/通道维度与校准数据不一致")
    if C.profile_fingerprint(bank.meta["profile"]) != C.profile_fingerprint(profile):
        raise C.ContractError("校准数据的预处理口径与源模型不一致")
    if bank.meta.get("outputs_fingerprint") != C.outputs_fingerprint(outputs):
        raise C.ContractError("校准数据的 6 维输出列顺序与源模型不一致")
    cal_cfg = dict(config.get("calibration") or {})
    _write_status(out, phase="calibrate", bank=bank.summary())
    report = calibrate_model(
        source_path, bank, C_value=float(cal_cfg.get("C", 1.0)),
        holdout_per_class=int(cal_cfg.get("holdout_per_class", 1)),
        seed=int(cal_cfg.get("seed", 0)),
        device=str(cal_cfg.get("device", "auto")) if cal_cfg.get("device", "auto") != "auto"
        else _auto_device(),
        out_dir=out, log=print)
    if config.get("export_onnx") and report.get("model"):
        _write_status(out, phase="export", message="导出 ONNX")
        sample = bank.X[: min(12, len(bank.X))]
        onnx_info = export_onnx(report["model"], out / "onnx", sample=sample,
                                name=str(config.get("name") or "MI studio calibrated model"))
        report["onnx"] = onnx_info
        print(f"[studio] ONNX: {onnx_info['onnx']} pass={onnx_info['verify']['pass_all']}")
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    (out / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    _write_status(out, state="done", phase="done", finished_at=_now(),
                  result=report, model=report.get("model"))
    _print_report(report)
    return report


def _auto_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def cmd_run(args) -> int:
    config = _load_config(args.config)
    kind = str(config.get("kind") or "train")
    if kind not in ("train", "source", "calibrate", "calibration", "bank"):
        raise C.ContractError(f"未知任务类型 {kind}")
    kind = {"source": "train", "calibration": "calibrate"}.get(kind, kind)
    out = _job_dir(config, kind)
    (out / "request.json").write_text(json.dumps(config, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    try:
        if kind == "bank":
            _run_bank(config, out)
        elif kind == "train":
            _run_train(config, out, config.get("bank"))
        else:
            _run_calibrate(config, out)
        return 0
    except Exception as exc:
        traceback.print_exc()
        _write_status(out, state="error", phase="error", finished_at=_now(),
                      error=f"{type(exc).__name__}: {exc}")
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return 2


def cmd_train(args) -> int:
    config = _load_config(args.config)
    bank = args.bank or config.get("bank")
    out = _job_dir(config, "train")
    (out / "request.json").write_text(json.dumps(config, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    try:
        _run_train(config, out, bank)
        return 0
    except Exception as exc:
        traceback.print_exc()
        _write_status(out, state="error", phase="error", finished_at=_now(),
                      error=f"{type(exc).__name__}: {exc}")
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return 2


def cmd_inspect(args) -> int:
    from .training import SourceModel
    model = SourceModel.load(_resolve(args.model))
    resolved, provenance = model.profile()
    print(json.dumps({
        "model": str(_resolve(args.model)),
        "profile": {"resolved": (resolved or {}).get("id"), "source": provenance},
        "outputs": model.outputs,
        "model_labels": model.model_labels,
        "channels": model.channels,
        "arch": model.arch,
        "calibrated": model.calibration is not None,
        "studio": model.payload.get("studio"),
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_export_onnx(args) -> int:
    from .onnx_export import export_onnx
    sample = None
    if args.sample_npz:
        import numpy as np
        data = np.load(args.sample_npz, allow_pickle=False)
        sample = np.asarray(data["X"], np.float32)
    info = export_onnx(_resolve(args.model), _resolve(args.out), sample=sample,
                       name=args.name)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0 if info["verify"]["pass_all"] else 1


def cmd_serve(args) -> int:
    from .web.server import serve
    return serve(host=args.host, port=args.port, root=_resolve(args.root) if args.root else ROOT,
                 token=args.token, open_browser=args.open_browser)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("mi_studio")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan", help="识别一个会话目录（或一个包含会话的父目录）")
    p.add_argument("path")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_scan)

    for name in ("run", "calibrate", "bank"):
        p = sub.add_parser(name, help="执行一个 JSON 任务配置")
        p.add_argument("--config", required=True, help="JSON 文件，或 - 表示标准输入")
        p.set_defaults(func=cmd_run)

    p = sub.add_parser("train", help="执行训练配置；--bank 复用已落盘的窗口库")
    p.add_argument("--config", required=True, help="JSON 文件，或 - 表示标准输入")
    p.add_argument("--bank", default="", help="窗口库目录（或 .npz）；给了就不重建")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("inspect", help="查看模型输出契约")
    p.add_argument("--model", required=True)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("export-onnx", help="导出 handbci.onnx.v1 图")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sample-npz")
    p.add_argument("--name", default="MI studio six-class model")
    p.set_defaults(func=cmd_export_onnx)

    p = sub.add_parser("serve", help="启动可视化网站（CLI 客户端）")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=57500)
    p.add_argument("--root", default="")
    p.add_argument("--token", default=os.environ.get("MI_STUDIO_TOKEN", ""))
    p.add_argument("--open-browser", action="store_true")
    p.set_defaults(func=cmd_serve)
    return ap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except C.ContractError as exc:
        print(f"[contract] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
