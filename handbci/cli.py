"""所有操作的统一命令入口；不依赖原工程目录或图形界面。"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]


def positive_int(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("须为正整数。")
    return n


def parser():
    p = argparse.ArgumentParser(description="脑控机械手基础工程：采集、训练、校准、解码与模拟输出")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="检查依赖，读取配置不会连接设备")
    ui = sub.add_parser("collect-ui", help="打开完整采集界面：准备、想象、休息及信号保存")
    ui.add_argument("--output", type=Path, default=ROOT / "runtime/records")
    ui.add_argument("--port", type=int, default=0, help="本机界面端口，0表示自动选择")
    ui.add_argument("--no-browser", action="store_true", help="只启动服务，手动打开打印的页面地址")
    online_ui = sub.add_parser("online-ui", help="打开网页版在线控制：选择被试、模型与实时解码")
    online_ui.add_argument("--output", type=Path, default=ROOT / "runtime/live")
    online_ui.add_argument("--registry", type=Path, default=ROOT / "configs/subject_models.json")
    online_ui.add_argument("--port", type=int, default=0)
    online_ui.add_argument("--no-browser", action="store_true")
    demo = sub.add_parser("demo", help="生成模拟记录并跑完整训练/校准/回放")
    demo.add_argument("--epochs", type=positive_int, default=15)
    demo.add_argument("--output", type=Path, default=ROOT / "runtime/demo")
    tcp_demo = sub.add_parser("collect-tcp-demo", help="启动本地TCP人工源，经真实端口采集并核对标签")
    tcp_demo.add_argument("--config", type=Path, default=ROOT / "configs/simulated_tcp_6class.json")
    tcp_demo.add_argument("--output", type=Path, default=ROOT / "runtime/tcp_collection")
    tcp_demo.add_argument("--port", type=int, help="覆盖本地端口，默认18712；0自动选择空闲端口")
    tcp_demo.add_argument("--loops", type=positive_int, help="覆盖采集轮数；默认每类采集1次")
    tcp_server = sub.add_parser("simulate-device", help="单独启动本地TCP人工背景信号服务")
    tcp_server.add_argument("--config", type=Path, default=ROOT / "configs/simulated_tcp_6class.json")
    tcp_server.add_argument("--port", type=int, help="覆盖本地端口，默认18712；0自动选择空闲端口")
    collect = sub.add_parser("collect", help="按提示采集一份带标签记录")
    collect.add_argument("--config", type=Path, required=True)
    collect.add_argument("--output", type=Path, default=ROOT / "runtime/records")
    collect.add_argument("--trigger-port", help="如COM3；指定后要求在EEG中收到硬件trigger")
    inspect = sub.add_parser("inspect", help="检查整理版记录及各类完整试次数量")
    inspect.add_argument("--session", type=Path, required=True)
    train = sub.add_parser("train", help="按会话或完整试次划分训练/验证/测试")
    train.add_argument("--sessions", nargs="+", required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--split", choices=["session", "trial"], default="session")
    train.add_argument("--epochs", type=positive_int, default=100)
    train.add_argument("--patience", type=positive_int, default=20)
    train.add_argument("--batch-size", type=positive_int, default=32)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=42)
    evaluate = sub.add_parser("evaluate", help="固定模型评价指定记录")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--sessions", nargs="+", required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    calibrate = sub.add_parser("calibrate", help="用新有标签数据校准，另存候选模型")
    calibrate.add_argument("--model", type=Path, required=True)
    calibrate.add_argument("--sessions", nargs="+", required=True)
    calibrate.add_argument("--valid-sessions", nargs="+")
    calibrate.add_argument("--replay-sessions", nargs="+")
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--mode", choices=["head", "all", "adabn"], default="head")
    calibrate.add_argument("--epochs", type=positive_int, default=5)
    calibrate.add_argument("--batch-size", type=positive_int, default=32)
    calibrate.add_argument("--lr", type=float, default=1e-4)
    replay = sub.add_parser("replay", help="按数据到达顺序回放并记录模拟动作")
    adaptive = sub.add_parser("adapt-replay", help="仅用已结束轮次做有标签校准回放")
    for item in (replay, adaptive):
        item.add_argument("--model", type=Path, required=True)
        item.add_argument("--session", type=Path, required=True)
        item.add_argument("--output", type=Path, default=ROOT / "runtime/replays")
    adaptive.add_argument("--replay-sessions", nargs="+")
    adaptive.add_argument("--adapt-epochs", type=positive_int, default=1)
    online = sub.add_parser("online", help="实时读设备并解码；机械手输出固定为模拟驱动")
    online.add_argument("--model", type=Path, required=True)
    online.add_argument("--config", type=Path, required=True)
    online.add_argument("--trials", type=positive_int, default=3)
    online.add_argument("--output", type=Path, default=ROOT / "runtime/live")
    register = sub.add_parser("register-model", help="把已有个人模型登记到相应被试名下")
    register.add_argument("--participant", required=True)
    register.add_argument("--model", type=Path, required=True)
    register.add_argument("--name", help="模型版本的显示名称")
    register.add_argument("--registry", type=Path, default=ROOT / "configs/subject_models.json")
    subject_online = sub.add_parser("subject-online", help="选择被试与模型，再检查配置或启动实时解码")
    subject_online.add_argument("--participant", help="不填则交互选择已登记的被试")
    subject_online.add_argument("--model-id", help="不填则交互选择该被试的模型版本")
    subject_online.add_argument("--registry", type=Path, default=ROOT / "configs/subject_models.json")
    subject_online.add_argument("--config", type=Path, help="可选的兼容连接配置；默认使用模型保存的配置")
    subject_online.add_argument("--trials", type=positive_int, default=3)
    subject_online.add_argument("--output", type=Path, default=ROOT / "runtime/live")
    subject_mode = subject_online.add_mutually_exclusive_group()
    subject_mode.add_argument("--check-only", action="store_true", help="只检查模型，不连接设备")
    subject_mode.add_argument("--start", action="store_true", help="选定模型后直接开始实时接收")
    for item in (train, evaluate, calibrate, replay, adaptive, online, subject_online):
        item.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            available = {n: importlib.util.find_spec(n) is not None for n in ("numpy", "scipy", "h5py", "torch", "serial")}
            print(json.dumps({"python": sys.executable, "dependencies": available,
                              "serial_note": "serial仅硬件TriggerBox需要；无该依赖仍可模拟、训练及Neuracle采集。"}, ensure_ascii=False, indent=2))
            return 0 if all(available[n] for n in ("numpy", "scipy", "h5py", "torch")) else 1
        from .config import load_config, write_json
        if args.command == "online-ui":
            from .online_web import run_online_ui
            result = run_online_ui(args.output, registry=args.registry, port=args.port,
                                   open_browser=not args.no_browser)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "collect-ui":
            from .collection_ui import run_collection_ui
            result = run_collection_ui(args.output, port=args.port, open_browser=not args.no_browser)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command in {"collect-tcp-demo", "simulate-device"}:
            from .tcp_simulator import run_tcp_collection_demo, serve_standalone
            config = load_config(args.config)
            progress = lambda message: print(message, flush=True)
            if args.command == "collect-tcp-demo":
                if args.loops is not None:
                    config["collection"]["loops"] = args.loops
                result = run_tcp_collection_demo(config, args.output, port=args.port, progress=progress)
            else:
                result = serve_standalone(config, port=args.port, progress=progress)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "collect":
            from .acquisition import collect
            print(collect(load_config(args.config), args.output, args.trigger_port))
            return 0
        if args.command == "inspect":
            from .data import load_session, trials_from_events
            r = load_session(args.session)
            trials = trials_from_events(r["events"], len(r["config"]["classes"]))
            print(json.dumps({"source": r["metadata"]["source"], "duration_seconds": len(r["indices"]) / r["config"]["device"]["sample_rate"],
                              "n_trials": len(trials), "counts": {c["name"]: sum(t["label"] == c["id"] for t in trials) for c in r["config"]["classes"]},
                              "marking_types": sorted({e.get("marking", "unknown") for e in r["events"]})}, ensure_ascii=False, indent=2))
            return 0
        import torch
        torch.set_num_threads(2)
        device = getattr(args, "device", "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("指定了CUDA，但当前环境不可用；请改为--device cpu。")
        if args.command == "register-model":
            from .subject_online import register_model
            result = register_model(args.participant, args.model, registry=args.registry, name=args.name)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "subject-online":
            from .subject_online import run_subject_online
            result = run_subject_online(participant=args.participant, model_id=args.model_id,
                                        registry=args.registry, config_path=args.config, trials=args.trials,
                                        output_root=args.output, device=device,
                                        check_only=args.check_only, start=args.start)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        from . import training
        if args.command == "train":
            result = training.train(args.sessions, args.output, epochs=args.epochs, patience=args.patience,
                                    batch_size=args.batch_size, lr=args.lr, seed=args.seed, split=args.split, device=device)
        elif args.command == "evaluate":
            from .data import build_dataset
            payload, model = training.load_model(args.model, device)
            ds = build_dataset(args.sessions, payload["config"])
            result, probabilities = training.score_dataset(payload, model, ds, device)
            result["rows"] = [{**row, "true_class": int(ds.y[i]), "probabilities": probabilities[i].tolist()} for i, row in enumerate(ds.rows)]
            write_json(args.output, result)
        elif args.command == "calibrate":
            result = training.calibrate(args.model, args.sessions, args.output, valid_paths=args.valid_sessions,
                                        replay_paths=args.replay_sessions, mode=args.mode, epochs=args.epochs,
                                        lr=args.lr, batch_size=args.batch_size, device=device)
        elif args.command in {"replay", "adapt-replay"}:
            from .runtime import run_replay
            path, result = run_replay(args.model, args.session, args.output, device=device,
                                      adapt=args.command == "adapt-replay", adapt_epochs=getattr(args, "adapt_epochs", 1),
                                      replay_paths=getattr(args, "replay_sessions", None))
            result["output"] = str(path)
        elif args.command == "online":
            from .runtime import run_live
            path = run_live(args.model, load_config(args.config), args.output, trials=args.trials, device=device)
            result = {"output": str(path), "driver": "mock"}
        else:
            from .demo import run_demo
            result = run_demo(args.output, args.epochs)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except EOFError:
        print("未完成选择，未启动在线接收。", file=sys.stderr)
        return 130
    except KeyboardInterrupt:
        print("已中止；已采集的数据保留为未完成会话。", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"操作失败：{exc}", file=sys.stderr)
        return 1
