"""按被试登记和选择个人模型，再调用现有实时解码入口。"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

from .config import load_config, require_compatible, validate_participant_id, write_json
from .training import load_model

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "configs/subject_models.json"
CATALOG_FORMAT = "handbci.subject_models.v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_registry(path=DEFAULT_REGISTRY):
    path = Path(path)
    if not path.is_file():
        return {"format": CATALOG_FORMAT, "models": []}
    catalog = json.loads(path.read_text(encoding="utf-8-sig"))
    if catalog.get("format") != CATALOG_FORMAT or not isinstance(catalog.get("models"), list):
        raise ValueError("被试模型列表格式无效。")
    keys = set()
    for entry in catalog["models"]:
        if not isinstance(entry, dict) or any(not isinstance(entry.get(key), str) or not entry[key] for key in ("participant_id", "model_id", "path", "sha256")):
            raise ValueError("模型列表中的被试、模型编号、路径和校验值须完整填写。")
        participant = validate_participant_id(entry["participant_id"])
        key = (participant, entry["model_id"])
        if key in keys:
            raise ValueError("被试模型列表包含重复编号，请检查配置。")
        if not entry.get("path") or not entry.get("sha256"):
            raise ValueError("模型登记缺少路径或校验值，请重新登记。")
        keys.add(key)
    return catalog


def checkpoint_participant(payload):
    participant = payload["config"].get("session_metadata", {}).get("participant_id")
    if not participant:
        raise ValueError("模型未保存被试编号，无法登记为个人模型；请使用带被试编号的采集记录训练。")
    return validate_participant_id(str(participant))


def resolve_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def register_model(participant, model_path, *, registry=DEFAULT_REGISTRY, name=None):
    participant = validate_participant_id(str(participant))
    model_path = Path(model_path).resolve()
    payload, _ = load_model(model_path, "cpu")
    owner = checkpoint_participant(payload)
    if owner != participant:
        raise ValueError(f"模型属于被试{owner}，不能登记为{participant}。")
    digest = file_sha256(model_path)
    try:
        stored_path = model_path.relative_to(ROOT).as_posix()
    except ValueError:
        stored_path = str(model_path)
    entry = {"participant_id": participant, "model_id": participant + "_" + digest[:12],
             "name": name or model_path.parent.name + " / " + model_path.name,
             "path": stored_path, "sha256": digest,
             "source": payload["config"]["device"]["source"],
             "n_channels": payload["architecture"]["n_channels"],
             "n_classes": payload["architecture"]["n_classes"],
             "registered_at_utc": datetime.now(timezone.utc).isoformat()}
    catalog = read_registry(registry)
    for current in catalog["models"]:
        if (current["participant_id"], current["model_id"]) == (participant, entry["model_id"]):
            if resolve_path(current["path"]) != model_path:
                raise ValueError("同一模型已登记在另一位置，请保留原路径或明确编辑模型列表。")
            return current
    catalog["models"].append(entry)
    path = Path(registry)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    write_json(temporary, catalog)
    temporary.replace(path)
    return entry


def prepare_subject_model(participant, model_id=None, *, registry=DEFAULT_REGISTRY, config_path=None):
    participant = validate_participant_id(str(participant))
    available = [e for e in read_registry(registry)["models"] if e["participant_id"] == participant]
    if not available:
        raise ValueError(f"没有登记被试{participant}的模型；不会自动使用其他被试的模型。")
    if model_id is None:
        if len(available) != 1:
            raise ValueError("该被试有多个模型，请明确选择模型版本。")
        entry = available[0]
    else:
        chosen = [e for e in available if e["model_id"] == model_id]
        if len(chosen) != 1:
            raise ValueError("所选模型不属于该被试，或模型编号不存在。")
        entry = chosen[0]
    path = resolve_path(entry["path"])
    if entry.get('runtime') == 'onnx':
        from .model_bundle import prepare_entry
        if config_path is not None:
            raise ValueError('ONNX模型包使用其manifest输入约定；修改后请重新打包导入。')
        return prepare_entry(entry, path.parent)
    if file_sha256(path) != entry["sha256"]:
        raise ValueError("登记后的模型文件发生变化；请核对文件并重新登记，当前不会连接数据源。")
    payload, _ = load_model(path, "cpu")
    if checkpoint_participant(payload) != participant:
        raise ValueError("模型内部被试编号与所选被试不一致。")
    config = load_config(config_path) if config_path else copy.deepcopy(payload["config"])
    require_compatible(payload["config"], config)
    trained_source, live_source = payload["config"]["device"]["source"], config["device"]["source"]
    trained_artificial = trained_source in {"synthetic", "synthetic_tcp"}
    live_artificial = live_source in {"synthetic", "synthetic_tcp"}
    if trained_artificial != live_artificial:
        raise ValueError("人工信号模型不能作为真实设备模型使用；请使用匹配的真实脑电训练模型。")
    configured_owner = config.get("session_metadata", {}).get("participant_id")
    if configured_owner is not None and str(configured_owner).strip() != participant:
        raise ValueError("连接配置的被试编号与所选被试不一致。")
    config["session_metadata"] = {**config.get("session_metadata", {}),
                                  "participant_id": participant, "online_model_id": entry["model_id"],
                                  "online_model_path": str(path), "online_model_sha256": entry["sha256"]}
    selection = {"participant_id": participant, "model_id": entry["model_id"], "name": entry["name"],
                 "model": str(path), "source": live_source,
                 "synthetic_model": trained_artificial,
                 "n_channels": payload["architecture"]["n_channels"],
                 "n_classes": payload["architecture"]["n_classes"],
                 "sample_rate": config["device"]["sample_rate"],
                 "host": config["device"].get("host"), "port": config["device"].get("port"),
                 "driver": "mock", "physical_action": False}
    return selection, config


def choose(items, title, *, key, input_fn=input, progress=print):
    if not items:
        raise ValueError("当前没有可选择的模型。先用 register-model 登记已有个人模型。")
    progress(title)
    for number, item in enumerate(items, 1):
        progress(f"  {number}. {item.get('name', item[key])}")
    if len(items) == 1:
        return items[0]
    while True:
        answer = input_fn("输入序号或编号：").strip()
        for item in items:
            if answer == item[key]:
                return item
        if answer.isdigit() and 1 <= int(answer) <= len(items):
            return items[int(answer) - 1]
        progress("选择无效，请重新输入。")


def run_subject_online(*, participant=None, model_id=None, registry=DEFAULT_REGISTRY,
                       config_path=None, output_root=ROOT / "runtime/live", trials=3,
                       device="cpu", check_only=False, start=False, input_fn=input, progress=print):
    if check_only and start:
        raise ValueError("配置检查与直接启动不能同时指定。")
    catalog = read_registry(registry)
    if participant is None:
        people = sorted({e["participant_id"] for e in catalog["models"]})
        chosen = choose([{"participant_id": p, "name": p} for p in people],
                        "选择被试：", key="participant_id", input_fn=input_fn, progress=progress)
        participant = chosen["participant_id"]
    participant = validate_participant_id(str(participant))
    options = [e for e in catalog["models"] if e["participant_id"] == participant]
    if model_id is None:
        chosen = choose(options, f"选择被试{participant}的模型：",
                        key="model_id", input_fn=input_fn, progress=progress)
        model_id = chosen["model_id"]
    selection, config = prepare_subject_model(participant, model_id, registry=registry, config_path=config_path)
    progress(f"\n被试：{participant}；模型：{selection['name']}")
    progress(f"输入：{selection['n_channels']}路，{selection['sample_rate']:g} Hz，"
             f"{selection['n_classes']}类；数据源：{selection['source']}。")
    progress("模型权重、标准化、通道及预处理配置随所选模型加载；当前机械手仅输出模拟指令。")
    if selection["synthetic_model"]:
        progress("此模型由人工信号训练，只用于软件测试，不能用于真实脑电控制。")
    if not check_only and not start:
        progress("\n1. 只检查模型与配置，结束（不连接数据源）\n2. 开始实时接收与解码（需要相应数据服务）")
        operation = input_fn("选择操作，直接回车为1：").strip() or "1"
        if operation not in {"1", "2"}:
            raise ValueError("操作须选择1或2。")
        check_only = operation == "1"
    if check_only:
        return {**selection, "check_only": True, "connected": False}
    progress(f"开始在线接收：{selection['host']}:{selection['port']}；按Ctrl+C停止。")
    from .runtime import run_live
    path = run_live(selection["model"], config, output_root, trials=trials, device=device, progress=progress)
    write_json(path / "model_selection.json", selection)
    return {**selection, "check_only": False, "output": str(path.resolve())}
