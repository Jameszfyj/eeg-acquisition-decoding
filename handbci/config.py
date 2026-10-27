"""唯一配置入口；模型文件会保存其训练时的输入约定。"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

HAND_ACTIONS = ("thumb", "index", "middle", "ring", "little", "pinch")
ACTION_NAMES = dict(zip(HAND_ACTIONS, ("拇指", "食指", "中指", "无名指", "小指", "拇食指捏合")))


def validate_participant_id(value: str) -> str:
    """保留输入编号作为目录后缀；提前拒绝Windows不允许的文件名字符。"""
    participant = value.strip()
    if not participant or len(participant) > 64:
        raise ValueError("被试编号须为1至64个字符。")
    if any(ord(c) < 32 or c in '<>:"/\\|?*' for c in participant) or participant.endswith("."):
        raise ValueError("被试编号不能包含斜杠、反斜杠、冒号等文件名禁用字符，且不能以句点结尾。")
    return participant


def validate_config(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    device, prep, window = cfg["device"], cfg["preprocess"], cfg["window"]
    channels = device["channels"]
    if not channels or len(channels) != len(set(channels)):
        raise ValueError("设备通道名称必须非空且不能重复。")
    if not 1 <= len(channels) <= 64:
        raise ValueError("本工程配置须包含1至64个脑电通道。")
    if int(device["stream_channels"]) <= len(channels):
        raise ValueError("流总通道必须另外包含末尾的trigger通道。")
    fs, out_fs = float(device["sample_rate"]), float(prep["output_rate"])
    factor = fs / out_fs if out_fs > 0 else 0
    if fs <= 0 or factor < 1 or abs(factor - round(factor)) > 1e-9:
        raise ValueError("目前只支持整数倍降采样，如1000 Hz到250 Hz。")
    if not 0 < prep["highpass_hz"] < prep["lowpass_hz"] < out_fs / 2:
        raise ValueError("滤波频率必须满足 0 < 高通 < 低通 < 输出采样率/2。")
    if int(prep["numtaps"]) < 3 or int(prep["numtaps"]) % 2 != 1:
        raise ValueError("FIR系数数量必须为不小于3的奇数。")
    if float(prep["trim_seconds"]) < 0:
        raise ValueError("开头剔除时长不能为负。")
    n = round(float(window["seconds"]) * out_fs)
    patch = round(0.5 * out_fs)
    if n < max(64, patch) or n % patch:
        raise ValueError("IFNet窗口须足够长，且为0.5秒的整数倍。")
    if float(window["stride_seconds"]) <= 0 or float(window["task_offset_seconds"]) < 0:
        raise ValueError("切窗步长必须为正，任务起点偏移不能为负。")
    classes = cfg["classes"]
    if len(classes) < 2 or [c["id"] for c in classes] != list(range(len(classes))):
        raise ValueError("类别id须从0连续编号，至少两类。")
    if len({c["name"] for c in classes}) != len(classes):
        raise ValueError("类别名称不能重复。")
    if any(c["action"] not in HAND_ACTIONS + ("left_hand", "right_hand", "custom") for c in classes):
        raise ValueError("动作须使用已知手指、left_hand/right_hand或custom标识。")
    if not device.get("unit"):
        raise ValueError("必须记录原始信号单位；未核实时可填device_native并保留同一约定。")
    if device["source"] not in {"synthetic", "synthetic_tcp", "neuracle", "neuracle_acp", "device_adapter"}:
        raise ValueError("source须使用synthetic、synthetic_tcp、neuracle、neuracle_acp或device_adapter。")
    if device["source"] == "synthetic_tcp":
        if device.get("host") != "127.0.0.1":
            raise ValueError("TCP模拟源仅支持本机127.0.0.1。")
        if device["stream_channels"] != len(channels) + 1:
            raise ValueError("TCP模拟源须为全部EEG通道加末尾1路trigger。")
        if device["unit"] != "synthetic_uV":
            raise ValueError("TCP模拟源的单位标识必须为synthetic_uV。")
    collection = cfg["collection"]
    for key in ("initial_rest_seconds", "prepare_seconds", "task_seconds", "rest_seconds", "block_rest_seconds", "tail_seconds"):
        value = float(collection.get(key, 1.0 if key == "tail_seconds" else 0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key}须为非负有限数值。")
    if int(collection["loops"]) < 1 or collection["rest_seconds"] < 0 or collection["initial_rest_seconds"] < 0:
        raise ValueError("采集轮数至少为1，静息时长不能为负。")
    if collection["task_seconds"] < window["seconds"] + window["task_offset_seconds"]:
        raise ValueError("采集任务时长不足以产生完整训练窗口。")
    return cfg


def load_config(path: str | Path) -> dict:
    try:
        return validate_config(json.loads(Path(path).read_text(encoding="utf-8-sig")))
    except KeyError as exc:
        raise ValueError(f"配置缺少字段：{exc}") from exc


def input_contract(cfg: dict) -> dict:
    """源种类、地址和训练时长不属于模型输入，但通道/单位/标签必须相符。"""
    return {"sample_rate": float(cfg["device"]["sample_rate"]),
            "channels": list(cfg["device"]["channels"]),
            "unit": cfg["device"]["unit"],
            "preprocess": cfg["preprocess"], "window": cfg["window"],
            "classes": cfg["classes"], "paradigm_id": cfg.get("paradigm_id"),
            "recording_condition": cfg.get('session_metadata',{}).get('recording_context',{}).get('condition'),
            "protocol_contract": cfg.get("protocol_contract")}


def require_compatible(reference: dict, other: dict) -> None:
    first, second = input_contract(reference), input_contract(other)
    mismatch = [key for key in first if first[key] != second[key]]
    if mismatch:
        raise ValueError("模型与数据的输入约定不一致：" + ", ".join(mismatch))


def write_json(path: str | Path, payload) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
