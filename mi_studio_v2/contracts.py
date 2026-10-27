# -*- coding: utf-8 -*-
"""Frozen contracts for the MI source/calibration studio (v2).

v2 与 v1 的唯一必需差别是取窗口径：v1 的 offsets 固定在 cue 之后
(0.6 / 1.0 / 1.4 s)，v2 把锚点换成 **prepare（目标提示出现）**，窗口为
``[anchor − P_history, task_end + post]``，在 ``[anchor, task_end + post]`` 内按
``stride_seconds`` 滑动切片（默认 0.5 s）。v1 的两个 profile 保持原样不动。

Everything that decides *which raw samples become which model column* lives
here, so the CLI, the web server, the tests and the model sidecar can never
drift apart silently.  Two preprocessing profiles are deliberately kept
separate:

``platform_replay_v1``
    The deployment contract used by ``offline_calib_20261027`` and the
    ``handbci.onnx.v1`` bundle: the column timeline is *delay compensated*
    (``raw / fs - 0.5 s``) and windows are cut at 1.0 / 1.4 / 1.8 s.

``l100d250h4_v1``
    The historical project contract (``预处理后数据_L100D250H4``): the column
    timeline is the *device clock* (no group-delay compensation) and windows
    are cut at 0.6 / 1.0 / 1.4 s.

A model records its profile; calibration inherits the source model's profile
and may never silently re-cut the data with another one.
"""
from __future__ import annotations

from typing import Any

STUDIO_VERSION = "mi_studio/2.0"
MODEL_FORMAT = "jixieshou.six_class_offline.v1"
SIDECAR_FORMAT = "mi_studio.model.v1"
JOB_FORMAT = "mi_studio.job.v1"

OUTPUT_COUNT = 6

#: 30 channels in model-input order (the same order as the deployed bundle).
MODEL_CHANNELS = [
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6",
    "Cz", "C3", "C4", "T7", "T8", "CP1", "CP2", "CP5", "CP6", "Pz", "P3",
    "P4", "P7", "P8", "PO3", "PO4", "Oz", "O1", "O2",
]

#: 32-device channels of the project amplifier, in raw array order.
DEVICE_CHANNELS = [
    "Cz", "CP1", "C4", "Pz", "O1", "Fp1", "CP6", "O2", "CP2", "P8", "C3", "P7",
    "Fp2", "F8", "CP5", "F7", "A1", "FC5", "A2", "P4", "T7", "F3", "T8", "FC2",
    "PO4", "FC6", "P3", "PO3", "Oz", "F4", "FC1", "Fz",
]

#: cargame records store the same 32 channels in this (montage) order.
CARGAME_CHANNELS = [
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6",
    "Cz", "C3", "C4", "T7", "T8", "A1", "A2", "CP1", "CP2", "CP5", "CP6",
    "Pz", "P3", "P4", "P7", "P8", "PO3", "PO4", "Oz", "O1", "O2",
]

#: Defaults shown in the UI.  They are *suggestions*, never an implicit
#: declaration: a job is refused until the operator confirms every output.
DEFAULT_OUTPUTS = [
    {"index": 0, "label_id": 2, "name": "左旋臂", "description": "左臂旋转"},
    {"index": 1, "label_id": 0, "name": "左握拳", "description": "左手握拳"},
    {"index": 2, "label_id": 1, "name": "右握拳", "description": "右手握拳"},
    {"index": 3, "label_id": 3, "name": "右旋臂", "description": "右臂旋转"},
    {"index": 4, "label_id": 4, "name": "双脚", "description": "双脚运动想象"},
    {"index": 5, "label_id": 5, "name": "舌头", "description": "舌头运动想象"},
]

#: cargame ``TASK_SPECS`` -> Chinese project names.  Only cue codes (never the
#: task-end code 9, the start markers 100/101, or control codes) appear here.
CARGAME_CODE_HINTS = {
    1: "左手抓握",
    2: "右手抓握",
    3: "舌头",
    4: "双脚",
    5: "双手抓握",
    18: "左握拳",
    19: "左旋臂",
    20: "右握拳",
    21: "右旋臂",
}
CARGAME_CUE_CODES = frozenset(CARGAME_CODE_HINTS)
CARGAME_TASK_END_CODE = 9

#: cargame log header task-code letters -> event id (from TASK_SPECS).
CARGAME_LETTERS = {
    "L": 1, "R": 2, "T": 3, "F": 4, "D": 5, "X": 10,
    "Lq": 18, "Lx": 19, "Rq": 20, "Rx": 21,
}

PROFILES: dict[str, dict[str, Any]] = {
    "platform_replay_v1": {
        "id": "platform_replay_v1",
        "label": "平台回放 · 延迟补偿（1.0 / 1.4 / 1.8 s）",
        "short": "平台回放",
        "timeline": "delay_compensated",
        "delay_seconds": 0.5,
        "offsets": [1.0, 1.4, 1.8],
        "boundary": {"mode": "legacy_replay", "task_end_slack_s": 0.5},
        "window_seconds": 2.0,
        "fs_in": 1000.0,
        "fs_out": 250.0,
        "lowpass_hz": 100.0,
        "highpass_hz": 4.0,
        "numtaps": 201,
        "trim_seconds": 1.0,
        "window": [201, 100.0, 4.0, 201, 250.0],
        "note": ("列时刻 = 原始样点 / 1000 − 0.5 s；窗 1.0/1.4/1.8 s。"
                 "部署包与 offline_calib_20261027 使用此口径。"),
    },
    "l100d250h4_v1": {
        "id": "l100d250h4_v1",
        "label": "L100D250H4 · 设备时钟（0.6 / 1.0 / 1.4 s）",
        "short": "L100D250H4",
        "timeline": "device_clock",
        "delay_seconds": 0.0,
        "offsets": [0.6, 1.0, 1.4],
        "boundary": {"mode": "strict_task", "task_end_slack_s": 0.0},
        "window_seconds": 2.0,
        "fs_in": 1000.0,
        "fs_out": 250.0,
        "lowpass_hz": 100.0,
        "highpass_hz": 4.0,
        "numtaps": 201,
        "trim_seconds": 1.0,
        "window": [201, 100.0, 4.0, 201, 250.0],
        "note": ("列时刻 = 原始样点自己的设备时刻（不补群延迟）；窗 0.6/1.0/1.4 s。"
                 "与 预处理后数据_L100D250H4 一致。"),
    },
    "l100d250h4_pre_v2": {
        "id": "l100d250h4_pre_v2",
        "label": "L100D250H4 v2 · prepare 锚点（[−P−1, T+1]，0.5 s stride）",
        "short": "prepare 锚点 v2",
        "timeline": "device_clock",
        "delay_seconds": 0.0,
        "anchor": "prepare",
        "pre_seconds": 1.0,
        "post_seconds": 1.0,
        "stride_seconds": 0.5,
        "min_windows_per_trial": 1,
        "offsets": [],
        "boundary": {"mode": "anchor_post", "task_end_slack_s": 1.0},
        "window_seconds": 2.0,
        "fs_in": 1000.0,
        "fs_out": 250.0,
        "lowpass_hz": 100.0,
        "highpass_hz": 4.0,
        "numtaps": 201,
        "trim_seconds": 1.0,
        "window": [201, 100.0, 4.0, 201, 250.0],
        "note": ("锚点 = prepare（目标提示）的 sample_index；窗口支持 "
                 "[anchor−1 s, task_end+1 s]，1 s 只是因果 FIR 历史；有效段 "
                 "[anchor, task_end+1 s] 内按 0.5 s 步长切 2 s 窗。"
                 "缺 prepare 元数据时按 trials_json 的 preview_sample、"
                 "metadata 的 prepare_seconds、profile 默认 2.0 s 依次回退。"),
    },
}

DEFAULT_PROFILE = "l100d250h4_pre_v2"

#: Human-facing names for the six global labels used by the project.  The
#: studio never maps by number across formats; these are hints only.
GLOBAL_NAMES = {
    0: "左握拳", 1: "右握拳", 2: "左旋臂", 3: "右旋臂",
    4: "双脚", 5: "舌头", 6: "双手", 7: "双手旋臂",
    8: "左手抓握", 9: "右手抓握", 10: "双手抓握",
}


class ContractError(ValueError):
    """A job violates a frozen contract (timeline, mapping, outputs, ...)."""


def profile(name: str | None) -> dict[str, Any]:
    key = str(name or DEFAULT_PROFILE)
    if key not in PROFILES:
        raise ContractError(f"未知预处理口径 {key!r}；可选 {sorted(PROFILES)}")
    return dict(PROFILES[key])


def default_outputs() -> list[dict[str, Any]]:
    return [dict(item) for item in DEFAULT_OUTPUTS]


def validate_outputs(outputs: Any) -> list[dict[str, Any]]:
    """Normalise and validate an explicit six-output declaration.

    Each slot is ``{index, label_id, name, description}``; names must be
    non-empty and label ids unique.  The list order *is* the model column
    order (``model_labels``).
    """
    if not isinstance(outputs, list) or len(outputs) != OUTPUT_COUNT:
        raise ContractError(f"必须显式给出 {OUTPUT_COUNT} 个输出维度，收到 {outputs!r}")
    seen_ids: set[int] = set()
    clean: list[dict[str, Any]] = []
    for index, raw in enumerate(outputs):
        if not isinstance(raw, dict):
            raise ContractError(f"输出维度 {index} 不是对象")
        if "index" in raw and raw["index"] is not None:
            try:
                declared_index = int(raw["index"])
            except (TypeError, ValueError):
                raise ContractError(f"输出维度 {index} 的 index 不是整数")
            if declared_index != index:
                raise ContractError(
                    f"输出列表第 {index} 项声明 index={declared_index}；拒绝重排，"
                    "列顺序必须与列表顺序一致")
        name = str(raw.get("name", "")).strip()
        if not name:
            raise ContractError(f"输出维度 {index} 的名称为空")
        try:
            label_id = int(raw.get("label_id"))
        except (TypeError, ValueError):
            raise ContractError(f"输出维度 {index} 的 label_id 不是整数")
        if label_id in seen_ids:
            raise ContractError(f"label_id {label_id} 重复出现在两个输出维度")
        seen_ids.add(label_id)
        clean.append({
            "index": index,
            "label_id": label_id,
            "name": name,
            "description": str(raw.get("description", "")).strip(),
        })
    return clean


def outputs_fingerprint(outputs: list[dict[str, Any]]) -> str:
    import hashlib
    import json

    blob = json.dumps(outputs, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def profile_fingerprint(prof: dict[str, Any]) -> str:
    import hashlib
    import json

    keys = ("id", "timeline", "offsets", "boundary", "window_seconds", "fs_in",
            "fs_out", "lowpass_hz", "highpass_hz", "numtaps", "trim_seconds",
            "anchor", "pre_seconds", "post_seconds", "stride_seconds")
    blob = json.dumps({k: prof.get(k) for k in keys}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
