# -*- coding: utf-8 -*-
"""Explicit, reviewed label mapping (source label id -> model output column)."""
from __future__ import annotations

from typing import Any

from . import contracts as C


def normalize_rows(rows: Any) -> list[dict[str, Any]]:
    if rows is None:
        raise C.ContractError("缺少标签映射（mapping.rows）")
    if not isinstance(rows, list) or not rows:
        raise C.ContractError("标签映射不能为空")
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in rows:
        if not isinstance(raw, dict):
            raise C.ContractError(f"映射行不是对象：{raw!r}")
        try:
            label_id = int(raw["id"])
        except (KeyError, TypeError, ValueError):
            raise C.ContractError(f"映射行缺少整数 id：{raw!r}")
        if label_id in seen:
            raise C.ContractError(f"源标签 {label_id} 出现了多行映射")
        seen.add(label_id)
        target = raw.get("target", None)
        if target is not None and target != "":
            try:
                target = int(target)
            except (TypeError, ValueError):
                raise C.ContractError(f"源标签 {label_id} 的目标列不是整数：{target!r}")
            if not 0 <= target < C.OUTPUT_COUNT:
                raise C.ContractError(f"源标签 {label_id} 的目标列越界：{target}")
        else:
            target = None
        unused = bool(raw.get("unused", False))
        if target is not None and unused:
            raise C.ContractError(f"源标签 {label_id} 同时给了目标列和「不使用」")
        # A target assignment is itself an explicit review.  A blank/unused row
        # needs its own confirmation flag; "untouched" must never count.
        if "reviewed" in raw:
            reviewed = bool(raw["reviewed"])
        else:
            reviewed = target is not None and not unused
        out.append({"id": label_id, "target": target, "unused": unused,
                    "reviewed": reviewed,
                    "hint": str(raw.get("hint") or "").strip() or None})
    return out


def validate_mapping(inventory: list[dict[str, Any]], rows: Any,
                     outputs: list[dict[str, Any]], *,
                     require_all_outputs: bool = True,
                     require_reviewed: bool = True,
                     many_to_one_confirmed: bool = False) -> dict[str, Any]:
    """Validate one session's mapping; return the effective id -> column map.

    ``rows`` must cover every source label id that actually occurs.  A blank
    target is *not* the same as "unused": the blank must be explicitly marked
    ``unused: true`` (reviewed) or the job is refused.
    """
    present = {int(item["id"]): item for item in inventory}
    parsed = normalize_rows(rows)
    given = {int(row["id"]) for row in parsed}
    missing = sorted(set(present) - given)
    if missing:
        raise C.ContractError(f"以下源标签还没有映射：{missing}")
    extra = sorted(given - set(present))
    if extra:
        raise C.ContractError(f"映射里出现了数据中不存在的源标签：{extra}")

    effective: dict[int, int | None] = {}
    by_target: dict[int, list[int]] = {}
    warnings: list[str] = []
    counts: dict[int, int] = {i: 0 for i in range(C.OUTPUT_COUNT)}
    for row in parsed:
        label_id = int(row["id"])
        target = row["target"]
        if require_reviewed and not row.get("reviewed"):
            raise C.ContractError(
                f"源标签 {label_id} 未复核：请选择目标列并确认，或显式标记「不使用」")
        if target is None and not row["unused"]:
            effective[label_id] = None
            continue
        effective[label_id] = target
        if target is not None:
            by_target.setdefault(target, []).append(label_id)
            counts[target] += int(present[label_id].get("count", 0))

    many = {t: ids for t, ids in by_target.items() if len(ids) > 1}
    if many and not many_to_one_confirmed:
        detail = "；".join(
            f"列 {t} ← 源标签 {ids}" for t, ids in sorted(many.items()))
        raise C.ContractError(f"存在多对一映射，必须显式确认：{detail}")
    for target, ids in sorted(many.items()):
        names = {present[i].get("hint") for i in ids if present[i].get("hint")}
        if len(names) > 1:
            warnings.append(
                f"列 {target} 合并了名称不同的源标签 {ids}：{sorted(names)}（已确认）")

    if require_all_outputs:
        empty = [i for i, n in counts.items() if n == 0]
        if empty:
            names = [outputs[i]["name"] for i in empty]
            raise C.ContractError(f"以下输出列没有可用源标签：{list(zip(empty, names))}")
    return {"map": effective, "counts": counts, "many_to_one": many,
            "warnings": warnings}


def effective_map(rows: Any) -> dict[str, int | None]:
    parsed = normalize_rows(rows)
    return {str(int(r["id"])): r["target"] for r in parsed}
