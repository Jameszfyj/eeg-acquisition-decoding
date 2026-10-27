# -*- coding: utf-8 -*-
import shutil

import numpy as np
import pytest

from mi_studio_v2 import adapters
from mi_studio_v2.contracts import ContractError


def test_cargame_trial_coordinates_are_source_indices(offline_session, make_cargame):
    folder = make_cargame("cg_gap", n_trials=2, gap_at=(0, 1000))
    session = adapters.load_cargame(folder)
    assert len(session.trials) == 2
    first = session.trials[0]
    assert first["start"] == int(session.indices[first["row_start"]])
    assert first["end"] == int(session.indices[first["row_end"]])
    assert first["start"] > 1000
    assert len(session.segments) >= 2


def test_offline_hints_and_incomplete_flag(offline_session, make_offline):
    session = adapters.load_offline(offline_session)
    assert [item["hint"] for item in session.label_inventory()] == ["左握拳", "右握拳"]
    incomplete = adapters.load_offline(make_offline("inc", incomplete=True))
    assert incomplete.complete is False
    assert any("未完整" in w for w in incomplete.warnings)


def test_online_external_trials_without_source_index(make_online):
    folder = make_online("online_rows", with_source_index=False, external_trials=True)
    session = adapters.load_online_decode(folder)
    assert session.trials
    assert session.trials[0]["start"] < 10_000
    assert any("没有 source_index" in w or "存储行" in w for w in session.warnings)


def test_online_calibration_hint_from_model_selection(make_online, tmp_path):
    import json

    folder = make_online("calib", with_source_index=False, external_trials=False)
    (folder / "model_selection.json").write_text(json.dumps({
        "classes": [{"id": i, "name": n} for i, n in enumerate(
            ["左握拳", "右握拳", "左旋臂", "右旋臂", "双脚", "舌头"])],
        "session_purpose": "calibration",
    }, ensure_ascii=False), encoding="utf-8")
    events = []
    for i in range(2):
        start = 5000 + i * 7000
        end = start + 4000
        events.append({"kind": "imagery", "trial": i + 1, "round": 1,
                       "task_id": i % 6, "source_index": start})
        events.append({"kind": "task_end", "trial": i + 1,
                       "source_index": end})
    (folder / "calibration.json").write_text(json.dumps({
        "complete": True, "events": events,
    }, ensure_ascii=False), encoding="utf-8")
    session = adapters.load_online_calibration(folder)
    assert session.trials
    assert session.label_inventory()[0]["hint"] in (
        "左握拳", "右握拳", "左旋臂", "右旋臂", "双脚", "舌头")


def test_content_fingerprint_survives_copy(offline_session, tmp_path):
    copied = tmp_path / "copy_of_session"
    shutil.copytree(offline_session, copied)
    first = adapters.load_offline(offline_session)
    second = adapters.load_offline(copied)
    assert first.fingerprint() == second.fingerprint()


def test_build_bank_rejects_duplicate_recordings(offline_session, tmp_path):
    import shutil

    from mi_studio_v2 import contracts as C
    from mi_studio_v2.preprocess import build_bank

    copied = tmp_path / "same_content"
    shutil.copytree(offline_session, copied)
    outputs = C.default_outputs()
    mapping = {"rows": [
        {"id": 0, "target": 1}, {"id": 1, "target": 2},
    ], "many_to_one_confirmed": False}
    with pytest.raises(ContractError, match="重复导入"):
        build_bank([
            {"path": str(offline_session), "mapping": mapping, "_outputs": outputs},
            {"path": str(copied), "mapping": mapping, "_outputs": outputs},
        ], "platform_replay_v1", require_all_outputs=False, log=lambda *a: None)
