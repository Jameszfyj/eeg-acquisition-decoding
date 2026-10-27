# -*- coding: utf-8 -*-
import pytest

from mi_studio_v2 import contracts as C
from mi_studio_v2.mapping import validate_mapping
from mi_studio_v2.training import model_outputs


def inv(*ids):
    return [{"id": i, "hint": f"h{i}", "count": 2} for i in ids]


def test_output_index_must_match_position():
    outputs = C.default_outputs()
    bad = [dict(o) for o in outputs]
    bad[0]["index"] = 1
    with pytest.raises(C.ContractError, match="拒绝重排"):
        C.validate_outputs(bad)


def test_untouched_blank_is_not_reviewed():
    rows = [{"id": 0, "target": None}]
    with pytest.raises(C.ContractError, match="未复核"):
        validate_mapping(inv(0), rows, C.default_outputs(), require_all_outputs=False)
    rows = [{"id": 0, "target": None, "unused": True}]
    with pytest.raises(C.ContractError, match="未复核"):
        validate_mapping(inv(0), rows, C.default_outputs(), require_all_outputs=False)
    rows = [{"id": 0, "target": None, "unused": True, "reviewed": True}]
    got = validate_mapping(inv(0), rows, C.default_outputs(), require_all_outputs=False)
    assert got["map"][0] is None


def test_target_assignment_is_reviewed_and_many_to_one_needs_confirmation():
    rows = [{"id": 0, "target": 0}, {"id": 1, "target": 0}]
    with pytest.raises(C.ContractError, match="多对一"):
        validate_mapping(inv(0, 1), rows, C.default_outputs(), require_all_outputs=False)
    got = validate_mapping(inv(0, 1), rows, C.default_outputs(),
                           require_all_outputs=False, many_to_one_confirmed=True)
    assert got["map"] == {0: 0, 1: 0}


def test_missing_source_label_row_is_rejected():
    with pytest.raises(C.ContractError, match="还没有映射"):
        validate_mapping(inv(0, 1), [{"id": 0, "target": 0}],
                         C.default_outputs(), require_all_outputs=False)


def test_model_outputs_rejects_conflicting_metadata():
    payload = {"outputs": C.default_outputs(), "model_labels": [0, 1, 2, 3, 4, 5],
               "class_names": [o["name"] for o in C.default_outputs()]}
    with pytest.raises(C.ContractError, match="model_labels"):
        model_outputs(payload)
    payload.pop("outputs")
    payload["model_labels"] = [2, 0, 1, 3, 4, 5]
    got = model_outputs(payload)
    assert [o["label_id"] for o in got] == [2, 0, 1, 3, 4, 5]


def test_resolve_profile_requires_validated_metadata(tmp_path):
    from mi_studio_v2.training import resolve_model_profile

    prof, source = resolve_model_profile({"studio": {"profile": {"id": "l100d250h4_v1"}}})
    assert prof["id"] == "l100d250h4_v1" and source == "studio.profile"
    assert resolve_model_profile({})[0] is None
    model = tmp_path / "model.pt"
    model.write_bytes(b"x")
    spec = tmp_path / "spec.json"
    spec.write_text('{"provenance": {"pipeline": "platform_replay"}}', encoding="utf-8")
    prof, source = resolve_model_profile({}, model)
    assert prof["id"] == "platform_replay_v1" and "spec.json" in source
