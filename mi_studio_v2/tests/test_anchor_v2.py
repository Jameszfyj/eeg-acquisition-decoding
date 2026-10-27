# -*- coding: utf-8 -*-
"""v2 取窗口径：prepare 锚点 + [anchor−1 s, task_end+1 s] + 0.5 s stride。"""
import json

import h5py
import numpy as np
import pytest

from mi_studio_v2 import contracts as C
from mi_studio_v2 import preprocess as P
from mi_studio_v2.adapters import load_offline

PROFILE = "l100d250h4_pre_v2"


def _bank(folder, profile=PROFILE, labels=(0,)):
    outputs = C.default_outputs()
    rows = [{"id": i, "target": t} for i, t in enumerate(labels)]
    return P.build_bank([{"path": str(folder),
                          "mapping": {"rows": rows, "many_to_one_confirmed": False},
                          "_outputs": outputs}], profile,
                        require_all_outputs=False, log=lambda *a: None)


def _strip_prepare(folder):
    with h5py.File(folder / "data.h5", "a") as h:
        events = [json.loads(e) for e in h["events_json"].asstr()[:]]
        kept = [e for e in events if e.get("kind") != "prepare"]
        del h["events_json"]
        h.create_dataset("events_json",
                         data=[json.dumps(e, ensure_ascii=False) for e in kept],
                         dtype=h5py.string_dtype("utf-8"))


def test_prepare_anchor_windows_and_support(make_offline):
    folder = make_offline("v2_anchor", n_trials=1, labels=[0], task_s=4.0)
    trial = load_offline(folder).trials[0]
    assert trial["anchor"] == trial["start"] - 2400
    assert trial["anchor_kind"] == "prepare"

    bank = _bank(folder)
    # [anchor, task_end+1] = 7.4 s，2 s 窗、0.5 s 步长 -> 11 窗
    assert bank.X.shape[0] == 11
    assert bank.meta["anchor_kinds"] == {"prepare": 1}
    sup = sorted(bank.meta["supports"], key=lambda r: r["offset"])
    assert sup[0]["offset"] == pytest.approx(0.0)
    assert sup[-1]["offset"] == pytest.approx(5.0)
    assert sup[0]["cue_minus_anchor_s"] == pytest.approx(2.4)
    # 每个窗的支持段都从 anchor−1 s（因果历史）开始，且不越过 task_end+1 s
    for row in sup:
        assert row["support_start_s"] == pytest.approx(row["begin_s"] - 1.0)
        assert row["support_end_s"] <= row["task_end_s"] + 1.0 + 1e-6
    assert bank.meta["rejected"] == {}


def test_anchor_falls_back_to_prepare_seconds_without_prepare_events(make_offline):
    folder = make_offline("v2_fallback", n_trials=1, labels=[0], task_s=4.0)
    _strip_prepare(folder)
    trial = load_offline(folder).trials[0]
    assert trial["anchor"] is None and trial["prepare_seconds"] == pytest.approx(2.4)
    bank = _bank(folder)
    assert bank.meta["anchor_kinds"] == {"prepare_seconds": 1}
    assert bank.X.shape[0] == 11


def test_legacy_profile_keeps_three_offsets(make_offline):
    folder = make_offline("v2_legacy", n_trials=1, labels=[0], task_s=4.0)
    bank = _bank(folder, profile="l100d250h4_v1")
    assert bank.X.shape[0] == 3
    assert sorted(bank.offset.tolist()) == [0.6, 1.0, 1.4]


def test_anchor_is_snapped_to_the_250hz_grid(make_offline):
    """prepare 的 sample_index 有设备时钟抖动：锚点吸附到最近列，窗数不变。"""
    folder = make_offline("v2_snap", n_trials=1, labels=[0], task_s=4.0)
    trial = load_offline(folder).trials[0]
    # 把 prepare 挪开 2 ms（不在 4 样点栅格上）
    import h5py
    with h5py.File(folder / "data.h5", "a") as h:
        events = [json.loads(e) for e in h["events_json"].asstr()[:]]
        for e in events:
            if e.get("kind") == "prepare":
                e["sample_index"] = int(e["sample_index"]) + 2
        del h["events_json"]
        h.create_dataset("events_json",
                         data=[json.dumps(e, ensure_ascii=False) for e in events],
                         dtype=h5py.string_dtype("utf-8"))
    assert load_offline(folder).trials[0]["anchor"] == trial["anchor"] + 2
    bank = _bank(folder)
    assert bank.X.shape[0] == 11
    assert bank.meta["rejected"] == {}
    snap = {abs(row["anchor_snap_ms"]) for row in bank.meta["supports"]}
    assert max(snap) <= 4.0 and max(snap) > 0.0


def test_bank_roundtrip_on_disk(tmp_path, make_offline):
    folder = make_offline("v2_disk", n_trials=2, labels=[0, 1], task_s=4.0)
    out = tmp_path / "bank"
    outputs = C.default_outputs()
    rows = [{"id": 0, "target": 0}, {"id": 1, "target": 1}]
    bank = P.build_bank([{"path": str(folder),
                          "mapping": {"rows": rows, "many_to_one_confirmed": False},
                          "_outputs": outputs}], PROFILE,
                        require_all_outputs=False, log=lambda *a: None, out_dir=out)
    assert (out / "X.npy").is_file() and (out / "meta.npz").is_file()
    assert not (out / "_chunks").exists()
    again = P.WindowBank.open(out)
    assert again.X.shape == bank.X.shape
    assert np.array_equal(np.asarray(again.X), np.asarray(bank.X))
    assert again.meta["profile"]["id"] == PROFILE
    assert again.meta["anchor_kinds"] == {"prepare": 2}
