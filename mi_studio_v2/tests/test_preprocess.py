# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import numpy as np
import pytest

from mi_studio_v2 import contracts as C
from mi_studio_v2 import preprocess as P
from mi_studio_v2.adapters import Session, _segments, load_offline

PLATFORM = Path("F:/project/机械手2026/eeg-acquisition-decoding")


def _session(eeg, indices, segments, fs=1000.0):
    indices = np.asarray(indices, np.int64)
    return Session(
        session_id="synthetic", kind="offline_session", path="/tmp/synthetic",
        date="20260101", participant="P01", fs=fs,
        channels=list(C.DEVICE_CHANNELS), eeg=np.asarray(eeg, np.float32),
        indices=indices, device_ms=indices.astype(np.float64),
        segments=segments, trials=[],
    )


def test_platform_parity(offline_session):
    if not (PLATFORM / "handbci/preprocess.py").is_file():
        pytest.skip("platform checkout not available")
    sys.path.insert(0, str(PLATFORM))
    from handbci.preprocess import preprocess_record

    session = load_offline(offline_session)
    mine = P.preprocess_session(session, C.profile("platform_replay_v1"))
    cfg = session.metadata["config"]
    ref, retained = preprocess_record(cfg, session.eeg, session.indices)
    assert np.array_equal(mine["retained"], retained)
    assert float(np.abs(mine["values"] - ref).max()) < 1e-6


def test_segment_ids_are_renumbered_after_empty_segments():
    n = 20_000
    eeg = np.random.default_rng(0).normal(0, 2, (32, n)).astype(np.float32)
    indices = np.arange(n)
    session = _session(eeg, indices, segments=[(0, 0), (0, n)])
    out = P.preprocess_session(session, C.profile("platform_replay_v1"))
    assert out["segments"] and out["segments"][0]["id"] == 0
    assert out["seg_id"].max() == 0
    assert out["segments"][0]["col_stop"] == out["values"].shape[1]


def test_task_boundary_is_reported_for_legacy_profile(make_offline):
    folder = make_offline("legacy_boundary", n_trials=1, labels=[0], task_s=4.0)
    outputs = C.default_outputs()
    bank = P.build_bank([{"path": str(folder), "mapping": {"rows": [
        {"id": 0, "target": 0}], "many_to_one_confirmed": False},
        "_outputs": outputs}], "platform_replay_v1",
        require_all_outputs=False, log=lambda *a: None)
    assert bank.X.shape[0] == 3
    assert bank.meta["boundary"]["overshoot_windows"] >= 1
    assert bank.meta["boundary"]["max_overshoot_s"] > 0


def test_strict_profile_rejects_beyond_task_end(make_offline):
    folder = make_offline("strict_boundary", n_trials=1, labels=[0], task_s=3.0)
    outputs = C.default_outputs()
    with pytest.raises(C.ContractError, match="没有产出"):
        P.build_bank([{"path": str(folder), "mapping": {"rows": [
            {"id": 0, "target": 0}], "many_to_one_confirmed": False},
            "_outputs": outputs}], "l100d250h4_v1",
            require_all_outputs=False, log=lambda *a: None)


def test_gap_crossing_trial_is_rejected(make_offline):
    folder = make_offline("gap_cross", n_trials=2, labels=[0, 1], task_s=4.0,
                          gap_at=(13_000, 13_500))
    outputs = C.default_outputs()
    bank = P.build_bank([{"path": str(folder), "mapping": {"rows": [
        {"id": 0, "target": 0}, {"id": 1, "target": 1}],
        "many_to_one_confirmed": False}, "_outputs": outputs}],
        "platform_replay_v1", require_all_outputs=False, log=lambda *a: None)
    assert bank.meta["rejected"].get("trial_crosses_gap", 0) >= 1
    assert bank.X.shape[0] == 3


def test_incomplete_session_is_rejected(make_offline):
    folder = make_offline("incomplete", n_trials=1, labels=[0], incomplete=True)
    outputs = C.default_outputs()
    with pytest.raises(C.ContractError, match="未完整结束"):
        P.build_bank([{"path": str(folder), "mapping": {"rows": [
            {"id": 0, "target": 0}], "many_to_one_confirmed": False},
            "_outputs": outputs}], "platform_replay_v1",
            require_all_outputs=False, log=lambda *a: None)
