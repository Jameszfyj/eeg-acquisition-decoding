# -*- coding: utf-8 -*-
import numpy as np

from mi_studio_v2.preprocess import WindowBank
from mi_studio_v2.training import (_stratified_trial_split, _trial_table,
                                _windows_for, trial_metrics)


def _bank():
    n_class = 6
    sessions = ["s1", "s2"]
    keys, dims, sessions_col = [], [], []
    X, y, tidx, off, day, session_per_window = [], [], [], [], [], []
    for si, session in enumerate(sessions):
        for cls in range(n_class):
            key = f"{session}#abc|{cls + 1}"
            keys.append(key)
            dims.append(cls)
            sessions_col.append(session)
            for window in range(3):
                X.append(np.full((30, 500), si * 10 + cls, np.float32))
                y.append(cls)
                tidx.append(len(keys) - 1)
                off.append(1.0 + 0.4 * window)
                day.append("20260101")
                session_per_window.append(session)
    return WindowBank(
        X=np.stack(X), y=np.asarray(y), label=np.asarray(y),
        session=np.asarray(session_per_window), trial_key=np.asarray(
            [keys[i] for i in tidx]), trial_index=np.asarray(tidx),
        offset=np.asarray(off), day=np.asarray(day),
        channels=["c"] * 30,
        meta={"trial_keys": keys, "trial_dims": dims, "profile": {"id": "x"},
              "n_time": 500, "fs": 250.0},
    )


def test_trial_table_and_windows_stay_grouped():
    bank = _bank()
    table = _trial_table(bank)
    assert len(table["dim"]) == 12
    X, y, tidx, n = _windows_for(bank, np.array([0, 1]))
    assert X.shape == (6, 30, 500)
    assert n == 2
    assert list(tidx) == [0, 0, 0, 1, 1, 1]
    assert np.all(y == np.asarray([0, 0, 0, 1, 1, 1]))


def test_stratified_split_is_disjoint_and_covers_classes():
    dims = np.repeat(np.arange(6), 4)
    train, valid = _stratified_trial_split(dims, 0.25, seed=0)
    assert not set(train.tolist()) & set(valid.tolist())
    assert len(train) + len(valid) == len(dims)
    for c in range(6):
        assert (dims[train] == c).sum() >= 1
        assert (dims[valid] == c).sum() >= 1


def test_metrics_report_missing_classes_as_na():
    metrics = trial_metrics(np.asarray([0, 0, 0]), np.asarray([0, 1, 0]))
    assert metrics["missing_classes"] == [1, 2, 3, 4, 5]
    assert metrics["balanced_accuracy"] == 2 / 3
    assert metrics["per_class_recall"][1] is None
