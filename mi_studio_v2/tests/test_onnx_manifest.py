# -*- coding: utf-8 -*-
from pathlib import Path
from types import SimpleNamespace

import pytest

from mi_studio_v2 import contracts as C
from mi_studio_v2.onnx_export import build_config, build_manifest

PLATFORM = Path("F:/project/机械手2026/eeg-acquisition-decoding")


def _model():
    return SimpleNamespace(outputs=C.default_outputs(),
                           channels=list(C.MODEL_CHANNELS),
                           model_labels=[2, 0, 1, 3, 4, 5])


def test_manifest_config_passes_platform_validation():
    if not (PLATFORM / "handbci/config.py").is_file():
        pytest.skip("platform checkout not available")
    import sys
    sys.path.insert(0, str(PLATFORM))
    from handbci.config import validate_config
    from handbci.model_bundle import validate_manifest

    prof = C.profile("platform_replay_v1")
    cfg = build_config(_model(), prof)
    assert cfg["device"]["stream_channels"] == 33
    assert cfg["collection"]["task_seconds"] >= 4.0
    validated = validate_config(cfg)
    manifest = build_manifest(_model(), prof, "studio test")
    out = validate_manifest(manifest)
    assert out["participant_id"] == "P01"
    assert len(out["config"]["classes"]) == 6
    assert out["input"]["pipeline"] == "ifnet_filterbank"


def test_manifest_rejects_non_platform_profile_window_claim():
    # l100d250h4 is still a valid graph, but the manifest must not pretend the
    # platform replay pipeline reproduces its windows.
    prof = C.profile("l100d250h4_v1")
    manifest = build_manifest(_model(), prof, "legacy")
    assert manifest["studio"]["profile"] == "l100d250h4_v1"
    assert manifest["studio"]["profile_fingerprint"] == C.profile_fingerprint(prof)
