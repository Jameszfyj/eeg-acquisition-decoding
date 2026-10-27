"""Optional local preferences; personal files and paths stay outside Git."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_SETTINGS = ROOT / 'configs/workbench.local.json'


def local_settings():
    if not LOCAL_SETTINGS.is_file():
        return {}
    value = json.loads(LOCAL_SETTINGS.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict) or value.get('format') != 'handbci.workbench.local.v1':
        raise ValueError('本机工作台配置格式无效。')
    return value


def local_path(key, default):
    path = Path(local_settings().get(key, default))
    return path if path.is_absolute() else ROOT / path


def model_preferences(model_id):
    return local_settings().get('models', {}).get(model_id, {})


def calibration_protocol(cfg):
    value = cfg.get('session_metadata', {}).get('brain_calibration', {})
    rounds = value.get('rounds', 3)
    if isinstance(rounds, bool) or not isinstance(rounds, int) or not 1 <= rounds <= 100:
        raise ValueError('脑电校准轮数须为1至100的整数，每轮包含全部6类。')
    imagery = value.get('imagery_seconds', 4.)
    rest = value.get('rest_seconds', 3.)
    for duration, low, name in ((imagery, 4., '想象'), (rest, 1., '休息')):
        if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or not low <= duration <= 30):
            raise ValueError(f'脑电校准{name}时间须为{low:g}至30秒。')
    return dict(rounds=rounds, trials=rounds * 6, imagery_seconds=float(imagery),
                rest_seconds=float(rest), windows_per_trial=3,
                window_offsets_seconds=[.6, 1., 1.4])
