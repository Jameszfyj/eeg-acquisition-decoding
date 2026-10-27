"""模拟数据端到端演示，包含训练、独立评价、校准、控制回放。"""
from pathlib import Path
import uuid

from .config import load_config, write_json
from .data import make_synthetic_session
from .runtime import run_replay
from .training import calibrate, train


def run_demo(output_root, epochs=15, progress=print):
    root = Path(output_root) / ("demo_" + uuid.uuid4().hex[:10])
    config = load_config(Path(__file__).resolve().parents[1] / "configs/demo_6class.json")
    paths = [make_synthetic_session(config, root / "records", f"session_{i + 1}", 100 + i) for i in range(5)]
    model = root / "models/base.pt"
    training = train(paths[:3], model, epochs=epochs, patience=10, progress=progress)
    calibrated = root / "models/calibrated_head.pt"
    calibration = calibrate(model, [paths[3]], calibrated, valid_paths=[paths[4]],
                            replay_paths=[paths[0]], mode="head", epochs=2)
    replay_path, replay = run_replay(model, paths[4], root / "replays", progress=progress)
    adapt_path, adapted = run_replay(model, paths[3], root / "replays", adapt=True,
                                     replay_paths=[paths[0]], progress=progress)
    report = {"notice": "全部为人工模拟信号；没有驱动机械手，不能作为比赛成绩或真实脑电效果。",
              "output": str(root.resolve()), "records": [str(p.resolve()) for p in paths],
              "base_model": str(model.resolve()), "calibrated_model": str(calibrated.resolve()),
              "training": training, "calibration": calibration,
              "replay": {"output": str(replay_path), **replay},
              "adapt_replay": {"output": str(adapt_path), **adapted}}
    write_json(root / "DEMO_REPORT.json", report)
    return report
