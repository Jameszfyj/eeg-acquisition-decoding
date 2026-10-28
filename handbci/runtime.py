"""连续解码、机械手模拟输出、完成轮次后的有标签校准回放。"""
from __future__ import annotations

import json
from pathlib import Path
import time
import uuid

import numpy as np

from .acquisition import CollectionCancelled, connect, ensure_healthy, make_recorder
from .config import require_compatible, validate_participant_id, write_json
from .control import DecisionGate, MockHandDriver
from .data import build_dataset, load_session, new_session, seal_session, trials_from_events
from .preprocess import StreamPreprocessor
from .training import load_model, predict_inputs, save_model, transform_windows, tune_arrays


class OnlineDecoder:
    """仅接收电压和采样索引；接口不接收目标类别。"""
    def __init__(self, payload, model, device="cpu"):
        self.payload, self.model, self.device = payload, model, device
        self.stream = StreamPreprocessor(payload["config"])
        self.n_time = payload["architecture"]["n_times"]
        self.reset()

    def reset(self, first_index=0):
        self.stream.reset(first_index)
        self.history = np.empty((self.payload["architecture"]["n_channels"], 0), np.float32)
        self.indices = np.empty(0, np.int64)
        self.last_window = None
        self.last_prediction = -float("inf")

    def recover_stream(self, first_source_index, reason):
        self.reset(first_source_index)

    def process(self, raw, indices):
        processed, retained = self.stream.process(raw, indices)
        if not len(retained):
            return None
        self.history = np.concatenate([self.history, processed], axis=1)[:, -self.n_time:]
        self.indices = np.concatenate([self.indices, retained])[-self.n_time:]
        now = float(indices[-1]) / self.stream.fs
        interval = self.payload["config"].get("decision", {}).get("update_seconds", 0.1)
        if self.history.shape[1] < self.n_time or now - self.last_prediction < interval - 1e-9:
            return None
        inputs = transform_windows(self.payload, self.history[None])
        p = predict_inputs(self.model, inputs, self.device)[0]
        self.last_prediction = now
        self.last_window = self.stream.window_bounds(self.indices, indices[-1])
        return now, p


def make_gate(config):
    settings = config.get("decision", {})
    keys = ("confidence", "margin", "stable_seconds", "timeout_seconds", "max_attempts")
    return DecisionGate(config["classes"], **{k: settings[k] for k in keys if k in settings})


def run_replay(model_path, session_path, output_root, *, device="cpu", chunk_seconds=0.04,
               adapt=False, adapt_epochs=1, replay_paths=None, progress=print):
    payload, model = load_model(model_path, device)
    record = load_session(session_path)
    require_compatible(payload["config"], record["config"])
    path = new_session(output_root, ("adapt_" if adapt else "replay_") + uuid.uuid4().hex[:10])
    decoder = OnlineDecoder(payload, model, device)
    decoder.reset(int(record["indices"][0]))
    gate = make_gate(payload["config"])
    driver = MockHandDriver(path / "commands.jsonl")
    events = sorted(record["events"], key=lambda e: e["sample_index"])
    trials = trials_from_events(events, len(payload["config"]["classes"]))
    cursor, in_task = 0, False
    predictions = []
    full_target = build_dataset([session_path], payload["config"]) if adapt else None
    replay_ds = build_dataset(replay_paths, payload["config"]) if replay_paths else None
    completed, rounds = [], []
    n_classes = len(payload["config"]["classes"])
    if adapt:
        heldout = set(payload.get("training", {}).get("valid_groups", [])) | set(payload.get("training", {}).get("test_groups", []))
        if heldout & {r["group"] for r in full_target.rows}:
            raise ValueError("不能把原模型的验证/测试会话用于更新；请使用新的校准会话。")
    step = max(1, int(chunk_seconds * decoder.stream.fs))
    next_round = n_classes
    with (path / "predictions.jsonl").open("x", encoding="utf-8") as log:
        for start in range(0, len(record["indices"]), step):
            ix = record["indices"][start:start + step]
            now = ix[-1] / decoder.stream.fs
            while cursor < len(events) and events[cursor]["sample_index"] <= ix[-1]:
                event = events[cursor]
                # 只使用提示起止时刻和试次编号，不把目标类别传给判定器。
                if event["kind"] == "cue":
                    gate.begin_trial(event["trial_id"], event["sample_index"] / decoder.stream.fs)
                    in_task = True
                elif event["kind"] == "task_end":
                    gate.end_trial()
                    in_task = False
                cursor += 1
            prediction = decoder.process(record["eeg"][:, start:start + step], ix)
            if prediction is not None:
                stamp, probabilities = prediction
                command = gate.update(probabilities, stamp)
                row = {"sample_index": int(ix[-1]), "time_seconds": stamp,
                       "probabilities": probabilities.tolist(), "model_round": len(rounds)}
                predictions.append(row)
                log.write(json.dumps(row) + "\n")
                if command is not None:
                    driver.execute(command)
            if adapt:
                # 等待名义滤波延迟后再释放完整试次；只更新已完成试次的数据。
                completed = [t["trial_id"] for t in trials if t["end"] / decoder.stream.fs + decoder.stream.delay_seconds <= now]
                if len(completed) >= next_round and not in_task:
                    available = set(completed)
                    rows = [i for i, r in enumerate(full_target.rows) if r["trial_id"] in available]
                    target = full_target.subset(rows)
                    if len(set(target.y.tolist())) == n_classes:
                        report = tune_arrays(payload, model, target, epochs=adapt_epochs, device=device,
                                             replay=replay_ds, mode="head", seed=42 + len(rounds))
                        report.update({"released_trials": list(completed), "release_time_seconds": now})
                        rounds.append(report)
                        snapshot = path / f"candidate_round_{len(rounds):03d}.pt"
                        payload["calibration"] = {"kind": "labelled_replay", "parent": str(Path(model_path).resolve()), **report}
                        save_model(snapshot, payload, model)
                        progress(f"已完成第{len(rounds)}轮回放校准，仅使用{len(completed)}个已结束试次。")
                        gate.clear_candidate()
                        next_round = len(completed) + n_classes
    targets = {t["trial_id"]: t["label"] for t in trials}
    commands = {c["trial_id"]: c for c in driver.commands}
    correct = sum(c["class_id"] == targets[tid] for tid, c in commands.items())
    report = {"source": record["metadata"]["source"], "driver": "mock", "physical_action": False,
              "replay_is_wall_clock_realtime": False, "n_trials": len(trials), "n_commands": len(commands),
              "correct_first_commands": int(correct), "first_command_success_rate": correct / len(trials),
              "abstained_trials": len(trials) - len(commands), "n_predictions": len(predictions),
              "adaptation_rounds": rounds,
              "latency_seconds": [c["decision_time"] - next(t["start"] for t in trials if t["trial_id"] == tid) / decoder.stream.fs
                                  for tid, c in commands.items()],
              "interpretation": "模拟信号结果只说明软件链路；窗口预测是重叠的，不是独立试次。"}
    write_json(path / "report.json", report)
    return path, report


def run_live(model_path, config, output_root, *, trials=3, device="cpu", progress=print,
             stop_event=None, on_update=None):
    from contextlib import nullcontext
    from .receiver_access import ReceiverLease
    source = config["device"]
    access = (ReceiverLease(source.get("host", "127.0.0.1"), source.get("port", 8712))
              if source["source"] == "neuracle" else nullcontext())
    with access:
        return _run_live(model_path, config, output_root, trials=trials, device=device,
                         progress=progress, stop_event=stop_event, on_update=on_update)


def _run_live(model_path, config, output_root, *, trials=3, device="cpu", progress=print,
              stop_event=None, on_update=None):
    """实时时钟采集和推理；可回传网页状态、停止保存，驱动固定为mock。"""
    payload, model = load_model(model_path, device)
    require_compatible(payload["config"], config)
    name = "live_" + time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    participant = config.get("session_metadata", {}).get("participant_id")
    if participant is not None:
        name += "_" + validate_participant_id(str(participant))
    path = new_session(output_root, name)
    recorder = make_recorder(config)
    decoder = OnlineDecoder(payload, model, device)
    gate, driver = make_gate(config), MockHandDriver(path / "commands.jsonl")
    last_index, generation, trial_number = -1, None, 0
    events, complete, error = [], False, ""
    last_arrival, last_update = time.monotonic(), -float("inf")

    def check_stop():
        if stop_event is not None and stop_event.is_set():
            raise CollectionCancelled("已停止在线解码，收到的数据保留为未完成记录。")

    def emit(update):
        if on_update is not None:
            on_update(update)

    try:
        emit({"kind": "session", "session": str(path.resolve())})
        recorder.start_recording(str(path / "data.h5"), write_chunk_seconds=1)
        check_stop()
        connect(recorder, config)
        epoch = recorder.epoch
        origin = time.monotonic()
        next_start = config["preprocess"]["trim_seconds"] + config["window"]["seconds"]
        with (path / "predictions.jsonl").open("x", encoding="utf-8") as log:
            while True:
                check_stop()
                ensure_healthy(recorder, epoch)
                if generation is None:
                    generation = recorder.raw_buffer.generation
                raw, ix, overrun = recorder.raw_buffer.read_after_source_index(last_index, generation=generation)
                if overrun:
                    raise RuntimeError("在线处理落后或源被重置；停止本次会话。")
                if not len(ix):
                    if time.monotonic() - last_arrival > 5:
                        raise TimeoutError("5秒没有新脑电。")
                    recorder.raw_buffer.wait_for_new_data(timeout=0.02)
                    continue
                last_arrival = time.monotonic()
                last_index = int(ix[-1])
                now = time.monotonic() - origin
                if not gate.active and now >= next_start and decoder.history.shape[1] >= decoder.n_time:
                    if trial_number >= trials:
                        break
                    trial_number += 1
                    gate.begin_trial(trial_number, now)
                    next_start = now + gate.timeout + 1.0
                    events.append({"kind": "unlabelled_trial_start", "trial_id": trial_number,
                                   "sample_index": last_index, "wall_elapsed_seconds": now})
                    progress(f"试次{trial_number}/{trials}开始：请执行您选择的已训练脑任务；输出仅写入模拟日志。")
                started = time.perf_counter()
                pred = decoder.process(raw[:len(config["device"]["channels"])], ix)
                inference_ms = (time.perf_counter() - started) * 1000
                if pred is not None:
                    stamp, probabilities = pred
                    decision_clock = time.monotonic() - origin
                    command = gate.update(probabilities, decision_clock)
                    row = {"sample_index": last_index, "source_time_seconds": stamp,
                           "wall_elapsed_seconds": decision_clock, "probabilities": probabilities.tolist()}
                    log.write(json.dumps(row) + "\n")
                    emit({"kind": "prediction", "probabilities": probabilities.tolist(),
                          "inference_ms": inference_ms})
                    if command is not None:
                        event = driver.execute(command)
                        emit({"kind": "command", "command": dict(event)})
                        progress(f"模拟动作：{command.action}，置信度{command.confidence:.3f}")
                if gate.active and now - gate.start > gate.timeout:
                    gate.end_trial()
                if now - last_update >= 0.1:
                    factor = max(1, int(config["device"]["sample_rate"] / 100))
                    points = [[float(i) / config["device"]["sample_rate"], float(y)]
                              for i, y in zip(ix[::factor], raw[0, ::factor])]
                    emit({"kind": "progress", "samples_received": last_index + 1,
                          "trial": trial_number,
                          "phase": "decoding" if gate.active else "warming" if not trial_number else "interval",
                          "trial_remaining": max(0.0, gate.timeout - (now - gate.start)) if gate.active else 0.0,
                          "trace": points})
                    last_update = now
        complete = True
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            recorder.disconnect()
        except BaseException as exc:
            complete, error = False, error or str(exc)
            raise
        finally:
            if (path / "data.h5").is_file():
                seal_session(path, config, events, complete=complete, source=config["device"]["source"], error=error)
    write_json(path / "report.json", {"driver": "mock", "physical_action": False,
                                      "participant_id": participant, "model_path": str(Path(model_path).resolve()),
                                      "commands": len(driver.commands), "labelled_for_training": False})
    return path
