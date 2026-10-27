"""设备采集与提示；硬件打标、软件近似打标分别记录，不自动混用。"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time
import uuid

import numpy as np

from .config import validate_participant_id
from .data import new_session, seal_session
from .receiver_access import ReceiverLease
from .triggerbox import configure_triggerbox, open_network_triggerbox
from .vendor.neuracle_recorder import EEGRecorder, decode_trigger_values


class CollectionCancelled(RuntimeError):
    """操作员停止；保留文件并标记为未完成。"""


def make_recorder(config: dict) -> EEGRecorder:
    device = config["device"]
    return EEGRecorder(n_channels=device["stream_channels"], n_eeg=len(device["channels"]),
                       srate=device["sample_rate"], buffer_seconds=30,
                       host=device.get("host", "127.0.0.1"), port=device.get("port", 8712))


def connect(recorder: EEGRecorder, config: dict) -> None:
    if config["device"]["source"] in {"neuracle_acp", "device_adapter"}:
        raise ValueError("ACP/设备适配器配置请通过脑电采集工作台连接；旧入口仅支持8712浮点流。")
    if config["device"]["source"] == "synthetic":
        recorder.connect_synthetic(seed=42)
    else:
        recorder.connect()


def ensure_healthy(recorder: EEGRecorder, epoch: int) -> None:
    recorder._check_acquisition_error()
    recorder._check_recording_error()
    if recorder.waiting_for_device or recorder.epoch != epoch:
        raise RuntimeError("设备发生断开或重连；本次采集停止，请另开会话，避免标签跨断段。")


def collect(config: dict, output_root: str | Path, trigger_port: str | None = None,
            on_cue=print, *, simulated_trigger=None, on_phase=None,
            on_progress=None, stop_event=None) -> Path:
    config = copy.deepcopy(config)
    if trigger_port is not None:
        config["device"]["trigger_port"] = trigger_port
        config["device"].pop("trigger_connection", None)
    trigger_settings = configure_triggerbox(config)
    trigger_port = config["device"]["trigger_port"] or None
    network_trigger = trigger_settings["mode"] == "network"
    has_trigger = bool(trigger_port) or network_trigger
    if simulated_trigger is not None and config["device"]["source"] != "synthetic_tcp":
        raise ValueError("模拟触发回调只能用于明确标注的TCP人工模拟源。")
    trigger_info = {"enabled": has_trigger, "port": trigger_port or trigger_settings.get("port", ""),
                    "transport": "tcp" if network_trigger else "serial" if trigger_port else "none",
                    "host": trigger_settings.get("host", ""),
                    "endpoint": f"{trigger_settings['host']}:{trigger_settings['port']}" if network_trigger else trigger_port or "",
                    "baudrate": 115200 if trigger_port else None,
                    "protocol": "neuracle_dcp_tcp" if network_trigger else "reference_neuracle_serial" if trigger_port else None,
                    "device_name": "", "verified_events": 0,
                    "eeg_verified": False, "status": "connecting" if has_trigger else "disabled"}
    config.setdefault("session_metadata", {})["triggerbox"] = trigger_info
    name = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    participant = config.get("session_metadata", {}).get("participant_id")
    if participant is not None:
        participant = validate_participant_id(str(participant))
        name += "_" + participant
        config = {**config, "session_metadata": {**config["session_metadata"], "participant_id": participant}}
    path = new_session(output_root, name)
    recorder = make_recorder(config)
    events, box, complete, error = [], None, False, ""
    receiver_lease = None
    journal = (path / "events.jsonl").open("x", encoding="utf-8")
    try:
        if config["device"]["source"] == "neuracle":
            receiver_lease = ReceiverLease(config["device"].get("host", "127.0.0.1"), config["device"].get("port", 8712))
            receiver_lease.acquire()
        last_data = [-1, time.monotonic()]

        def check():
            if stop_event is not None and stop_event.is_set():
                raise CollectionCancelled("采集已由操作员停止；本次记录保留为未完成。")

        def progress():
            check()
            latest = recorder.raw_buffer.latest_source_index
            if latest is not None:
                if int(latest) != last_data[0]:
                    last_data[:] = [int(latest), time.monotonic()]
                elif time.monotonic() - last_data[1] > 5:
                    raise TimeoutError("5秒未收到新信号，采集停止；本次记录保留为未完成。")
            if on_progress is not None:
                on_progress({"session": str(path.resolve()),
                             "samples_received": 0 if latest is None else int(latest) + 1,
                             "triggerbox": dict(trigger_info),
                             "sample_rate": config["device"]["sample_rate"]})

        progress()
        if network_trigger:
            box = open_network_triggerbox(trigger_settings)
            trigger_info.update(device_name=box.device_name, status="network_connected")
        elif trigger_port:
            from .vendor.trigger_box import TriggerBoxConnection
            box = TriggerBoxConnection()
            box.connect(trigger_port)
            trigger_info.update(device_name=box.device_name, status="serial_connected")
        recorder.start_recording(str(path / "data.h5"), write_chunk_seconds=1.0)
        connect(recorder, config)
        epoch = recorder.epoch
        deadline = time.monotonic() + 5
        while recorder.raw_buffer.latest_source_index is None:
            check()
            recorder._check_acquisition_error()
            if time.monotonic() > deadline:
                raise TimeoutError("连接后5秒仍未收到脑电。")
            recorder.raw_buffer.wait_for_new_data(timeout=0.05)

        def wait(seconds):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                ensure_healthy(recorder, epoch)
                progress()
                time.sleep(min(0.03, max(0, end - time.monotonic())))

        def mark(kind, trial_id, label, code, phase_meta=None):
            check()
            before = recorder.raw_buffer.latest_source_index
            if before is None:
                raise RuntimeError("当前没有可对齐的样本。")
            source_index = int(before) + 1
            marking = "software_receive_clock"
            timing = {}
            if box is not None or simulated_trigger is not None:
                send_event = box.send_event if box is not None else simulated_trigger
                sent_ns = time.time_ns()
                send_event(int(code))
                ack_ns = time.time_ns()
                end = time.monotonic() + 1
                while True:
                    check()
                    ensure_healthy(recorder, epoch)
                    raw, indices, lost = recorder.raw_buffer.read_after_source_index(int(before))
                    if lost:
                        raise RuntimeError("等待触发期间缓存被覆盖。")
                    found = np.flatnonzero(decode_trigger_values(raw[-1]) == code)
                    if len(found):
                        source_index = int(indices[found[0]])
                        break
                    if time.monotonic() > end:
                        raise TimeoutError(f"事件码{code}已发送，但1秒内未在EEG触发通道观察到。请检查TriggerBox到脑电系统的触发连接及厂家触发输出设置；不自动退回软件标记。")
                    recorder.raw_buffer.wait_for_new_data(timeout=0.02)
                marking = "hardware_trigger" if box is not None else "synthetic_tcp_trigger"
                if box is not None:
                    trigger_info.update(verified_events=trigger_info["verified_events"] + 1,
                                        eeg_verified=True, status="receiving_triggers")
                    timing = {"trigger_port": trigger_port or "", "trigger_transport": trigger_info["transport"],
                              "trigger_endpoint": trigger_info["endpoint"], "trigger_device_name": box.device_name,
                              "trigger_send_wall_ns": sent_ns, "trigger_ack_wall_ns": ack_ns,
                              "trigger_observed_wall_ns": time.time_ns()}
            event = {"kind": kind, "trial_id": trial_id, "label": label,
                     "sample_index": source_index, "event_id": code,
                     "marking": marking, "wall_time_ns": time.time_ns()}
            event.update(phase_meta or {})
            event.update(timing)
            events.append(event)
            journal.write(json.dumps(event, ensure_ascii=False) + "\n")
            journal.flush()

        collection = config["collection"]
        if collection["task_seconds"] < config["window"]["seconds"] + config["window"]["task_offset_seconds"]:
            raise ValueError("任务时长不足以产生完整训练窗口。")
        tasks_per_round = len(config["classes"])
        total = collection["loops"] * tasks_per_round

        def phase(name, seconds, trial_id=0, label=None, loop=0):
            check()
            if on_phase is None:
                return {}
            round_complete = (name in {"rest", "block_rest", "finishing"}
                              and trial_id > 0 and trial_id % tasks_per_round == 0)
            completed_rounds = loop if round_complete else max(0, loop - 1)
            round_progress = {"round_complete": round_complete,
                              "completed_rounds": completed_rounds,
                              "remaining_rounds": collection["loops"] - completed_rounds}
            info = {"phase": name, "duration_seconds": seconds,
                    "trial_id": trial_id, "total_trials": total, "loop": loop,
                    "loops": collection["loops"], "label": label,
                    **round_progress,
                    "task_text": "" if label is None else config["classes"][label]["task_text"],
                    "prompt_text": "" if label is None else
                    config.get("session_metadata", {}).get("prompt_texts",
                        [c["task_text"] for c in config["classes"]])[label]}
            meta = on_phase(info) or {}
            display_text = (info["prompt_text"] if name in {"prepare", "task"} else
                            "结束" if name == "finishing" else "休息")
            return {**meta, **round_progress, "phase": name, "display_text": display_text}

        on_cue("开始前静息；请保持身体放松。")
        meta = phase("initial_rest", collection["initial_rest_seconds"])
        if on_phase is not None:
            mark("initial_rest", 0, None, 10, meta)
        wait(collection["initial_rest_seconds"])
        rng = np.random.default_rng(collection.get("seed", 42))
        trial_id = 0
        for loop in range(collection["loops"]):
            for label in rng.permutation(len(config["classes"])):
                label = int(label)
                trial_id += 1
                prepare_seconds = collection.get("prepare_seconds", 0)
                # Sound-enabled UI always announces the task, even with zero extra preparation.
                if prepare_seconds or config.get("session_metadata", {}).get("audio", {}).get("enabled", False):
                    on_cue("准备下一项任务；请暂时保持放松。")
                    meta = phase("prepare", prepare_seconds, trial_id, label, loop + 1)
                    mark("prepare", trial_id, label, 12, meta)
                    wait(meta.get("prepare_remaining_seconds", prepare_seconds))
                on_cue(f"轮次{loop + 1} 试次{trial_id}：{config['classes'][label]['task_text']}")
                meta = phase("task", collection["task_seconds"], trial_id, label, loop + 1)
                mark("cue", trial_id, label, label + 1, meta)
                wait(collection["task_seconds"])
                on_cue("静息")
                meta = phase("rest", collection["rest_seconds"], trial_id, label, loop + 1)
                mark("task_end", trial_id, label, 9, meta)
                wait(collection["rest_seconds"])
            block_rest = collection.get("block_rest_seconds", 0)
            if loop + 1 < collection["loops"] and block_rest:
                meta = phase("block_rest", block_rest, trial_id, None, loop + 1)
                mark("block_rest", trial_id, None, 13, meta)
                wait(block_rest)
        tail = collection.get("tail_seconds", 1.0)
        meta = phase("finishing", tail, trial_id, loop=collection["loops"])
        if on_phase is not None:
            mark("session_end", trial_id, None, 14, meta)
        wait(tail)
        ensure_healthy(recorder, epoch)
        progress()
        complete = True
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        journal.close()
        try:
            recorder.disconnect()
        except BaseException as exc:
            complete = False
            error = error or str(exc)
            raise
        finally:
            if receiver_lease is not None:
                receiver_lease.close()
            try:
                if box is not None:
                    box.disconnect()
            except Exception as exc:
                complete = False
                error = error or str(exc)
                raise
            finally:
                if has_trigger:
                    trigger_info["status"] = "completed" if complete else "incomplete"
                if (path / "data.h5").is_file():
                    seal_session(path, config, events, complete=complete,
                                 source=config["device"]["source"], error=error)
                    from .collection_export import export_collection
                    export_collection(path, config, events, complete=complete, error=error)
    return path
