"""采集结果的CSV和摘要；不把预测或休息段自动当成训练类别。"""
from __future__ import annotations

import csv
from pathlib import Path

import h5py

from .config import ACTION_NAMES, write_json


def export_collection(path, config, events, *, complete, error=""):
    path = Path(path)
    classes = {c["id"]: c for c in config["classes"]}
    fs = float(config["device"]["sample_rate"])
    fields = ["kind", "phase", "trial_id", "label", "task_text", "display_text", "action", "event_id",
              "sample_index", "source_seconds", "marking", "wall_time_ns",
              "trigger_port", "trigger_transport", "trigger_endpoint", "trigger_device_name", "trigger_send_wall_ns", "trigger_ack_wall_ns",
              "trigger_observed_wall_ns",
              "display_requested_wall_ns", "display_ack_wall_ns", "browser_render_time_ms",
              "audio_enabled", "audio_text", "audio_kind", "audio_scheduled_browser_ms",
              "audio_completed_browser_ms"]
    with (path / "events.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in events:
            c = classes.get(event.get("label"), {})
            row = {key: event.get(key, "") for key in fields}
            audio = event.get("audio_cue", {})
            row.update(audio_enabled=audio.get("enabled", False),
                       audio_text=audio.get("text", ""), audio_kind=audio.get("kind", ""),
                       audio_scheduled_browser_ms=audio.get("scheduled_browser_ms", ""),
                       audio_completed_browser_ms=audio.get("completed_browser_ms", ""))
            row.update(task_text=c.get("task_text", ""), action=c.get("action", ""),
                       source_seconds=event["sample_index"] / fs)
            writer.writerow(row)
    counts = {c["name"]: 0 for c in classes.values()}
    starts = {}
    trials = []
    for event in events:
        if event["kind"] == "cue":
            starts[event["trial_id"]] = event
        elif event["kind"] == "task_end" and event["trial_id"] in starts:
            begin = starts.pop(event["trial_id"])
            c = classes[begin["label"]]
            counts[c["name"]] += 1
            trials.append({"trial_id": event["trial_id"], "label": begin["label"],
                           "task_text": c["task_text"], "action": c["action"],
                           "start_sample": begin["sample_index"], "end_sample": event["sample_index"],
                           "start_seconds": begin["sample_index"] / fs,
                           "end_seconds": event["sample_index"] / fs})
    with (path / "trials.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trial_id", "label", "task_text", "action",
                                                  "start_sample", "end_sample", "start_seconds", "end_seconds"])
        writer.writeheader()
        writer.writerows(trials)
    with h5py.File(path / "data.h5", "r") as handle:
        samples = handle["eeg"].shape[1]
        writer_complete = bool(handle.attrs.get("complete", False))
    report = {"session": str(path.resolve()), "complete": bool(complete and writer_complete),
              "source": config["device"]["source"], "sample_rate": fs,
              "eeg_channels": len(config["device"]["channels"]), "samples_received": samples,
              "signal_duration_seconds": samples / fs, "n_trials": len(trials),
              "class_counts": counts, "n_events": len(events),
              "marking_types": sorted({e["marking"] for e in events}), "error": error,
              "triggerbox": config.get("session_metadata", {}).get("triggerbox", {"enabled": False}),
              "tasks": [{"label": c["id"], "task_text": c["task_text"], "name": c["name"],
                         "action": c["action"], "action_text": ACTION_NAMES[c["action"]],
                         "completed_trials": counts[c["name"]]} for c in classes.values()],
              "audio": {"enabled": config.get("session_metadata", {}).get("audio", {}).get("enabled", False),
                        "volume": config.get("session_metadata", {}).get("audio", {}).get("volume", 70),
                        "task_beeps": sum(e.get("audio_cue", {}).get("kind") == "beep" for e in events),
                        "prepared_task_speeches": sum(e.get("phase") == "prepare"
                                                     and "completed_browser_ms" in e.get("audio_cue", {}) for e in events),
                        "timing_note": "声音时间为浏览器播放安排/回调的记录，未测量扬声器实际出声时刻。"},
              "display_timing_note": "浏览器显示确认及主机时间为软件测量，不能替代屏幕物理呈现时刻的测量。"}
    write_json(path / "collection_report.json", report)
    return report
