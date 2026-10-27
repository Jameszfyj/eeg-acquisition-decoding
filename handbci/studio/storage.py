"""One asynchronous disk writer. Raw counts and timing gaps are never repaired."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import queue
import struct
import threading
import time
import uuid

import h5py
import numpy as np

from ..config import validate_participant_id


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def paradigm_id(tasks):
    content = [{"id": i, "text": x["text"].strip(), "action": x["action"]} for i, x in enumerate(tasks)]
    return hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


class SessionWriter:
    def __init__(self, root, participant, description, protocol, config, validation=None):
        participant = validate_participant_id(participant)
        if not 1 <= int(description['settings']['fs']) <= 100000 or not description['labels']:
            raise ValueError('非法采样率或通道数。')
        if len(description['labels']) != description['settings']['channels']:
            raise ValueError('通道标签数与设备通道数不符。')
        if not np.isfinite(description['count_scale']) or description['count_scale'] <= 0 or not description['unit']:
            raise ValueError('适配器须提供有效缩放及单位。')
        name = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8] + "_" + participant
        self.path = Path(root) / name
        self.path.mkdir(parents=True, exist_ok=False)
        self.description = copy.deepcopy(description)
        self.protocol, self.config = copy.deepcopy(protocol), copy.deepcopy(config)
        self.recording_context = {'condition':self.protocol.get('condition','未记录'),
                                  'notes':self.protocol.get('condition_notes','')}
        self.config.setdefault('session_metadata',{})['recording_context'] = copy.deepcopy(self.recording_context)
        self.participant = participant
        self.validation = validation
        if validation:
            self.config['session_metadata']['offline_validation'] = copy.deepcopy(validation.contract)
            validation.start(self.path)
        self.count = 0
        self.error = ""
        self.summary = {}
        self.events = []
        self.queue = queue.Queue(maxsize=16384)
        self.ready = threading.Event()
        self.finished = threading.Event()
        self.closing = threading.Event()
        self.submit_lock = threading.Lock()
        self.requested_complete = False
        self.requested_error = ""
        json_write(self.path / 'session.json', {'complete': False, 'participant_id': participant,
                   'recording_context': self.recording_context,
                   'device': self.description, 'protocol': self.protocol, 'config': self.config})
        self.thread = threading.Thread(target=self._run, name="studio-disk-writer", daemon=True)
        self.thread.start()
        if not self.ready.wait(5) or self.error:
            raise RuntimeError("无法开始记录：" + (self.error or "写盘线程未响应"))

    def submit(self, kind, value):
        with self.submit_lock:
            if self.error or self.finished.is_set() or self.closing.is_set():
                raise RuntimeError("写盘已停止：" + self.error)
            try:
                self.queue.put_nowait((kind, value))
            except queue.Full as exc:
                self.error = "写盘队列已满，停止会话；不静默丢弃采样点。"
                raise RuntimeError(self.error) from exc

    def close(self, complete, error=""):
        with self.submit_lock:
            self.requested_complete, self.requested_error = bool(complete and not self.error), error or self.error
            self.closing.set()
        if not self.finished.is_set():
            self.thread.join(20)
        if self.thread.is_alive():
            self.requested_complete = False
            self.requested_error = '写盘未能在20秒内封存。'
            raise RuntimeError("写盘线程尚未结束，文件保留为未完成。")
        if self.error:
            raise RuntimeError(self.error)
        return self.summary

    def _run(self):
        fs, ch = self.description["settings"]["fs"], len(self.description["labels"])
        started_wall = time.time_ns()
        gaps, previous_end, all_events, faults = [], None, [], []
        period = 1000*self.description['settings']['points']/fs
        tolerance = .501 if period % 1 else .001
        try:
            with h5py.File(self.path / "data.h5", "x") as h, \
                    (self.path / "frames.nrb").open("xb", buffering=1024*1024) as raw, \
                    (self.path / "events.jsonl").open("x", encoding="utf-8") as journal:
                h.attrs.update(srate=fs, n_eeg=ch, complete=False, n_samples=0,
                               recording_context_json=json.dumps(self.recording_context,ensure_ascii=False),
                               opened_wall_ns=started_wall, unit=self.description['unit'],
                               count_scale=self.description['count_scale'], studio_version=1)
                ds = {"eeg": h.create_dataset("eeg", (ch, 0), maxshape=(ch, None), dtype="<f4", chunks=(ch, fs)),
                      "counts": h.create_dataset("counts", (ch, 0), maxshape=(ch, None), dtype="<i4", chunks=(ch, fs))}
                for name, dtype in (("source_indices", "<i8"), ("device_time_ms", "<f8"),
                                    ('raw_device_time_ms', '<f8'), ('shared_hardware_time_ms', '<f8'), ('packet_id', 'u1'),
                                    ("arrival_wall_ns", "<i8"), ("arrival_mono_ns", "<i8"), ("trigger", "<u4")):
                    ds[name] = h.create_dataset(name, (0,), maxshape=(None,), dtype=dtype, chunks=(fs,))
                header = json.dumps({"session": self.path.name, "device": self.description,
                                     "started_wall_ns": started_wall,
                                     "raw_format": "parsed_adapter_payload_records_not_full_transport_capture"}, ensure_ascii=False).encode()
                raw.write(struct.pack("<4sII", b"NRB1", 1, len(header))+header)
                self.ready.set()
                batch = []
                pending = 0

                def flush():
                    nonlocal pending, previous_end
                    if not batch:
                        return
                    values = np.concatenate([x.counts for x in batch], axis=1)
                    stamps = np.concatenate([x.device_ms+np.arange(x.counts.shape[1])*1000/fs for x in batch])
                    if previous_end is not None:
                        diffs = np.diff(np.r_[previous_end, stamps])
                    else:
                        diffs = np.diff(stamps)
                    jumps = np.flatnonzero((diffs <= 0) | (np.abs(diffs-1000/fs) > tolerance))
                    if len(jumps):
                        gaps.append({"near_sample": self.count, "jumps": int(len(jumps)),
                                     "largest_step_ms": float(np.max(np.abs(diffs)))})
                    previous_end = float(stamps[-1])
                    n, begin = values.shape[1], self.count
                    for name in ("eeg", "counts"):
                        ds[name].resize(begin+n, axis=1)
                    eeg = values.astype(np.float32)*self.description['count_scale']
                    ds["eeg"][:, begin:] = eeg
                    ds["counts"][:, begin:] = values
                    wall = np.concatenate([np.full(x.counts.shape[1], x.wall_ns, np.int64) for x in batch])
                    mono = np.concatenate([np.full(x.counts.shape[1], x.mono_ns, np.int64) for x in batch])
                    for name, values1 in (("source_indices", np.arange(begin, begin+n)),
                                           ('packet_id', np.concatenate([np.full(x.counts.shape[1], x.packet_id) for x in batch])),
                                           ('raw_device_time_ms', np.concatenate([(x.raw_device_ms if x.raw_device_ms is not None else x.device_ms)+np.arange(x.counts.shape[1])*1000/fs for x in batch])),
                                           ('shared_hardware_time_ms', np.concatenate([((x.detail or {}).get('sync_time_ms') if (x.detail or {}).get('sync_time_ms') is not None else np.nan)+np.arange(x.counts.shape[1])*1000/fs for x in batch])),
                                           ("device_time_ms", stamps), ("arrival_wall_ns", wall),
                                           ("arrival_mono_ns", mono), ("trigger", np.zeros(n, np.uint32))):
                        ds[name].resize((begin+n,))
                        ds[name][begin:] = values1
                    self.count += n
                    h.attrs["n_samples"] = self.count
                    if self.validation:
                        self.validation.submit('data', (eeg, stamps, begin))
                    batch.clear()
                    pending = 0

                while not (self.closing.is_set() and self.queue.empty()):
                    try:
                        kind, value = self.queue.get(timeout=.1)
                    except queue.Empty:
                        flush()
                        continue
                    if kind in ('event', 'fault'):
                        if self.validation and kind == 'event' and value.get('kind') in ('prepare','cue','task_end'):
                            flush()
                            self.validation.submit('event', copy.deepcopy(value))
                        (all_events if kind == 'event' else faults).append(dict(value))
                        # Crash journal is written immediately; final alignment is rewritten on close.
                        journal.write(json.dumps(value, ensure_ascii=False)+"\n")
                        journal.flush()
                        continue
                    packet = value
                    payload = packet.payload
                    if len(payload) <= 65535:
                        raw.write(struct.pack("<QQIHBBBB", packet.wall_ns, packet.mono_ns,
                            int(packet.raw_device_ms if packet.raw_device_ms is not None else packet.device_ms) & 0xFFFFFFFF, len(payload), packet.packet_id & 255,
                            packet.module & 255, 1, 0)+payload)
                    if packet.kind == "eeg":
                        if packet.counts.ndim != 2 or packet.counts.shape[0] != ch or not packet.counts.shape[1]:
                            raise ValueError('数据包维度与配置不一致。')
                        batch.append(packet)
                        pending += packet.counts.shape[1]
                        if pending >= fs:
                            flush()
                flush()
                stamps = ds["device_time_ms"][:]
                monotonic = len(stamps) > 0 and np.all(np.diff(stamps) > 0)
                for event in all_events:
                    target = event.get("device_time_ms")
                    if target is not None and monotonic and stamps[0] <= target <= stamps[-1]+1000/fs:
                        i = int(np.searchsorted(stamps, target))
                        if 0 < i < len(stamps) and target < stamps[i] and stamps[i]-stamps[i-1] > 1000/fs+tolerance:
                            event.update(sample_index=None, alignment='device_clock_gap')
                            continue
                        event["sample_index"] = max(0, min(i, len(stamps)))
                        event["alignment"] = 'device_clock_lower_bound' if event.get('echo_verified') or event['kind'] == 'device_event' else 'estimated_host_receive_clock'
                        if event["kind"] == "device_event" and i < len(stamps) and event.get("event_id") is not None:
                            code, existing = int(event['event_id']), int(ds['trigger'][i])
                            if existing:
                                event['trigger_dataset_collision'] = True
                            if not existing or (code in (1,2,3,4,5,6,9) and existing not in (1,2,3,4,5,6,9)):
                                ds['trigger'][i] = code
                    else:
                        event["sample_index"] = None
                        event["alignment"] = "unavailable"
                cue_events = [e for e in all_events if e["kind"] in ("cue", "task_end")]
                reasons = []
                expected = self.protocol['loops']*len(self.protocol['tasks'])*2
                if not self.requested_complete or self.error or not self.count: reasons.append('incomplete_or_empty')
                if gaps or faults: reasons.append('clock_or_receive_faults')
                if len(cue_events) != expected or any(e['sample_index'] is None or not e.get('echo_verified') for e in cue_events): reasons.append('unverified_or_missing_labels')
                active, seen, invalid = None, set(), False
                for event in cue_events:
                    if event['kind'] == 'cue':
                        if active is not None or event.get('trial_id') in seen or not 0 <= event.get('label',-1) < len(self.protocol['tasks']): invalid = True
                        active = event
                        seen.add(event.get('trial_id'))
                    else:
                        if active is None or event.get('trial_id') != active.get('trial_id') or event.get('label') != active.get('label') or event.get('sample_index') is None or active.get('sample_index') is None or event['sample_index'] <= active['sample_index']:
                            invalid = True
                        active = None
                if invalid or active is not None: reasons.append('invalid_trial_boundaries')
                if not self.description.get('timestamp_verified'): reasons.append('device_event_time_semantics_unverified')
                if self.description['unit'] not in ('uV', 'synthetic_uV'): reasons.append('unit_not_supported_by_training')
                if self.protocol.get('task_seconds', 0) < 2.5: reasons.append('legacy_training_window_too_short')
                trainable = not reasons
                unverified_trials = sorted({e['trial_id'] for e in cue_events if e.get('trial_id') is not None
                                            and (not e.get('echo_verified') or e['sample_index'] is None)})
                marking_warnings = sum(f.get('severity') == 'warning' for f in faults)
                quality = {"training_compatible": trainable, "device_time_gaps": gaps,
                           'reasons': reasons, 'receive_faults': faults,
                           'marking_warning_count': marking_warnings,
                           'unverified_trial_ids': unverified_trials,
                           "time_convention": "packet_device_timestamp_as_first_sample; arrival_time_stored_separately",
                           "physical_onset_measured": False}
                meta = {"format": "handbci.raw.v1", "session_id": self.path.name,
                        "complete": bool(self.requested_complete and self.count and not self.error), "source": self.config["device"]["source"],
                        "config": self.config, "device": self.description, "quality": quality,
                        "protocol": self.protocol, "participant_id": self.participant,
                        'recording_context': self.recording_context,
                        "paradigm_id": paradigm_id(self.protocol["tasks"]),
                        "error": self.requested_error}
                h.attrs.update(handbci_format="handbci.raw.v1", metadata_json=json.dumps(meta, ensure_ascii=False),
                               complete=meta['complete'], closed_wall_ns=time.time_ns())
                h.create_dataset("events_json", data=[json.dumps(e, ensure_ascii=False) for e in all_events],
                                 dtype=h5py.string_dtype("utf-8"))
                self.events = all_events
                self.summary = {"path": str(self.path.resolve()), "samples": self.count,
                                "signal_seconds": self.count/fs, "complete": meta['complete'], 'reasons': reasons,
                                "training_compatible": trainable, "device_time_gaps": gaps,
                                'marking_warning_count': marking_warnings,
                                'unverified_trial_ids': unverified_trials,
                                "error": self.requested_error, "participant": self.participant,
                                'recording_context': self.recording_context,
                                "paradigm_id": meta["paradigm_id"]}
                if self.validation:
                    result = self.validation.finish()
                    meta['onnx_validation'] = self.summary['onnx_validation'] = dict(
                        path='onnx_validation/summary.json', status=result['status'],
                        completed_trials=result['completed_trials'], scored_trials=result['scored_trials'],
                        correct_trials=result['correct_trials'], error=result['error'])
                    h.attrs['metadata_json'] = json.dumps(meta, ensure_ascii=False)
                json_write(self.path / "session.json", meta)
                json_write(self.path / "collection_report.json", self.summary)
            with (self.path / "events.jsonl").open("w", encoding="utf-8") as journal:
                for event in self.events+faults:
                    journal.write(json.dumps(event, ensure_ascii=False)+"\n")
        except BaseException as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            json_write(self.path / "failure.json", {"error": self.error, "complete": False})
        finally:
            if self.validation and not self.validation.closed.is_set():
                self.validation.finish()
            self.ready.set()
            self.finished.set()
