"""验证界面API控制的是实际采集流程：显示确认、阶段标记、停止和故障。"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import socket
import threading
import time
import unittest
import urllib.error
import urllib.request
import uuid

from handbci.collection_ui import CollectionController, create_ui_server
from handbci.config import load_config
from handbci.data import build_dataset, load_session

ROOT = Path(__file__).resolve().parents[1]


class CollectionUITests(unittest.TestCase):
    def setUp(self):
        self.work = ROOT / "runtime/ui_tests" / uuid.uuid4().hex[:10]
        self.controller = CollectionController(self.work)
        self.server = create_ui_server(self.controller)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        with urllib.request.urlopen(self.url + "/") as response:
            html = response.read().decode("utf-8")
        self.token = re.search(r'name="collection-token" content="([^"]+)"', html).group(1)
        self.cfg = load_config(ROOT / "configs/collection_ui_6class.json")
        self.cfg["device"]["port"] = 0
        self.cfg["window"].update(seconds=0.5, stride_seconds=0.5, task_offset_seconds=0.1)
        self.cfg["collection"].update(initial_rest_seconds=1.2, prepare_seconds=0.05,
                                       task_seconds=0.8, rest_seconds=0.05, tail_seconds=0.7)

    def tearDown(self):
        self.controller.stop()
        if self.controller.worker:
            self.controller.worker.join(timeout=6)
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, path, payload=None, *, authenticated=True):
        headers = {}
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
            if authenticated:
                headers["X-Collection-Token"] = self.token
        request = urllib.request.Request(self.url + path, data=data, headers=headers)
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read())

    def wait(self, *, ack=True, until=None, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.request("/api/state")
            p = state.get("phase")
            if ack and state["running"] and p and p["awaiting_ack"]:
                self.request("/api/ack", {"phase_id": p["id"], "render_time_ms": time.time() * 1000})
            if until and until(state):
                return state
            if not state["running"]:
                return state
            time.sleep(0.015)
        self.fail("采集界面未在预期时间内结束。")

    def test_full_ui_schedule_exports_display_times_and_all_six_classes(self):
        sessions = []
        for participant in ("TEST_UI", "被试02"):
            self.request("/api/start", {"config": self.cfg, "participant": participant})
            state = self.wait()
            self.assertEqual(state["status"], "completed", state["error"])
            session = Path(state["session"])
            sessions.append(session)
            self.assertTrue(session.name.endswith("_" + participant))
            self.assertEqual(session.parent.resolve(), self.work.resolve())
            self.assertEqual(state["report"]["session"], str(session.resolve()))
            record = load_session(session)
            self.assertEqual(record["config"]["session_metadata"]["participant_id"], participant)
            self.assertEqual(state["report"]["n_trials"], 6)
            self.assertEqual(state["report"]["n_events"], 20)
            self.assertTrue(state["report"]["complete"])
            self.assertEqual({e["phase"] for e in record["events"]},
                             {"initial_rest", "prepare", "task", "rest", "finishing"})
            for e in record["events"]:
                self.assertLessEqual(e["display_requested_wall_ns"], e["display_ack_wall_ns"])
                self.assertLessEqual(e["display_ack_wall_ns"], e["wall_time_ns"])
                self.assertEqual(int(record["trigger"][e["sample_index"]]), e["event_id"])
            ds = build_dataset([session])
            self.assertEqual(set(ds.y.tolist()), set(range(6)))
            for name in ("events.csv", "trials.csv", "collection_report.json"):
                self.assertTrue((session / name).is_file())
        self.assertNotEqual(sessions[0], sessions[1])
        first = load_session(sessions[0])
        self.assertEqual(first["config"]["session_metadata"]["participant_id"], "TEST_UI")

    def test_round_feedback_uses_configured_class_count_with_or_without_block_rest(self):
        for task_count, block_rest in ((2, 0), (6, 0.1)):
            with self.subTest(task_count=task_count, block_rest=block_rest):
                cfg = copy.deepcopy(self.cfg)
                cfg["classes"] = cfg["classes"][:task_count]
                cfg["collection"].update(loops=2, initial_rest_seconds=0.1,
                                         block_rest_seconds=block_rest, tail_seconds=0.1)
                self.request("/api/start", {"config": cfg, "participant": "ROUND_TEST"})
                seen = []
                state = self.wait(until=lambda s: self._observe_phase(s, seen), timeout=25)
                self.assertEqual(state["status"], "completed", state["error"])
                for p in seen:
                    boundary = (p["phase"] in {"rest", "block_rest", "finishing"}
                                and p["trial_id"] > 0 and p["trial_id"] % task_count == 0)
                    self.assertEqual(p["round_complete"], boundary, p)
                    self.assertEqual(p["remaining_rounds"], 2 - p["completed_rounds"])
                rests = [p for p in seen if p["phase"] == "rest" and p["round_complete"]]
                self.assertEqual([p["remaining_rounds"] for p in rests], [1, 0])
                self.assertEqual(len([p for p in seen if p["phase"] == "block_rest"]),
                                 1 if block_rest else 0)
                events = load_session(state["session"])["events"]
                endings = [e for e in events if e["kind"] == "task_end" and e["round_complete"]]
                self.assertEqual([e["remaining_rounds"] for e in endings], [1, 0])
                self.assertEqual(events[-1]["remaining_rounds"], 0)
                self.assertEqual(events[-1]["completed_rounds"], 2)

    def test_custom_subset_saves_only_selected_classes_with_contiguous_labels(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["classes"] = [copy.deepcopy(cfg["classes"][i]) for i in (0, 3, 5)]
        for i, task in enumerate(cfg["classes"]):
            task["id"] = i
        prompts = ["左手", "双脚", "右手与左脚"]
        texts = ["自定义说明：" + prompt for prompt in prompts]
        self.request("/api/start", {"config": cfg, "participant": "CUSTOM_CLASSES",
                                    "task_texts": texts, "prompt_texts": prompts})
        state = self.wait()
        self.assertEqual(state["status"], "completed", state["error"])
        self.assertEqual(state["total_trials"], 3)
        self.assertEqual(state["report"]["n_trials"], 3)
        self.assertEqual(set(state["report"]["class_counts"]), {c["name"] for c in cfg["classes"]})
        self.assertTrue(all(n == 1 for n in state["report"]["class_counts"].values()))
        record = load_session(state["session"])
        self.assertEqual(len(record["config"]["classes"]), 3)
        self.assertEqual([c["task_text"] for c in record["config"]["classes"]], texts)
        self.assertEqual(record["config"]["session_metadata"]["prompt_texts"], prompts)
        for event in (e for e in record["events"] if e["kind"] == "cue"):
            self.assertEqual(event["event_id"], event["label"] + 1)
            self.assertEqual(event["display_text"], prompts[event["label"]])
            self.assertEqual(int(record["trigger"][event["sample_index"]]), event["event_id"])
        self.assertEqual(set(build_dataset([state["session"]]).y.tolist()), {0, 1, 2})

    def test_invalid_class_count_and_repeated_prompt_names_are_rejected_before_recording(self):
        for count in (1, 7):
            cfg = copy.deepcopy(self.cfg)
            cfg["classes"] = [{**self.cfg["classes"][i % 6], "id": i, "name": f"class_{i}"}
                              for i in range(count)]
            with self.subTest(count=count), self.assertRaises(urllib.error.HTTPError) as error:
                self.request("/api/start", {"config": cfg})
            self.assertEqual(error.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request("/api/start", {"config": self.cfg, "prompt_texts": ["左手", "右手"] * 3})
        self.assertEqual(error.exception.code, 400)
        self.assertFalse(self.controller.snapshot()["running"])
        self.assertFalse(self.work.exists())

    @staticmethod
    def _observe_phase(state, seen):
        phase = state.get("phase")
        if phase and (not seen or seen[-1]["id"] != phase["id"]):
            seen.append(phase)
        return False

    def test_stop_during_imagery_keeps_an_incomplete_record(self):
        self.request("/api/start", {"config": self.cfg})
        state = self.wait(until=lambda s: (s.get("phase") or {}).get("phase") == "task")
        self.assertTrue(state["running"])
        self.request("/api/stop", {})
        stopped = self.wait()
        self.assertEqual(stopped["status"], "cancelled")
        self.assertFalse(stopped["report"]["complete"])
        with self.assertRaises(ValueError):
            load_session(stopped["session"])

    def test_audio_schedule_preserves_custom_labels_and_zero_prepare(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["collection"]["prepare_seconds"] = 0
        texts = ["左手", "右手", "双手", "双脚", "左手右脚", "右手左脚"]
        self.request("/api/start", {"config": cfg, "participant": "AUDIO_TEST",
                                    "prompt_texts": texts, "audio": {"enabled": True, "volume": 60}})
        deadline = time.monotonic() + 18
        while time.monotonic() < deadline:
            state = self.request("/api/state")
            p = state.get("phase")
            if state["running"] and p and p["awaiting_ack"]:
                word = (texts[p["label"]] if p["phase"] == "prepare" else
                        {"initial_rest": "开始", "task": "滴", "rest": "休息",
                         "block_rest": "休息", "finishing": "结束"}[p["phase"]])
                stamp = time.time() * 1000
                audio = {"enabled": True, "kind": "beep" if p["phase"] == "task" else "speech",
                         "text": word, "scheduled_browser_ms": stamp, "duration_seconds": 0.1}
                if p["phase"] in {"prepare", "finishing"}:
                    # Simulate finished playback before the phase acknowledgement.
                    time.sleep(0.08)
                    audio["completed_browser_ms"] = time.time() * 1000
                self.request("/api/ack", {"phase_id": p["id"], "render_time_ms": stamp, "audio": audio})
            if not state["running"]:
                break
            time.sleep(0.01)
        self.assertFalse(state["running"])
        self.assertEqual(state["status"], "completed", state["error"])
        record = load_session(state["session"])
        self.assertEqual(record["config"]["classes"], cfg["classes"])
        self.assertEqual(record["config"]["session_metadata"]["prompt_texts"], texts)
        self.assertTrue(record["config"]["session_metadata"]["audio"]["enabled"])
        events = record["events"]
        self.assertEqual(len([e for e in events if e["kind"] == "prepare"]), 6)
        for cue in (e for e in events if e["kind"] == "cue"):
            preparation = next(e for e in events if e["kind"] == "prepare" and e["trial_id"] == cue["trial_id"])
            end = next(e for e in events if e["kind"] == "task_end" and e["trial_id"] == cue["trial_id"])
            self.assertEqual(preparation["label"], cue["label"])
            self.assertEqual(preparation["audio_cue"]["text"], texts[cue["label"]])
            self.assertEqual(preparation["display_text"], texts[cue["label"]])
            self.assertEqual(cue["display_text"], texts[cue["label"]])
            self.assertEqual(end["display_text"], "休息")
            self.assertGreaterEqual(preparation["audio_cue"]["completed_browser_ms"],
                                    preparation["audio_cue"]["scheduled_browser_ms"])
            self.assertEqual(cue["audio_cue"]["kind"], "beep")
            self.assertEqual(end["audio_cue"]["text"], "休息")
            # The end event does not wait for the rest speech to finish.
            duration = (end["wall_time_ns"] - cue["wall_time_ns"]) / 1e9
            self.assertGreaterEqual(duration, 0.8)
            self.assertLess(duration, 1.2)
        self.assertEqual(events[-1]["audio_cue"]["text"], "结束")

    def test_preparation_duration_includes_speech_time_and_waits_for_long_speech(self):
        for i, elapsed in enumerate((0.7, 2.6)):
            self.controller.state.update(running=True, audio={"enabled": True},
                phase={"id": 900+i, "phase": "prepare", "duration_seconds": 2,
                       "prompt_text": "左手", "task_text": "完整想象任务说明", "awaiting_ack": True})
            self.controller.phase_ack = None
            audio={"enabled": True, "kind": "speech", "text": "左手",
                   "scheduled_browser_ms": 1000, "completed_browser_ms": 1000+elapsed*1000,
                   "duration_seconds": elapsed}
            self.request("/api/ack", {"phase_id": 900+i, "render_time_ms": 1000,
                                      "audio": audio, "prepare_elapsed_seconds": elapsed})
            remaining=self.controller.phase_ack["prepare_remaining_seconds"]
            self.assertAlmostEqual(remaining, 1.3 if elapsed==0.7 else 0)
            self.assertLessEqual(self.controller.state["phase"]["deadline_monotonic"]-time.monotonic(),
                                 remaining+0.02)

    def test_missing_or_wrong_audio_ack_is_rejected_and_audio_failure_keeps_partial_record(self):
        self.request("/api/start", {"config": self.cfg, "audio": {"enabled": True}})
        state = self.wait(ack=False, until=lambda s: bool(s.get("phase")))
        p = state["phase"]
        with self.assertRaises(urllib.error.HTTPError) as missing:
            self.request("/api/ack", {"phase_id": p["id"], "render_time_ms": time.time() * 1000})
        self.assertEqual(missing.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError):
            self.request("/api/ack", {"phase_id": p["id"], "render_time_ms": time.time() * 1000,
                                     "audio": {"enabled": True, "kind": "speech", "text": "错误任务"}})
        self.assertTrue(self.controller.snapshot()["phase"]["awaiting_ack"])
        self.request("/api/audio-error", {"phase_id": p["id"], "error": "测试播放失败"})
        stopped = self.wait(ack=False)
        self.assertEqual(stopped["status"], "failed")
        self.assertIn("测试播放失败", stopped["error"])
        self.assertFalse(stopped["report"]["complete"])
        self.assertEqual(stopped["report"]["n_trials"], 0)

    def test_no_display_ack_stops_before_any_imagery_task(self):
        self.request("/api/start", {"config": self.cfg})
        state = self.wait(ack=False, timeout=8)
        self.assertEqual(state["status"], "failed")
        self.assertIn("未确认显示", state["error"])
        self.assertEqual(state["report"]["n_trials"], 0)
        self.assertFalse(state["report"]["complete"])

    def test_unavailable_real_source_is_reported_without_completed_trials(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["device"].update(source="neuracle", unit="device_native")
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            cfg["device"]["port"] = reserved.getsockname()[1]
            self.request("/api/start", {"config": cfg})
            state = self.wait()
        self.assertEqual(state["status"], "failed")
        self.assertFalse(state["report"]["complete"])
        self.assertEqual(state["report"]["n_trials"], 0)

    def test_profiles_and_control_require_valid_configuration_and_page_token(self):
        profiles = self.request("/api/profiles")
        self.assertEqual(len(profiles), 3)
        self.assertTrue(all(len(p["config"]["classes"]) == 6 for p in profiles))
        with self.assertRaises(urllib.error.HTTPError) as missing_token:
            self.request("/api/start", {"config": self.cfg}, authenticated=False)
        self.assertEqual(missing_token.exception.code, 403)
        with self.assertRaises(urllib.error.HTTPError) as invalid_time:
            self.request("/api/start", {"config": self.cfg, "collection": {"task_seconds": 0.1}})
        self.assertEqual(invalid_time.exception.code, 400)
        self.assertFalse(self.controller.snapshot()["running"])
        with self.assertRaises(urllib.error.HTTPError) as invalid_participant:
            self.request("/api/start", {"config": self.cfg, "participant": "P01/child"})
        self.assertEqual(invalid_participant.exception.code, 400)
        self.assertFalse(self.controller.snapshot()["running"])


if __name__ == "__main__":
    unittest.main()
