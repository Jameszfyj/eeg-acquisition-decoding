"""本机浏览器采集界面：主线程服务页面，工作线程采集；显示确认后再发送标记。"""
from __future__ import annotations

import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import secrets
import threading
import time
import webbrowser

from .acquisition import CollectionCancelled, collect
from .config import load_config, validate_config, validate_participant_id
from .speech import SpeechCache
from .triggerbox import configure_triggerbox, network_defaults, probe_network_triggerbox, probe_triggerbox, serial_ports

ROOT = Path(__file__).resolve().parents[1]
MAX_CLASSES = 6
PROFILE_FILES = {"demo": ("TCP模拟 · 任务采集体验", "collection_ui_6class.json"),
                 "neuracle32": ("Neuracle · 32路 EEGCap32-V3.0 · 8712", "collection_ui_neuracle_ch32.json"),
                 "neuracle64": ("Neuracle · 64流设备示例", "collection_ui_neuracle_ch64.json")}


class CollectionController:
    def __init__(self, output_root, *, probe_log=None):
        self.output_root = Path(output_root)
        self.probe_log = Path(probe_log) if probe_log is not None else None
        self.condition = threading.Condition(threading.RLock())
        self.stop_event = threading.Event()
        self.worker = None
        self.phase_id = 0
        self.last_browser_seen = time.monotonic()
        self.phase_ack = None
        self.audio_error = ""
        self.trigger_probe_busy = False
        self.state = {"status": "idle", "running": False, "phase": None,
                      "samples_received": 0, "session": "", "error": "", "report": None}

    def profiles(self):
        return [{"id": key, "title": title, "max_classes": MAX_CLASSES,
                 "config": load_config(ROOT / "configs" / filename)}
                for key, (title, filename) in PROFILE_FILES.items()]

    def snapshot(self, *, browser_seen=False):
        with self.condition:
            if browser_seen:
                self.last_browser_seen = time.monotonic()
            result = copy.deepcopy(self.state)
            result["trigger_probe_busy"] = self.trigger_probe_busy
            phase = result.get("phase")
            if phase:
                deadline = phase.pop("deadline_monotonic", None)
                phase["remaining_seconds"] = (phase["duration_seconds"] if deadline is None
                                               else max(0, deadline - time.monotonic()))
            return result

    def test_triggerbox(self, port, connection=None):
        with self.condition:
            if self.state["running"] or self.trigger_probe_busy:
                raise ValueError("采集或连接测试正在进行，请结束后再测试TriggerBox。")
            self.trigger_probe_busy = True
        record = {"tested_wall_ns": time.time_ns(), "read_only": True,
                  "connection": copy.deepcopy(connection) if connection is not None else {"mode": "hardware", "port": port},
                  "ok": False}
        try:
            if connection is not None:
                if not isinstance(connection, dict) or connection.get("mode") != "network":
                    raise ValueError("网络连接测试参数无效。")
                result = probe_network_triggerbox(connection)
            else:
                result = probe_triggerbox(port)
            record.update(result)
            return result
        except Exception as exc:
            record["error"] = str(exc)
            raise
        finally:
            with self.condition:
                self.trigger_probe_busy = False
                self.state["trigger_probe_result"] = record
                if self.probe_log is not None:
                    try:
                        self.probe_log.parent.mkdir(parents=True, exist_ok=True)
                        with self.probe_log.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    except OSError as exc:
                        record["log_error"] = str(exc)

    def start(self, request):
        with self.condition:
            if self.trigger_probe_busy:
                raise ValueError("TriggerBox连接测试正在进行，请等测试结束后开始采集。")
            if self.state["running"]:
                raise ValueError("已有采集正在进行，请先停止。")
            if "config" in request:
                cfg = validate_config(request["config"])
            else:
                profile = request.get("profile", "neuracle32")
                if profile not in PROFILE_FILES:
                    raise ValueError("未知采集配置。")
                cfg = load_config(ROOT / "configs" / PROFILE_FILES[profile][1])
            if cfg["device"]["source"] not in {"synthetic_tcp", "neuracle"}:
                raise ValueError("此界面使用TCP模拟源或Neuracle设备源。")
            if not 2 <= len(cfg["classes"]) <= MAX_CLASSES:
                raise ValueError("采集界面支持2至6类任务。")
            trigger_settings = configure_triggerbox(cfg, request.get("triggerbox"))
            participant = validate_participant_id(str(request.get("participant", "P01")))
            times = request.get("collection", {})
            for key in ("initial_rest_seconds", "prepare_seconds", "task_seconds", "rest_seconds",
                        "block_rest_seconds"):
                if key in times:
                    number = float(times[key])
                    if not math.isfinite(number) or not 0 <= number <= 120:
                        raise ValueError("各阶段时长须在0至120秒之间。")
                    cfg["collection"][key] = number
            loops = request.get("loops", cfg["collection"]["loops"])
            if isinstance(loops, bool) or not math.isfinite(float(loops)) or not float(loops).is_integer() or not 1 <= int(loops) <= 100:
                raise ValueError("采集轮数须为1至100的整数。")
            cfg["collection"]["loops"] = int(loops)
            # 各次会话更换随机顺序/模拟噪声，并把实际种子保存在会话配置中。
            cfg["collection"]["seed"] = secrets.randbits(32)
            texts = request.get("task_texts")
            if texts is not None:
                if not isinstance(texts, list) or len(texts) != len(cfg["classes"]):
                    raise ValueError("任务文字数量与类别数不一致。")
                for c, text in zip(cfg["classes"], texts):
                    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 100:
                        raise ValueError("每条任务提示须为1至100个字符。")
                    c["task_text"] = text.strip()
            audio = request.get("audio", {"enabled": False, "volume": 70})
            if not isinstance(audio, dict) or not isinstance(audio.get("enabled"), bool):
                raise ValueError("声音开关须为布尔值。")
            volume = audio.get("volume", 70)
            if isinstance(volume, bool) or not isinstance(volume, (int, float)) or not math.isfinite(volume) or not 1 <= volume <= 100:
                raise ValueError("声音音量须为1至100；静音请关闭声音提示。")
            audio = {"enabled": audio["enabled"], "volume": volume,
                     "engine": "windows_local_wav_webaudio",
                     "prepare_rule": "max_of_prepare_from_display_and_speech_end"}
            prompts = request.get("prompt_texts", [c["task_text"] for c in cfg["classes"]])
            if (not isinstance(prompts, list) or len(prompts) != len(cfg["classes"])
                    or any(not isinstance(t, str) or not 1 <= len(t.strip()) <= 100 for t in prompts)):
                raise ValueError("屏幕/语音名称须与类别一一对应，每条1至100个字符。")
            prompts = [text.strip() for text in prompts]
            if len(set(prompts)) != len(prompts):
                raise ValueError("不同类别的屏幕/语音名称不能重复；相同任务请合并为一类并增加采集轮数。")
            cfg["session_metadata"] = {"participant_id": participant,
                                       "presentation": "browser_fullscreen_static_text",
                                       "prompt_texts": prompts,
                                       "audio": audio,
                                       "task_set_status": "candidate_unvalidated"}
            cfg = validate_config(cfg)
            self.stop_event.clear()
            self.phase_ack = None
            self.audio_error = ""
            self.last_browser_seen = time.monotonic()
            self.state = {"status": "connecting", "running": True, "phase": None,
                          "samples_received": 0, "sample_rate": cfg["device"]["sample_rate"],
                          "session": "", "error": "", "report": None,
                          "source": cfg["device"]["source"], "participant": participant,
                          "audio": audio,
                          "triggerbox": trigger_settings,
                          "total_trials": int(loops) * len(cfg["classes"])}
            self.worker = threading.Thread(target=self._run, args=(cfg,), name="handbci-collection", daemon=True)
            self.worker.start()
            return self.snapshot()

    def stop(self):
        self.stop_event.set()
        with self.condition:
            if self.state["running"]:
                self.state["status"] = "stopping"
            self.condition.notify_all()

    def ack(self, phase_id, render_time_ms, audio=None, prepare_elapsed_seconds=0):
        stamp = float(render_time_ms)
        if not math.isfinite(stamp):
            raise ValueError("无效的显示时间。")
        with self.condition:
            phase = self.state.get("phase")
            if phase and phase["id"] == phase_id and self.phase_ack is None:
                if self.state.get("audio", {}).get("enabled", False):
                    expected = {"initial_rest": "开始", "prepare": phase.get("prompt_text", phase.get("task_text", "")),
                                "task": "滴", "rest": "休息", "block_rest": "休息",
                                "finishing": "结束"}[phase["phase"]]
                    kind = "beep" if phase["phase"] == "task" else "speech"
                    if (not isinstance(audio, dict) or audio.get("enabled") is not True
                            or audio.get("text") != expected or audio.get("kind") != kind):
                        raise ValueError("声音提示与当前任务不一致，未开始本阶段。")
                    clean = {"enabled": True, "kind": kind, "text": expected,
                             "engine": "local_wav_webaudio"}
                    for key in ("scheduled_browser_ms", "duration_seconds", "completed_browser_ms"):
                        value = audio.get(key)
                        if value is None and key == "completed_browser_ms":
                            continue
                        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                            raise ValueError("声音时间记录无效。")
                        clean[key] = value
                    if not 0 < clean["duration_seconds"] <= 35:
                        raise ValueError("提示音时长无效。")
                    if phase["phase"] in {"prepare", "finishing"}:
                        if clean.get("completed_browser_ms", -1) < clean["scheduled_browser_ms"]:
                            raise ValueError("必须等语音播报完成再进入下一阶段。")
                else:
                    clean = {"enabled": False}
                self.phase_ack = {"display_ack_wall_ns": time.time_ns(),
                                  "browser_render_time_ms": stamp, "audio_cue": clean}
                remaining = phase["duration_seconds"]
                if phase["phase"] == "prepare":
                    elapsed = prepare_elapsed_seconds
                    if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                            or not math.isfinite(elapsed) or not 0 <= elapsed <= 45):
                        self.phase_ack = None
                        raise ValueError("准备阶段的已显示时间无效。")
                    remaining = max(0, remaining - elapsed)
                    self.phase_ack.update(prepare_display_elapsed_seconds=elapsed,
                                          prepare_remaining_seconds=remaining,
                                          prompt_text=phase.get("prompt_text", phase.get("task_text", "")))
                phase["awaiting_ack"] = False
                phase["deadline_monotonic"] = time.monotonic() + remaining
                self.condition.notify_all()

    def audio_failed(self, phase_id, message):
        with self.condition:
            phase = self.state.get("phase")
            if (self.state["running"] and self.state.get("audio", {}).get("enabled")
                    and phase and phase["id"] == phase_id):
                self.audio_error = "声音提示失败：" + str(message)[:300]
                self.stop()

    def _on_phase(self, payload):
        with self.condition:
            self.phase_id += 1
            requested = time.time_ns()
            self.phase_ack = None
            self.state["status"] = "collecting"
            self.state["phase"] = {**payload, "id": self.phase_id, "awaiting_ack": True,
                                   "deadline_monotonic": None}
            audio_enabled = self.state.get("audio", {}).get("enabled", False)
            deadline = time.monotonic() + (45 if audio_enabled else 5)
            while self.phase_ack is None:
                if self.stop_event.is_set():
                    raise CollectionCancelled("采集已停止；保留未完成记录。")
                if time.monotonic() >= deadline:
                    raise RuntimeError("提示页面未确认显示或声音播放，采集已停止。请保持提示页面可见。")
                if time.monotonic() - self.last_browser_seen > 5:
                    raise RuntimeError("提示页面已失去连接，采集已停止并保留未完成记录。")
                self.condition.wait(0.05)
            return {"display_requested_wall_ns": requested, **self.phase_ack}

    def _on_progress(self, payload):
        with self.condition:
            self.state.update(payload)
            if time.monotonic() - self.last_browser_seen > 5:
                raise RuntimeError("提示页面已失去连接，采集已停止并保留未完成记录。")

    def _run(self, cfg):
        simulator = None
        status, error = "completed", ""
        try:
            callback = None
            if cfg["device"]["source"] == "synthetic_tcp":
                from .tcp_simulator import TcpEEGSimulator
                simulator = TcpEEGSimulator(cfg, progress=lambda _: None).start()
                cfg["device"]["port"] = simulator.port
                callback = simulator.send_event
            path = collect(cfg, self.output_root, trigger_port=cfg["device"].get("trigger_port") or None,
                           on_cue=lambda _: None, on_phase=self._on_phase, on_progress=self._on_progress,
                           stop_event=self.stop_event, simulated_trigger=callback)
            with self.condition:
                self.state["session"] = str(path.resolve())
        except CollectionCancelled as exc:
            status = "failed" if self.audio_error else "cancelled"
            error = self.audio_error or str(exc)
        except Exception as exc:
            status = "failed"
            error = (f"设备软件未在 {cfg['device'].get('host', '127.0.0.1')}:{cfg['device'].get('port', 8712)} 接受连接。请在厂家 Data Sending Confirm 窗口点击 Start，确认设备正在采集并开启数据转发后重试。"
                     if isinstance(exc, ConnectionRefusedError) else f"{type(exc).__name__}: {exc}")
        finally:
            if simulator is not None:
                try:
                    simulator.stop()
                except Exception as exc:
                    status, error = "failed", error or str(exc)
            with self.condition:
                session = self.state.get("session", "")
                report_path = Path(session) / "collection_report.json" if session else None
                report = None
                if report_path and report_path.is_file():
                    try:
                        report = json.loads(report_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError) as exc:
                        status, error = "failed", error or f"无法读取采集汇总：{exc}"
                self.state.update(status=status, running=False, error=error, report=report)
                self.condition.notify_all()


def create_ui_server(controller, *, port=0):
    token = secrets.token_hex(24)
    assets = Path(__file__).resolve().parent / "ui"
    speech = SpeechCache(ROOT / "runtime" / "audio_cache")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, payload, content_type="application/json; charset=utf-8"):
            data = (json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
                    if not isinstance(payload, bytes) else payload)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def valid_host(self):
            return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

        def do_GET(self):
            if not self.valid_host():
                self.reply(403, {"error": "仅允许本机页面访问。"})
            elif self.path == "/api/state":
                self.reply(200, controller.snapshot(browser_seen=True))
            elif self.path == "/api/profiles":
                self.reply(200, controller.profiles())
            elif self.path == "/api/triggerbox/ports":
                self.reply(200, serial_ports())
            elif self.path == "/api/triggerbox/defaults":
                self.reply(200, network_defaults())
            elif self.path.startswith("/audio/") and self.path.endswith(".wav"):
                try:
                    path = speech.file(self.path[len("/audio/"):-4])
                    self.reply(200, path.read_bytes(), "audio/wav")
                except (ValueError, OSError):
                    self.reply(404, {"error": "提示音不存在，请重新试听。"})
            elif self.path in {"/", "/app.js", "/cues.js", "/styles.css"}:
                name = "index.html" if self.path == "/" else self.path[1:]
                data = (assets / name).read_bytes().replace(b"__UI_TOKEN__", token.encode())
                mime = {"index.html": "text/html; charset=utf-8", "app.js": "text/javascript; charset=utf-8",
                        "cues.js": "text/javascript; charset=utf-8",
                        "styles.css": "text/css; charset=utf-8"}[name]
                self.reply(200, data, mime)
            else:
                self.reply(404, {"error": "页面不存在。"})

        def do_POST(self):
            if not self.valid_host() or self.headers.get("X-Collection-Token") != token:
                self.reply(403, {"error": "页面验证失败，请重新打开本机提示页面。"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 100_000:
                    raise ValueError("请求大小无效。")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("请求须为JSON对象。")
                if self.path == "/api/start":
                    self.reply(200, controller.start(payload))
                elif self.path == "/api/triggerbox/test":
                    self.reply(200, controller.test_triggerbox(payload.get("port"), payload.get("connection")))
                elif self.path == "/api/audio/prepare":
                    if controller.snapshot()["running"]:
                        raise ValueError("采集中不能更换语音，请结束后再试听。")
                    self.reply(200, speech.prepare(payload.get("texts")))
                elif self.path == "/api/audio-error":
                    controller.audio_failed(int(payload["phase_id"]), payload.get("error", "播放失败"))
                    self.reply(200, {"ok": True})
                elif self.path == "/api/ack":
                    controller.ack(int(payload["phase_id"]), payload["render_time_ms"], payload.get("audio"),
                                   payload.get("prepare_elapsed_seconds", 0))
                    self.reply(200, {"ok": True})
                elif self.path == "/api/stop":
                    controller.stop()
                    self.reply(200, {"ok": True})
                elif self.path == "/api/shutdown":
                    controller.stop()
                    self.reply(200, {"ok": True})

                    def shutdown():
                        if controller.worker:
                            controller.worker.join()
                        self.server.shutdown()

                    threading.Thread(target=shutdown, daemon=True).start()
                else:
                    self.reply(404, {"error": "接口不存在。"})
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.reply(400, {"error": str(exc)})

    server = ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)
    server.daemon_threads = True
    return server


def run_collection_ui(output_root, *, port=0, open_browser=True):
    controller = CollectionController(output_root, probe_log=ROOT / "runtime/analysis/20261020_180658_8a745e9d_P01/real_device_timing_20261021/triggerbox_connection_checks.jsonl")
    server = create_ui_server(controller, port=port)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"采集界面：{url}", flush=True)
    print("结束时点击页面中的“关闭程序”；终端窗口需保持运行。", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        controller.stop()
    finally:
        controller.stop()
        if controller.worker:
            controller.worker.join()
        server.server_close()
    return {"ui_closed": True}
