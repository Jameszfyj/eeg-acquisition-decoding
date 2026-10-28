"""本机在线解码网页：按被试选模型，显示实时信号、概率和模拟动作。"""
from __future__ import annotations

import base64
import copy
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import numpy as np
from pathlib import Path
import secrets
import socket
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid
import webbrowser

from .acquisition import CollectionCancelled
from .config import ACTION_NAMES, validate_config, validate_participant_id, write_json
from .subject_online import (DEFAULT_REGISTRY, checkpoint_participant, file_sha256,
                             prepare_subject_model, read_registry, register_model)
from .training import load_model

ROOT = Path(__file__).resolve().parents[1]


class LocalOnlineServer(ThreadingHTTPServer):
    # Windows otherwise permits multiple listeners on the same loopback port,
    # leaving requests routed to an older controller after an apparent restart.
    allow_reuse_address = not hasattr(socket, 'SO_EXCLUSIVEADDRUSE')

    def server_bind(self):
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class OnlineController:
    service_name = 'handbci.online.v1'

    def __init__(self, output_root, *, registry=DEFAULT_REGISTRY, import_root=None):
        self.output_root = Path(output_root)
        self.registry = Path(registry)
        self.import_root = Path(import_root) if import_root else ROOT / "runtime/models/imported"
        self.condition = threading.RLock()
        self.stop_event = threading.Event()
        self.worker = None
        self.simulator = None
        self.last_simulation_label = None
        self.last_browser_seen = time.monotonic()
        self.trace = deque(maxlen=1000)
        self.state = {"status": "idle", "running": False, "session": "", "error": "",
                      "samples_received": 0, "n_predictions": 0, "commands": [],
                      "probabilities": [], "phase": "idle", "trial": 0,
                      "trial_remaining": 0, "inference_ms": 0, "report": None,
                      "simulation_label": 1}

    def models(self):
        return read_registry(self.registry)["models"]

    def snapshot(self, *, browser_seen=False):
        with self.condition:
            if browser_seen:
                self.last_browser_seen = time.monotonic()
            result = copy.deepcopy(self.state)
            result["trace"] = list(self.trace)
            return result

    def model_details(self, participant, model_id):
        selection, cfg = prepare_subject_model(participant, model_id, registry=self.registry)
        return {**selection, "classes": cfg["classes"], "channels": cfg["device"]["channels"],
                "unit": cfg["device"]["unit"], "decision": cfg["decision"],
                "window_seconds": cfg["window"]["seconds"]}

    def _prepare(self, request):
        selection, cfg = prepare_subject_model(request["participant"], request["model_id"], registry=self.registry)
        mode = request.get("mode", "simulated" if selection["synthetic_model"] else "device")
        if mode not in {"simulated", "device"}:
            raise ValueError("请选择模拟数据或真实设备。")
        if mode == "simulated":
            if not selection["synthetic_model"]:
                raise ValueError("此模型由真实脑电训练，请连接对应设备；人工演示请使用人工模型。")
            cfg["device"].update(source="synthetic_tcp", host="127.0.0.1", port=0,
                                 stream_channels=len(cfg["device"]["channels"]) + 1)
        else:
            if selection["synthetic_model"]:
                raise ValueError("当前模型来自人工信号，不能连接真实设备；请先导入真实被试模型。")
            host = str(request.get("host", cfg["device"].get("host", "127.0.0.1"))).strip()
            if not host or len(host) > 253 or any(c.isspace() for c in host):
                raise ValueError("数据转发地址无效。")
            try:
                port = int(request.get("port", cfg["device"].get("port", 8712)))
            except (TypeError, ValueError):
                raise ValueError("端口须为1至65535的整数。")
            if not 1 <= port <= 65535:
                raise ValueError("端口须为1至65535的整数。")
            cfg["device"].update(source="neuracle", host=host, port=port)
        trials = request.get("trials", 3)
        if isinstance(trials, bool) or not float(trials).is_integer() or not 1 <= int(trials) <= 100:
            raise ValueError("判断次数须为1至100的整数。")
        label = request.get("simulation_label", 1)
        if isinstance(label, bool) or not float(label).is_integer() or not 0 <= int(label) < len(cfg["classes"]):
            raise ValueError("模拟任务编号无效。")
        cfg["session_metadata"]["presentation"] = "browser_online_monitor"
        return selection, validate_config(cfg), mode, int(trials), int(label)

    def check(self, request):
        selection, cfg, mode, trials, label = self._prepare(request)
        return {**selection, "mode": mode, "trials": trials, "connected": False,
                "message": "模型与配置检查通过，尚未连接数据源。"}

    def start(self, request):
        with self.condition:
            if self.state["running"]:
                raise ValueError("已有在线解码正在运行，请先停止。")
            selection, cfg, mode, trials, label = self._prepare(request)
            self.trace.clear()
            self.stop_event.clear()
            self.last_browser_seen = time.monotonic()
            self.last_simulation_label = None
            self.state = {"status": "connecting", "running": True, "session": "", "error": "",
                          "samples_received": 0, "n_predictions": 0, "commands": [],
                          "probabilities": [], "phase": "warming", "trial": 0, "trial_remaining": 0,
                          "inference_ms": 0, "report": None, "selection": selection,
                          "participant": selection["participant_id"], "mode": mode,
                          "simulation_label": label, "classes": cfg["classes"],
                          "channel_name": cfg["device"]["channels"][0], "unit": cfg["device"]["unit"],
                          "sample_rate": cfg["device"]["sample_rate"], "total_trials": trials,
                          "window_seconds": cfg["window"]["seconds"], "decision": cfg["decision"]}
            self.worker = threading.Thread(target=self._run, args=(selection, cfg, mode, trials),
                                           name="handbci-online-ui", daemon=True)
            self.worker.start()
            return self.snapshot()

    def stop(self):
        self.stop_event.set()
        with self.condition:
            if self.state["running"]:
                self.state["status"] = "stopping"

    def set_simulation(self, label):
        with self.condition:
            if not self.state["running"] or self.state.get("mode") != "simulated":
                raise ValueError("只有运行中的模拟模式可以切换人工任务。")
            if isinstance(label, bool) or not float(label).is_integer() or not 0 <= int(label) < len(self.state["classes"]):
                raise ValueError("模拟任务编号无效。")
            self.state["simulation_label"] = int(label)
            return {"simulation_label": int(label)}

    def _update(self, update):
        with self.condition:
            if time.monotonic() - self.last_browser_seen > 5:
                raise CollectionCancelled("在线页面失去连接，已停止并保留记录。")
            kind = update.get("kind")
            if kind == "session":
                self.state["session"] = update["session"]
            elif kind == "progress":
                self.state.update({k: update[k] for k in ("samples_received", "trial", "phase", "trial_remaining")})
                self.trace.extend(update.get("trace", []))
                if self.trace:
                    cutoff = self.trace[-1][0] - 5
                    while self.trace and self.trace[0][0] < cutoff:
                        self.trace.popleft()
                if self.state["status"] != "stopping":
                    self.state["status"] = "running"
                label = self.state["simulation_label"]
                if self.simulator is not None and self.simulator.connected.is_set() and label != self.last_simulation_label:
                    self.simulator.send_event(label + 1)
                    self.last_simulation_label = label
            elif kind == "prediction":
                self.state["n_predictions"] += 1
                self.state["probabilities"] = update["probabilities"]
                self.state["inference_ms"] = update["inference_ms"]
            elif kind == "command":
                row = update["command"]
                row["task_text"] = self.state["classes"][row["class_id"]]["task_text"]
                row["action_text"] = ACTION_NAMES[row["action"]]
                self.state["commands"].append(row)

    def _run(self, selection, cfg, mode, trials):
        status, error = "completed", ""
        try:
            if mode == "simulated":
                from .tcp_simulator import TcpEEGSimulator
                simulator = TcpEEGSimulator(cfg, port=0, progress=lambda _: None).start()
                with self.condition:
                    self.simulator = simulator
                cfg["device"]["port"] = simulator.port
            from .runtime import run_live
            run_live(selection["model"], cfg, self.output_root, trials=trials,
                     progress=lambda _: None, stop_event=self.stop_event, on_update=self._update)
        except CollectionCancelled as exc:
            status, error = "stopped", str(exc)
        except Exception as exc:
            status = "failed"
            error = ("无法连接数据源，请先在厂家软件中启动数据转发并核对地址、端口。"
                     if isinstance(exc, ConnectionRefusedError) else f"{type(exc).__name__}: {exc}")
        finally:
            with self.condition:
                simulator = self.simulator
                self.simulator = None
            if simulator is not None:
                try:
                    simulator.stop()
                except Exception as exc:
                    status, error = "failed", error or str(exc)
            with self.condition:
                session = self.state.get("session")
                if session and (Path(session) / "data.h5").is_file():
                    try:
                        import h5py
                        with h5py.File(Path(session) / "data.h5", "r") as handle:
                            self.state["samples_received"] = int(handle["eeg"].shape[1])
                    except (OSError, KeyError, ValueError) as exc:
                        status, error = "failed", error or f"无法复核保存的采样点数：{exc}"
                report = {"complete": status == "completed", "participant_id": selection["participant_id"],
                          "model_id": selection["model_id"], "model_path": selection["model"],
                          "source": cfg["device"]["source"], "mode": mode,
                          "samples_received": self.state["samples_received"],
                          "duration_seconds": self.state["samples_received"] / cfg["device"]["sample_rate"],
                          "n_predictions": self.state["n_predictions"], "n_commands": len(self.state["commands"]),
                          "driver": "mock", "physical_action": False, "error": error}
                if session:
                    try:
                        write_json(Path(session) / "model_selection.json", selection)
                        write_json(Path(session) / "online_ui_report.json", report)
                    except (OSError, ValueError) as exc:
                        status, error = "failed", error or str(exc)
                        report.update(complete=False, error=error)
                self.state.update(status=status, running=False, phase="finished", error=error, report=report)

    def import_model(self, request):
        with self.condition:
            if self.state["running"]:
                raise ValueError("请先停止在线解码，再导入模型。")
            participant = validate_participant_id(str(request["participant"]))
            name = str(request.get("name", "")).strip()[:100] or None
            try:
                data = base64.b64decode(request["data"], validate=True)
            except (ValueError, TypeError):
                raise ValueError("模型文件传输无效，请重新选择。")
            if not 0 < len(data) <= 32 * 1024 * 1024:
                raise ValueError("模型文件须为不超过32MB的整理版.pt文件。")
            if data.startswith(b'PK'):
                import zipfile, io
                with zipfile.ZipFile(io.BytesIO(data)) as bundle:
                    is_onnx = 'manifest.json' in bundle.namelist()
                if is_onnx:
                    from .model_bundle import import_bundle
                    return import_bundle(data, participant, registry=self.registry,
                                         import_root=self.import_root, name=name)
            folder = self.import_root / (time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8] + "_" + participant)
            folder.mkdir(parents=True, exist_ok=False)
            model = folder / "model.pt"
            try:
                model.write_bytes(data)
                from .workbench_settings import local_settings
                if local_settings().get('studio_v2'):
                    import torch
                    with torch.serialization.safe_globals([np.core.multiarray._reconstruct,np.ndarray,np.dtype,
                            type(np.dtype(np.float64)),type(np.dtype(np.float32)),type(np.dtype(np.int64))]):
                        probe=torch.load(model,map_location='cpu',weights_only=True)
                    if isinstance(probe,dict) and probe.get('format')=='jixieshou.six_class_offline.v1':
                        from .studio_v2_runtime import register_model as register_studio
                        from .subject_online import resolve_path
                        options=local_settings()['studio_v2']
                        return register_studio(participant,model,resolve_path(options['source_manifest']),self.registry,name)
                import torch
                metadata=torch.load(model,map_location='cpu',weights_only=True)
                if metadata.get('format')=='handbci.p01_six_retrained_20261027.v1':
                    from .retrained_runtime import register_model as register_retrained
                    from .subject_online import resolve_path
                    entry=register_retrained(participant,model,self.registry,name)
                    if resolve_path(entry['path'])!=model.resolve():
                        model.unlink();folder.rmdir()
                    return entry
                payload, _ = load_model(model)
                if checkpoint_participant(payload) != participant:
                    raise ValueError("文件内部的被试编号与填写编号不一致。")
                digest = file_sha256(model)
                for entry in self.models():
                    if entry["participant_id"] == participant and entry.get("sha256") == digest:
                        model.unlink()
                        folder.rmdir()
                        return entry
                return register_model(participant, model, registry=self.registry, name=name)
            except Exception as exc:
                if model.is_file():
                    model.unlink()
                if folder.is_dir() and not any(folder.iterdir()):
                    folder.rmdir()
                raise ValueError(f"模型导入失败：{exc}") from exc


def create_online_server(controller, *, port=0, assets=None):
    token = secrets.token_hex(24)
    assets = Path(assets) if assets else Path(__file__).resolve().parent / "online_ui"
    from .screen_recording import ScreenRecordings
    recordings = ScreenRecordings(controller.output_root/'screen_recordings')
    from .calibration_import import CalibrationImports
    calibration_imports = CalibrationImports(controller)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, payload, content_type="application/json; charset=utf-8"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
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
                return
            parsed = urlsplit(self.path)
            try:
                if parsed.path == '/api/service':
                    self.reply(200, {'application': controller.service_name,
                                     'root': str(ROOT.resolve())})
                elif parsed.path == "/api/state":
                    if hasattr(controller, 'browser_snapshot'):
                        query = parse_qs(parsed.query)
                        state = controller.browser_snapshot(query.get('subject_window_id', [None])[0])
                    else:
                        state = controller.snapshot(browser_seen=True)
                    self.reply(200, state)
                elif parsed.path == "/api/models":
                    self.reply(200, controller.models())
                elif parsed.path == '/api/recording/file':
                    query = parse_qs(parsed.query)
                    path, mime = recordings.file(query.get('recording_id', [''])[0])
                    self.send_response(200)
                    self.send_header('Content-Type', mime)
                    self.send_header('Content-Length', str(path.stat().st_size))
                    self.send_header('Content-Disposition', 'attachment; filename="'+path.name+'"')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('X-Content-Type-Options', 'nosniff')
                    self.end_headers()
                    with path.open('rb') as video:
                        while data := video.read(1024*1024):
                            self.wfile.write(data)
                elif parsed.path == "/api/model":
                    query = parse_qs(parsed.query)
                    self.reply(200, controller.model_details(query["participant"][0], query["model_id"][0]))
                elif parsed.path == "/api/hand/ports" and hasattr(controller, "hand_ports"):
                    self.reply(200, controller.hand_ports())
                elif parsed.path == '/cues.js' and hasattr(controller, 'speech'):
                    self.reply(200, (Path(__file__).parent/'ui/cues.js').read_bytes(), 'text/javascript; charset=utf-8')
                elif parsed.path.startswith('/audio/') and hasattr(controller, 'speech'):
                    key = parsed.path.removeprefix('/audio/').removesuffix('.wav')
                    self.reply(200, controller.speech.file(key).read_bytes(), 'audio/wav')
                elif parsed.path in ('/calibration_prompts.js','/action_prompts.js') and hasattr(controller, 'speech'):
                    self.reply(200, (assets/parsed.path.lstrip('/')).read_bytes(), 'text/javascript; charset=utf-8')
                elif parsed.path.startswith('/task-images/'):
                    name = parsed.path.removeprefix('/task-images/')
                    allowed = {'01_thumb.png','02_index.png','03_middle.png',
                               '04_ring.png','05_little.png','06_pinch.png'}
                    if name not in allowed:
                        self.reply(404, {'error': '图片不存在。'})
                    else:
                        image = assets.parent.parent/'assets/task_cues_20261025'/name
                        self.reply(200, image.read_bytes(), 'image/png')
                elif parsed.path in {"/", "/app.js", "/styles.css", "/subject", "/subject.js", "/subject.css", "/screen_recorder.js"}:
                    name = "index.html" if parsed.path == "/" else parsed.path[1:]
                    if name == 'subject':
                        name = 'subject.html'
                    data = (assets / name).read_bytes().replace(b"__UI_TOKEN__", token.encode())
                    self.reply(200, data, {"index.html": "text/html; charset=utf-8",
                                          "subject.html": "text/html; charset=utf-8",
                                          "subject.js": "text/javascript; charset=utf-8",
                                          "screen_recorder.js": "text/javascript; charset=utf-8",
                                          "subject.css": "text/css; charset=utf-8",
                                          "app.js": "text/javascript; charset=utf-8",
                                          "styles.css": "text/css; charset=utf-8"}[name])
                else:
                    self.reply(404, {"error": "页面不存在。"})
            except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
                self.reply(400, {"error": str(exc)})

        def do_POST(self):
            if not self.valid_host() or self.headers.get("X-Online-Token") != token:
                self.reply(403, {"error": "页面验证失败，请重新打开在线页面。"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                parsed = urlsplit(self.path)
                if parsed.path == '/api/calibration/import/chunk':
                    if not 0 < length <= calibration_imports.MAX_CHUNK:
                        raise ValueError('校准上传分段须不超过8MB。')
                    query = parse_qs(parsed.query)
                    data = self.rfile.read(length)
                    if len(data) != length:raise ValueError('校准数据分段不完整。')
                    self.reply(200,calibration_imports.append(query.get('id',[''])[0],
                        query.get('name',[''])[0],int(query.get('offset',['-1'])[0]),data))
                    return
                if parsed.path == '/api/recording/chunk':
                    if not 0 < length <= recordings.MAX_CHUNK:
                        raise ValueError('录屏分段须不超过16MB。')
                    query = parse_qs(parsed.query)
                    data = self.rfile.read(length)
                    if len(data) != length:
                        raise ValueError('录屏分段传输不完整。')
                    self.reply(200, recordings.append(query.get('recording_id', [''])[0],
                        query.get('subject_window_id', [''])[0], int(query.get('sequence', ['-1'])[0]), data))
                    return
                if not 0 < length <= 46 * 1024 * 1024:
                    raise ValueError("请求大小无效。")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("请求须为JSON对象。")
                if self.path == "/api/check":
                    self.reply(200, controller.check(payload))
                elif self.path == '/api/calibration/import/open':
                    self.reply(200,calibration_imports.open(payload))
                elif self.path == '/api/calibration/import/finish':
                    self.reply(200,calibration_imports.finish(payload))
                elif self.path == '/api/calibration/import/cancel':
                    self.reply(200,calibration_imports.cancel(payload))
                elif self.path == '/api/recording/open':
                    self.reply(200, recordings.open(payload))
                elif self.path == '/api/recording/finish':
                    self.reply(200, recordings.finish(payload, controller.snapshot()))
                elif self.path == "/api/start":
                    self.reply(200, controller.start_decoding(payload) if hasattr(controller,'start_decoding')
                               else controller.start(payload))
                elif self.path == '/api/calibration/start' and hasattr(controller,'start_calibration'):
                    self.reply(200, controller.start_calibration(payload))
                elif self.path == '/api/decode/start' and hasattr(controller,'start_decoding'):
                    self.reply(200, controller.start_decoding(payload))
                elif self.path == "/api/eeg/check" and hasattr(controller, 'check_eeg_connection'):
                    self.reply(200, controller.check_eeg_connection(payload))
                elif self.path == "/api/eeg/discover" and hasattr(controller, 'discover_eeg_devices'):
                    self.reply(200, controller.discover_eeg_devices(payload))
                elif self.path == '/api/trigger/discover' and hasattr(controller, 'discover_trigger_devices'):
                    self.reply(200, controller.discover_trigger_devices(payload))
                elif self.path == '/api/trigger/check' and hasattr(controller, 'check_trigger_connection'):
                    self.reply(200, controller.check_trigger_connection(payload))
                elif self.path == "/api/stop":
                    controller.stop()
                    self.reply(200, {"ok": True})
                elif self.path == '/api/disconnect' and hasattr(controller, 'disconnect_devices'):
                    self.reply(200, controller.disconnect_devices())
                elif self.path == '/api/subject/stop' and hasattr(controller, 'stop_subject'):
                    self.reply(200, controller.stop_subject(payload))
                elif self.path == '/api/monitor/stop' and hasattr(controller, 'stop_monitor'):
                    self.reply(200, controller.stop_monitor(payload))
                elif self.path == "/api/simulation-task":
                    self.reply(200, controller.set_simulation(payload["label"]))
                elif self.path == "/api/calibration/ack" and hasattr(controller, 'acknowledge_calibration'):
                    self.reply(200, controller.acknowledge_calibration(payload))
                elif self.path == '/api/prompted/ack' and hasattr(controller, 'acknowledge_prompted'):
                    self.reply(200, controller.acknowledge_prompted(payload))
                elif self.path == '/api/audio/prepare' and hasattr(controller, 'prepare_calibration_audio'):
                    self.reply(200, controller.prepare_calibration_audio(payload))
                elif self.path == "/api/import-model":
                    self.reply(200, controller.import_model(payload))
                elif self.path == "/api/hand/check" and hasattr(controller, "check_hand_connection"):
                    self.reply(200, controller.check_hand_connection(payload))
                elif self.path == "/api/hand/test" and hasattr(controller, "start_hand_test"):
                    self.reply(200, controller.start_hand_test(payload))
                elif self.path == "/api/hand/test/stop" and hasattr(controller, "stop_hand_test"):
                    self.reply(200, controller.stop_hand_test())
                elif self.path == "/api/shutdown":
                    controller.stop()
                    self.reply(200, {"ok": True})
                    def shutdown():
                        if hasattr(controller, 'eeg_probe_done'):
                            controller.eeg_probe_done.wait(30)
                        if controller.worker:
                            controller.worker.join()
                        if hasattr(controller, "manual_hand") and controller.manual_hand.worker:
                            controller.manual_hand.worker.join()
                        self.server.shutdown()
                    threading.Thread(target=shutdown, daemon=True).start()
                else:
                    self.reply(404, {"error": "接口不存在。"})
            except (ValueError, KeyError, TypeError, OSError, RuntimeError, EOFError) as exc:
                self.reply(400, {"error": str(exc)})

    server = LocalOnlineServer(("127.0.0.1", int(port)), Handler)
    server.daemon_threads = True
    return server


def run_online_ui(output_root, *, registry=DEFAULT_REGISTRY, port=0, open_browser=True):
    import torch
    torch.set_num_threads(2)
    controller = OnlineController(output_root, registry=registry)
    server = create_online_server(controller, port=port)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"在线控制界面：{url}", flush=True)
    print("先选择被试和模型。结束时点击页面中的“关闭程序”。", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        controller.stop()
    finally:
        controller.stop()
        if controller.worker:
            controller.worker.join(timeout=8)
        server.server_close()
    return {"url": url, "output_root": str(Path(output_root).resolve()), "closed": True}
