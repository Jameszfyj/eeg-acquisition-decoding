"""Loopback-only HTTP workbench. Start: python -m handbci.studio.server."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs
from .controller import Controller
from .devices import ADAPTERS
from ..triggerbox import serial_ports


def make_server(root, port=57373):
    c = Controller(root)
    token = secrets.token_urlsafe(32)
    web = Path(__file__).parent/'web'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass

        def handle(self):
            try: super().handle()
            except (ConnectionResetError, BrokenPipeError): pass

        def send(self, value, code=200):
            raw = json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
            self.respond(raw,'application/json; charset=utf-8',code)

        def respond(self, raw, content_type, code=200):
            self.send_response(code)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(raw)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('X-Frame-Options','DENY')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; media-src 'self' blob:; frame-ancestors 'none'")
            self.end_headers()
            try: self.wfile.write(raw)
            except (BrokenPipeError,ConnectionResetError): pass

        def valid_host(self):
            return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')

        def do_GET(self):
            if not self.valid_host(): return self.send({'error':'Host rejected'},403)
            u = urlsplit(self.path)
            q = {k:v[0] for k,v in parse_qs(u.query).items()}
            try:
                if u.path == '/api/state': return self.send({**c.state(),'token':token,'service':'handbci.studio',
                    'server_root':str(c.root),'version':'2026.10.23.3'})
                if u.path == '/api/waveform': return self.send(c.waveform())
                if u.path == '/api/ports': return self.send(serial_ports())
                if u.path == '/api/sessions': return self.send(c.sessions())
                if u.path == '/api/review': return self.send(c.review(q['id'],int(q.get('trial',0)),q.get('start'),q.get('duration')))
                if u.path.startswith('/audio/'):
                    key = u.path.removeprefix('/audio/').removesuffix('.wav')
                    return self.respond(c.speech.file(key).read_bytes(),'audio/wav')
                files = {'/':('index.html','text/html; charset=utf-8'), '/app.js':('app.js','text/javascript; charset=utf-8'),
                         '/style.css':('style.css','text/css; charset=utf-8')}
                if u.path not in files: return self.send({'error':'Not found'},404)
                name,mime = files[u.path]
                return self.respond((web/name).read_bytes(),mime)
            except BaseException as exc: return self.send({'error':str(exc)},400)

        def do_POST(self):
            if not self.valid_host(): return self.send({'error':'Host rejected'},403)
            origin = self.headers.get('Origin')
            allowed = (f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}')
            if origin not in allowed or self.headers.get('X-Studio-Token') != token:
                return self.send({'error':'同源验证失败，请刷新工作台。'},403)
            try:
                length = int(self.headers.get('Content-Length','0'))
                route = urlsplit(self.path).path
                limit = 46*1024*1024 if route == '/api/validation/model' else 1048576
                if not 0 < length <= limit: raise ValueError('请求大小无效。')
                body = json.loads(self.rfile.read(length))
                if route == '/api/connect': result = c.connect(body['adapter'],body.get('options',{}))
                elif route == '/api/disconnect': result = c.disconnect()
                elif route == '/api/impedance': result = c.impedance(bool(body['enabled']))
                elif route == '/api/discover': result = ADAPTERS[body['adapter']].discover()
                elif route == '/api/trigger/connect': result = c.trigger_connect(body['mode'],body.get('port',''),
                    wireless_host=str(body.get('wireless_host','')).strip(), wireless_port=int(body.get('wireless_port',4097)))
                elif route == '/api/trigger/test': result = c.trigger_test()
                elif route == '/api/audio': result = c.prepare_audio(body['protocol'])
                elif route == '/api/validation/model': result = c.import_validation(body['participant'],body['data'],body['filename'],body.get('manifest'))
                elif route == '/api/start': result = c.start(body['participant'],body['protocol'],require_subject=True,
                    validation=bool(body.get('validation',False)),validation_model_id=body.get('validation_model_id'))
                elif route == '/api/subject-heartbeat':
                    import time
                    c.subject_heartbeat = time.perf_counter()
                    result = None
                elif route == '/api/abort': result = c.abort(body.get('message','操作员中止'))
                elif route == '/api/client-event': result = c.local_event('subject_client', **{k:body[k] for k in ('revision','status','client_wall_ms') if k in body})
                else: return self.send({'error':'Not found'},404)
                return self.send({'ok':True,'result':result})
            except BaseException as exc: return self.send({'error':str(exc)},400)

    server = ThreadingHTTPServer(('127.0.0.1',int(port)),Handler)
    server.daemon_threads = True
    server.controller = c
    return server


def main():
    parser = argparse.ArgumentParser(description='脑电采集工作台（仅本机）')
    parser.add_argument('--port',type=int,default=57373)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    server = make_server(args.root,args.port)
    print(f'脑电采集工作台 http://127.0.0.1:{server.server_port} · 默认不连接设备',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        c = server.controller
        c.abort('服务关闭')
        if c.experiment: c.experiment.thread.join(30)
        c.disconnect()
        server.server_close()


if __name__ == '__main__': main()
