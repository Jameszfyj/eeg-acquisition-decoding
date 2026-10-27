"""Bound an isolated TriggerBox connection check, including stuck native I/O.

The subprocess owns only EEG/TriggerBox test connections, never the hand.
Cancellation/timeout terminates that process before releasing the controller's
exclusive test slot. A timeout never reports the hardware check as passed.
"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import time

from .config import write_json

def _wait_process(process, cancel, timeout):
    deadline=time.monotonic()+timeout
    try:
        while process.poll() is None:
            if cancel.is_set():
                raise RuntimeError('TriggerBox连接检验已取消，测试连接已释放。')
            if time.monotonic()>=deadline:
                raise RuntimeError('TriggerBox连接检验超时，测试连接已释放；请检查USB与无线连接后重新检验。')
            try:process.wait(timeout=.1)
            except subprocess.TimeoutExpired:pass
        if cancel.is_set():
            raise RuntimeError('TriggerBox连接检验已取消，未将结果记为通过。')
        return process.returncode
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill();process.wait(timeout=3)

def run_trigger_check(folder, device, settings, cancel, timeout=25.):
    folder=Path(folder)
    folder.parent.mkdir(parents=True,exist_ok=True)
    config_path=folder.parent/(folder.name+'_request.json')
    log_path=folder.parent/(folder.name+'_worker.log')
    write_json(config_path,dict(folder=str(folder.resolve()),device=device,settings=settings))
    root=Path(__file__).resolve().parents[1]
    try:
        if cancel.is_set():raise RuntimeError('TriggerBox连接检验已取消。')
        with log_path.open('xb') as log:
            process=subprocess.Popen([sys.executable,'-B','-X','utf8','-m','handbci.trigger_probe',str(config_path)],
                cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            code=_wait_process(process,cancel,timeout)
        path=folder/'connection_check.json'
        if not path.is_file():raise RuntimeError('连接检验未返回诊断结果，请查看测试日志。')
        record=json.loads(path.read_text(encoding='utf-8'))
        if code!=0 or record.get('error') or not record.get('result'):
            raise RuntimeError(record.get('error') or 'TriggerBox连接检验失败。')
        return record['result']
    except Exception as exc:
        folder.mkdir(parents=True,exist_ok=True)
        write_json(folder/'connection_check.json',dict(result=None,error=str(exc),complete=False,
            cancelled=cancel.is_set(),test_process_isolated=True,physical_action=False))
        raise

def main():
    import threading
    from .trigger_connection import probe_trigger_connection
    request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    try:
        probe_trigger_connection(Path(request['folder']),request['device'],request['settings'],threading.Event())
    except Exception as exc:
        print(str(exc),flush=True)
        return 1
    return 0

if __name__=='__main__':raise SystemExit(main())
