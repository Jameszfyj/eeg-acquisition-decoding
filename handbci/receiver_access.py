"""Exclusive ownership of a Neuracle forwarding endpoint, before TCP connect."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
from pathlib import Path
import socket
import subprocess
import threading

ROOT = Path(__file__).resolve().parents[1] / "runtime/receiver_locks"
_registry_lock = threading.Lock()
_held = set()


def _canonical_host(host):
    value = str(host).strip()
    if value.lower() == "localhost":
        return "127.0.0.1"
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        return socket.gethostbyname(value)


def existing_clients(host, port):
    """Windows endpoint table only; never probes/connects to the data port."""
    if os.name != "nt":
        return []  # File lock still excludes cooperating receivers on POSIX.
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    result = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True,
                            timeout=12, creationflags=flags)
    if result.returncode:
        raise RuntimeError("无法核对脑电端口占用，未建立新的采集连接。")
    matches = []
    for line in result.stdout.decode(errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 5 or fields[0] != "TCP" or fields[3] != "ESTABLISHED":
            continue
        address, separator, target_port = fields[2].rpartition(":")
        if separator and target_port == str(port) and address.strip("[]") == host:
            matches.append({"pid": fields[4], "local": fields[1], "remote": fields[2]})
    return matches


class ReceiverLease:
    """A process-safe lock held until the receiver socket has been closed."""
    def __init__(self, host, port, *, lock_root=None):
        self.host, self.port = _canonical_host(host), int(port)
        self.key = f"{self.host}:{self.port}"
        self.root = Path(lock_root) if lock_root is not None else ROOT
        self.handle = None
        self.registered = False

    def acquire(self):
        with _registry_lock:
            if self.key in _held:
                raise ValueError(f"{self.key}已有本程序的采集连接，请先停止另一处采集或诊断。")
            _held.add(self.key)
            self.registered = True
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.root / (hashlib.sha256(self.key.encode()).hexdigest() + ".lock")
            descriptor = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
            self.handle = os.fdopen(descriptor, "r+b")
            if os.fstat(descriptor).st_size == 0:
                self.handle.write(b"\x00")
                self.handle.flush()
            self.handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ValueError(f"{self.key}已有另一进程在采集或诊断，请先停止该程序；不会重复连接。") from exc
            clients = existing_clients(self.host, self.port)
            if clients:
                pids = ", ".join(sorted(set(client["pid"] for client in clients)))
                raise ValueError(f"{self.key}已被接收程序占用（PID {pids}），请先停止其他实时接收；不会抢占连接。")
            self.handle.seek(1)
            self.handle.write(json.dumps({"pid": os.getpid(), "endpoint": self.key}).encode())
            self.handle.truncate()
            self.handle.flush()
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.handle is not None:
            self.handle.close()  # OS releases the byte-range/flock, including on process exit.
            self.handle = None
        if self.registered:
            with _registry_lock:
                _held.discard(self.key)
            self.registered = False

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.close()
