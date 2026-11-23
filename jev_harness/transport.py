"""Persistent JSON-line RPC, locally or over the existing SSH connection."""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
import threading
import time
from queue import Queue, Empty


class WorkerUnavailable(RuntimeError):
    """Infrastructure failure, outside the agent's action space."""


class WorkerError(RuntimeError):
    """Error returned by a worker that is still communicating."""


class Worker:
    def __init__(self, spec: dict, log_path: Path):
        command = [spec["python"], "-u", spec["script"], *spec.get("args", [])]
        if "host" in spec:
            if spec.get("temp_dir"):
                command = ["env", "TMPDIR="+spec["temp_dir"], *command]
            command = ["ssh", *(["-C"] if spec.get("compression") else []), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                       "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4",
                       "-i", spec["identity"], "-p", str(spec["port"]),
                       spec["user"] + "@" + spec["host"], shlex.join(command)]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log_path.open("a", encoding="utf-8")
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, encoding="utf-8", bufsize=1)
        self.lock = threading.Lock()
        self.broken = False
        self.index = 0
        self.started = time.monotonic()
        self.read_timeout = spec.get("read_timeout")
        if self.read_timeout is not None:
            self.lines = Queue()
            def read_lines():
                try:
                    for line in self.process.stdout:
                        self.lines.put(line)
                finally:
                    self.lines.put("")
            threading.Thread(target=read_lines, daemon=True).start()

    def request(self, op: str, **values):
        with self.lock:
            if self.broken:
                raise WorkerUnavailable(f"Worker connection already closed; inspect {self.log.name}")
            self.index += 1
            request = {"id": self.index, "op": op, **values}
            try:
                self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                self.process.stdin.flush()
                line = self.lines.get(timeout=self.read_timeout) if self.read_timeout else self.process.stdout.readline()
                if not line:
                    raise WorkerUnavailable(f"Worker exited ({self.process.poll()}); inspect {self.log.name}")
                result = json.loads(line)
                if not isinstance(result, dict) or result.get("id") != self.index:
                    raise WorkerUnavailable("Worker response belongs to another request")
            except (OSError, Empty, json.JSONDecodeError, WorkerUnavailable) as exc:
                self.broken = True
                if self.process.poll() is None:
                    self.process.terminate()
                if isinstance(exc, WorkerUnavailable):
                    raise
                reason = "Invalid worker response" if isinstance(exc, json.JSONDecodeError) else "Worker stopped responding"
                raise WorkerUnavailable(f"{reason}; inspect {self.log.name}") from exc
            if result.get("error"):
                raise WorkerError(result["error"])
            return result["result"]

    def close(self):
        try:
            if not self.broken and self.process.poll() is None:
                self.request("close")
        finally:
            try:
                try:
                    self.process.stdin.close()
                except BrokenPipeError:
                    pass
                try:
                    self.process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    self.process.wait(timeout=20)
            finally:
                self.process.stdout.close()
                self.log.close()
