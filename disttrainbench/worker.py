"""Small authenticated job API for coordinator-to-worker execution."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse


HOST = "0.0.0.0"
PORT = int(os.environ.get("DISTTRAINBENCH_WORKER_PORT", "8765"))
TOKEN = os.environ.get("DISTTRAINBENCH_WORKER_TOKEN", "")
MAX_REQUEST_BYTES = 1024 * 1024
MAX_JOBS = 16
MAX_RETAINED_JOBS = 128
MAX_TAIL_BYTES = 512 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def trace(event: str, **fields: Any) -> None:
    print(json.dumps({"worker_event": event, "time_utc": utc_now(), **fields}), flush=True)


class Job:
    def __init__(self, command: list[str], cwd: str, env: dict[str, str]) -> None:
        self.id = uuid.uuid4().hex
        self.command = command
        self.cwd = cwd
        self.started_at = utc_now()
        self.finished_at: str | None = None
        self.returncode: int | None = None
        self.output_tail = bytearray()
        self.output_truncated = False
        self.output_lock = threading.Lock()
        self.process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()
        self.waiter = threading.Thread(target=self._wait, daemon=True)
        self.waiter.start()

    def _read_output(self) -> None:
        assert self.process.stdout is not None
        while True:
            chunk = self.process.stdout.read(8192)
            if not chunk:
                break
            with self.output_lock:
                self.output_tail.extend(chunk)
                overflow = len(self.output_tail) - MAX_TAIL_BYTES
                if overflow > 0:
                    del self.output_tail[:overflow]
                    self.output_truncated = True
            sys.stdout.write(f"[job {self.id}] {chunk.decode('utf-8', errors='replace')}")
            sys.stdout.flush()

    def _wait(self) -> None:
        self.returncode = self.process.wait()
        self.finished_at = utc_now()
        self.reader.join(timeout=1)
        trace(
            "job_finished",
            job_id=self.id,
            returncode=self.returncode,
            started_at_utc=self.started_at,
            finished_at_utc=self.finished_at,
        )

    def as_dict(self, include_output: bool = False) -> dict[str, Any]:
        result = {
            "job_id": self.id,
            "command": self.command,
            "cwd": self.cwd,
            "started_at_utc": self.started_at,
            "finished_at_utc": self.finished_at,
            "running": self.returncode is None,
            "returncode": self.returncode,
        }
        if include_output:
            with self.output_lock:
                result["output_tail"] = bytes(self.output_tail).decode("utf-8", errors="replace")
                result["output_tail_truncated"] = self.output_truncated
        return result

    def terminate(self, sig: int) -> None:
        if self.returncode is None:
            try:
                os.killpg(self.process.pid, sig)
            except ProcessLookupError:
                pass


jobs: dict[str, Job] = {}
jobs_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = "DistTrainBenchWorker/1"

    def log_message(self, format_string: str, *args: Any) -> None:
        trace("http_request", client=self.client_address[0], message=format_string % args)

    def _authorized(self) -> bool:
        expected = f"Bearer {TOKEN}"
        supplied = self.headers.get("Authorization", "")
        return bool(TOKEN) and supplied == expected

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("request body is too large")
        data = self.rfile.read(length)
        parsed = json.loads(data or b"{}")
        if not isinstance(parsed, dict):
            raise ValueError("JSON body must be an object")
        return parsed

    def _route(self) -> tuple[str, str | None]:
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path in ("/health", "/jobs"):
            return path, None
        pieces = path.split("/")
        if len(pieces) in (3, 4) and pieces[1] == "jobs":
            action = pieces[3] if len(pieces) == 4 else None
            return "/jobs", pieces[2] if action is None else f"{pieces[2]}/{action}"
        return path, None

    def do_GET(self) -> None:  # noqa: N802
        if not self._authorized():
            self._send(401, {"error": "unauthorized"})
            return
        route, job_ref = self._route()
        if route == "/health":
            self._send(200, {"status": "ok", "worker": os.environ.get("HOSTNAME", "unknown")})
            return
        if route != "/jobs":
            self._send(404, {"error": "not found"})
            return
        with jobs_lock:
            if job_ref is None:
                self._send(200, {"jobs": [job.as_dict() for job in jobs.values()]})
                return
            job_id, _, suffix = job_ref.partition("/")
            job = jobs.get(job_id)
        if job is None:
            self._send(404, {"error": "unknown job"})
        else:
            self._send(200, job.as_dict(include_output=suffix == "output"))

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            self._send(401, {"error": "unauthorized"})
            return
        route, job_ref = self._route()
        if route != "/jobs":
            self._send(404, {"error": "not found"})
            return
        try:
            body = self._body()
        except (ValueError, json.JSONDecodeError) as error:
            self._send(400, {"error": str(error)})
            return

        if job_ref is None:
            command = body.get("command")
            cwd = body.get("cwd", "/workspace")
            extra_env = body.get("env", {})
            if (
                not isinstance(command, list)
                or not command
                or not all(isinstance(item, str) and item and "\x00" not in item for item in command)
            ):
                self._send(400, {"error": "command must be a non-empty list of strings"})
                return
            if not isinstance(cwd, str) or not os.path.isdir(cwd):
                self._send(400, {"error": "cwd must be an existing directory"})
                return
            if not isinstance(extra_env, dict) or not all(
                isinstance(key, str)
                and key
                and "=" not in key
                and "\x00" not in key
                and isinstance(value, str)
                and "\x00" not in value
                for key, value in extra_env.items()
            ):
                self._send(400, {"error": "env must map strings to strings"})
                return
            with jobs_lock:
                if sum(job.returncode is None for job in jobs.values()) >= MAX_JOBS:
                    self._send(429, {"error": "worker job limit reached"})
                    return
                while len(jobs) >= MAX_RETAINED_JOBS:
                    completed = next(
                        (job_id for job_id, old_job in jobs.items() if old_job.returncode is not None),
                        None,
                    )
                    if completed is None:
                        self._send(429, {"error": "worker job history limit reached"})
                        return
                    del jobs[completed]
                env = os.environ.copy()
                env.update(extra_env)
                try:
                    job = Job(command, cwd, env)
                except (OSError, ValueError) as error:
                    self._send(400, {"error": str(error)})
                    return
                jobs[job.id] = job
            trace("job_started", job_id=job.id, command=command, cwd=cwd)
            self._send(202, job.as_dict())
            return

        job_id, _, action = job_ref.partition("/")
        with jobs_lock:
            job = jobs.get(job_id)
        if job is None:
            self._send(404, {"error": "unknown job"})
            return
        if action == "signal":
            name = body.get("signal", "TERM")
            allowed = {"TERM": signal.SIGTERM, "INT": signal.SIGINT, "KILL": signal.SIGKILL}
            if name not in allowed:
                self._send(400, {"error": "signal must be TERM, INT, or KILL"})
                return
            job.terminate(allowed[name])
            trace("job_signaled", job_id=job_id, signal=name)
            self._send(200, job.as_dict())
            return
        self._send(404, {"error": "not found"})

    def do_DELETE(self) -> None:  # noqa: N802
        if not self._authorized():
            self._send(401, {"error": "unauthorized"})
            return
        route, job_ref = self._route()
        if route != "/jobs" or job_ref is None or "/" in job_ref:
            self._send(404, {"error": "not found"})
            return
        with jobs_lock:
            job = jobs.get(job_ref)
        if job is None:
            self._send(404, {"error": "unknown job"})
            return
        job.terminate(signal.SIGTERM)
        self._send(202, job.as_dict())


def main() -> None:
    if not TOKEN:
        raise SystemExit("DISTTRAINBENCH_WORKER_TOKEN must be set")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    trace("worker_api_started", host=HOST, port=PORT)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
