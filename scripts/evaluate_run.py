#!/usr/bin/env python3
"""Run one DistTrainBench trial and save its trace, metadata, and final score."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import os
import platform
import re
import secrets
import shutil
import stat
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
IMAGE = "disttrainbench:dev"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_dotenv(path: Path) -> dict[str, str]:
    """Read the simple KEY=value entries used by this repository's .env file."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def builtin_agent_command(agent: str, model: str | None) -> list[str]:
    if agent == "codex":
        command = [
            "codex", "exec", "--json", "--skip-git-repo-check",
            "--sandbox", "danger-full-access",
        ]
        if model:
            command.extend(["--model", model])
        command.append("-")
        return command
    if agent == "claude":
        command = [
            "claude", "--print", "--verbose", "--output-format", "stream-json",
            "--permission-mode", "bypassPermissions",
        ]
        if model:
            command.extend(["--model", model])
        return command
    raise ValueError(f"Unsupported built-in agent: {agent}")


def render_task_prompt(
    template_path: Path,
    *,
    run_id: str,
    time_limit_seconds: int,
    workers: list[str],
) -> str:
    template = template_path.read_text(encoding="utf-8")
    replacements = {
        "{{CHECKPOINT}}": "/runs/checkpoint.pt",
        "{{DATA_DIR}}": "/data",
        "{{RUN_ID}}": run_id,
        "{{TIME_LIMIT}}": str(time_limit_seconds),
        "{{WORKERS}}": ", ".join(workers),
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def log_event(path: Path, event: str, **fields: Any) -> None:
    record = {"time_utc": utc_now(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(record, sort_keys=True) + "\n")


def run_checked(
    command: list[str], *, cwd: Path = ROOT, timeout: int = 60, env: dict[str, str] | None = None
) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        env=env,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"Command failed ({Path(command[0]).name}): {detail[-2000:]}")
    return result.stdout.strip()


def read_cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith(("model name", "hardware")):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def available_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return os.cpu_count() or 1


def benchmark_openblas(size: int, requested_threads: int) -> dict[str, Any]:
    library_name = ctypes.util.find_library("openblas")
    if not library_name:
        return {
            "status": "unavailable",
            "reason": "OpenBLAS shared library not found",
            "matrix_size": size,
            "requested_threads": requested_threads,
            "threads": None,
        }
    try:
        library = ctypes.CDLL(library_name)
        set_threads = library.openblas_set_num_threads
        set_threads.argtypes = [ctypes.c_int]
        get_threads = library.openblas_get_num_threads
        get_threads.restype = ctypes.c_int
        get_config = library.openblas_get_config
        get_config.restype = ctypes.c_char_p
        sgemm = library.cblas_sgemm
        sgemm.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_float, ctypes.c_void_p, ctypes.c_int,
        ]
        sgemm.restype = None
        threads = max(1, min(requested_threads, available_cpu_count()))
        set_threads(threads)
        actual_threads = int(get_threads())
        nbytes = size * size * ctypes.sizeof(ctypes.c_float)
        # Zeroed inputs are sufficient: SGEMM still executes the full matrix multiply.
        a = ctypes.create_string_buffer(nbytes)
        b = ctypes.create_string_buffer(nbytes)
        c = ctypes.create_string_buffer(nbytes)

        def multiply() -> None:
            sgemm(101, 111, 111, size, size, size, 1.0, a, size, b, size, 0.0, c, size)

        multiply()  # warm-up
        elapsed = []
        for _ in range(3):
            start = time.perf_counter()
            multiply()
            elapsed.append(time.perf_counter() - start)
        median_seconds = sorted(elapsed)[len(elapsed) // 2]
        config = get_config() or b""
        config_text = config.decode("utf-8", errors="replace")
        version_match = re.search(r"OpenBLAS\s+([^\s]+)", config_text)
        return {
            "status": "completed",
            "library": library_name,
            "openblas_version": version_match.group(1) if version_match else None,
            "openblas_config": config_text,
            "matrix_size": size,
            "requested_threads": requested_threads,
            "threads": actual_threads,
            "repetitions": len(elapsed),
            "median_seconds": median_seconds,
            "gflop_per_second": (2.0 * size**3 / median_seconds) / 1e9,
        }
    except (AttributeError, OSError, ValueError) as error:
        return {
            "status": "unavailable",
            "library": library_name,
            "reason": str(error),
            "matrix_size": size,
            "requested_threads": requested_threads,
            "threads": None,
        }


def host_metadata(matrix_size: int, benchmark_threads: int, do_benchmark: bool) -> dict[str, Any]:
    memory_total = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                memory_total = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu_model": read_cpu_model(),
        "visible_cpu_count": available_cpu_count(),
        "memory_total_bytes": memory_total,
        "python_version": sys.version.split()[0],
        "fp32_matmul": benchmark_openblas(matrix_size, benchmark_threads)
        if do_benchmark
        else {"status": "skipped"},
    }


def inspect_config(compose: list[str], env: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(run_checked(compose + ["config", "--format", "json"], env=env))
    services = config.get("services", {})
    resource_limits = {
        name: {"cpus": service.get("cpus"), "memory": service.get("mem_limit")}
        for name, service in services.items()
        if name == "coordinator" or name.startswith("worker-")
    }
    workers = sorted(name for name in services if name.startswith("worker-"))
    network_names = sorted((config.get("networks") or {}).keys())
    network = {
        name: {
            "driver": (config.get("networks", {}).get(name) or {}).get("driver", "bridge"),
            "profile": "unshaped Docker bridge; no bandwidth or latency limits configured",
        }
        for name in network_names
    }
    return {"resource_limits": resource_limits, "workers": workers, "network": network}, config


def container_limits(compose: list[str], services: list[str], env: dict[str, str]) -> dict[str, Any]:
    limits: dict[str, Any] = {}
    for service in services:
        container_id = run_checked(compose + ["ps", "-q", service], env=env).splitlines()
        if not container_id:
            continue
        inspect = json.loads(run_checked(["docker", "inspect", container_id[0]]))[0]
        host_config = inspect.get("HostConfig", {})
        limits[service] = {
            "nano_cpus": host_config.get("NanoCpus"),
            "cpu_quota": host_config.get("CpuQuota"),
            "cpu_period": host_config.get("CpuPeriod"),
            "memory_bytes": host_config.get("Memory"),
        }
    return limits


def container_package_versions(compose: list[str], env: dict[str, str]) -> dict[str, Any]:
    probe = (
        "import json,sys; import torch,torchvision; "
        "print(json.dumps({'python':sys.version.split()[0], 'pytorch':torch.__version__, "
        "'torchvision':torchvision.__version__}))"
    )
    versions = json.loads(
        run_checked(
            compose + ["exec", "-T", "coordinator", "python", "-c", probe],
            timeout=60,
            env=env,
        )
    )
    for cli in ("codex", "claude"):
        result = subprocess.run(
            compose + ["exec", "-T", "coordinator", cli, "--version"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
            env=env,
            check=False,
        )
        version = result.stdout.strip()
        versions[f"{cli}_cli"] = version or None
    return versions


def wait_for_worker_apis(
    compose: list[str], workers: list[str], env: dict[str, str]
) -> None:
    probe = (
        "import json,time,urllib.request; "
        f"workers={workers!r}; "
        "token=__import__('os').environ['DISTTRAINBENCH_WORKER_TOKEN']; "
        "\nfor worker in workers:\n"
        " deadline=time.monotonic()+45\n"
        " while time.monotonic()<deadline:\n"
        "  try:\n"
        "   req=urllib.request.Request(f'http://{worker}:8765/health',headers={'Authorization':f'Bearer {token}'})\n"
        "   with urllib.request.urlopen(req,timeout=2) as response:\n"
        "    if json.load(response).get('status')=='ok': break\n"
        "  except Exception:\n"
        "   time.sleep(0.5)\n"
        " else: raise SystemExit(f'worker API did not become ready: {worker}')\n"
        "print('worker APIs ready')"
    )
    run_checked(
        compose + ["exec", "-T", "coordinator", "python", "-c", probe],
        timeout=60,
        env=env,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_records(private_dir: Path, public_dir: Path) -> None:
    """Publish host-owned logs after training; never follow agent-created symlinks."""
    if public_dir.is_symlink() or not public_dir.is_dir():
        raise RuntimeError("Run output directory was removed or replaced during the agent run")
    for source in private_dir.iterdir():
        if source.name == "agent-workspace" and source.is_dir() and not source.is_symlink():
            shutil.copytree(source, public_dir / source.name, ignore=ignore_symlinks)
            continue
        if source.is_symlink() or not source.is_file():
            continue
        temporary = public_dir / f".{source.name}.{uuid.uuid4().hex}.tmp"
        with source.open("rb") as src, temporary.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(temporary, public_dir / source.name)


def ignore_symlinks(directory: str, entries: list[str]) -> set[str]:
    ignored = set()
    for name in entries:
        mode = (Path(directory) / name).lstat().st_mode
        if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            ignored.add(name)
    return ignored


def configured_secrets(*environments: dict[str, str]) -> list[tuple[str, str]]:
    secrets_found = []
    for environment in environments:
        for name, value in environment.items():
            if name.endswith(("_API_KEY", "_TOKEN")) and len(value) >= 8:
                pair = (name, value)
                if pair not in secrets_found:
                    secrets_found.append(pair)
    return sorted(secrets_found, key=lambda pair: len(pair[1]), reverse=True)


def redact_text(text: str, secrets_found: list[tuple[str, str]]) -> str:
    for name, value in secrets_found:
        text = text.replace(value, f"[REDACTED:{name}]")
    return text


def redact_record_tree(root: Path, secrets_found: list[tuple[str, str]]) -> None:
    """Redact keys and tokens from saved traces and agent-created text files."""
    if not secrets_found:
        return
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            raw = path.read_bytes()
            if b"\0" in raw:
                continue
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        sanitized = redact_text(text, secrets_found)
        if sanitized != text:
            path.write_text(sanitized, encoding="utf-8")


def clear_agent_outputs(public_dir: Path) -> None:
    """Keep only the final checkpoint written by the agent in its shared output folder."""
    if public_dir.is_symlink() or not public_dir.is_dir():
        raise RuntimeError("Run output directory was removed or replaced during the agent run")
    for child in public_dir.iterdir():
        if child.name == "checkpoint.pt" and child.is_file() and not child.is_symlink():
            continue
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    agent_group = parser.add_mutually_exclusive_group(required=True)
    agent_group.add_argument("--agent", choices=("codex", "claude"), help="Run a supported coding-agent CLI")
    agent_group.add_argument("--agent-command", help="Custom shell command to run inside coordinator")
    parser.add_argument("--model", help="Model identifier passed to the selected built-in CLI")
    parser.add_argument("--api-key-env", choices=("CODEX_API_KEY", "ANTHROPIC_API_KEY"),
                        help="Provider key to expose to a custom command")
    parser.add_argument("--prompt-file", help="Prompt file for the agent; defaults to the benchmark prompt")
    parser.add_argument("--time-limit-seconds", type=int, default=3600)
    parser.add_argument("--run-id", help="Lowercase letters, digits, and hyphens")
    parser.add_argument("--benchmark-matrix-size", type=int, default=4096)
    parser.add_argument("--benchmark-threads", type=int, default=2)
    parser.add_argument("--skip-host-benchmark", action="store_true")
    parser.add_argument("--no-build", action="store_true", help="Use the existing Docker image")
    args = parser.parse_args()
    if args.agent and args.api_key_env:
        parser.error("--api-key-env is only for --agent-command; built-in agents receive their provider key")
    if args.time_limit_seconds <= 0:
        parser.error("--time-limit-seconds must be positive")
    if args.benchmark_matrix_size <= 0 or args.benchmark_threads <= 0:
        parser.error("matrix size and benchmark thread count must be positive")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", run_id):
        parser.error("run id must contain only lowercase letters, digits, and hyphens")

    host_environment = read_dotenv(ROOT / ".env")
    host_environment.update(os.environ)
    api_key_env = args.api_key_env
    if args.agent == "codex":
        api_key_env = "CODEX_API_KEY"
        if not host_environment.get("CODEX_API_KEY") and host_environment.get("OPENAI_API_KEY"):
            host_environment["CODEX_API_KEY"] = host_environment["OPENAI_API_KEY"]
    elif args.agent == "claude":
        api_key_env = "ANTHROPIC_API_KEY"
    if api_key_env and not host_environment.get(api_key_env):
        parser.error(f"{api_key_env} is empty; set it in .env or the shell environment")

    prompt_template_path: Path | None = None
    if args.prompt_file:
        prompt_template_path = Path(args.prompt_file).expanduser()
        if not prompt_template_path.is_absolute():
            prompt_template_path = Path.cwd() / prompt_template_path
        if not prompt_template_path.is_file():
            parser.error(f"prompt file does not exist: {prompt_template_path}")
    elif args.agent:
        prompt_template_path = ROOT / "prompts" / "training_task.md"

    built_command = builtin_agent_command(args.agent, args.model) if args.agent else None
    command_for_hash = args.agent_command if args.agent_command else json.dumps(built_command)

    docker = shutil.which("docker")
    if not docker:
        parser.error("docker CLI was not found")
    project = f"dtb-{run_id}"
    compose = [docker, "compose", "-f", str(ROOT / "compose.yaml"), "-p", project]
    results_root = ROOT / "runs"
    results_root.mkdir(parents=True, exist_ok=True)
    if results_root.is_symlink():
        parser.error("runs/ must be a real directory, not a symlink")
    public_run_dir = results_root / run_id
    public_run_dir.mkdir(exist_ok=False)
    run_dir = ROOT / ".disttrainbench" / "records" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    workspace_source = ROOT / "workspace"
    if workspace_source.is_symlink():
        parser.error("workspace/ must be a real directory, not a symlink")
    workspace_source.mkdir(parents=True, exist_ok=True)
    workspace_snapshot = ROOT / ".disttrainbench" / "workspaces" / run_id
    workspace_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(workspace_source, workspace_snapshot, ignore=ignore_symlinks)
    training_data = ROOT / ".disttrainbench" / "training-data" / run_id
    training_data.mkdir(parents=True, exist_ok=False)
    worker_token = secrets.token_urlsafe(32)
    compose_environment = host_environment.copy()
    compose_environment.update(
        {
            "WORKSPACE_DIR": str(workspace_snapshot),
            "TRAIN_DATA_DIR": str(training_data),
            "RUN_OUTPUT_DIR": str(public_run_dir),
            "WORKER_TOKEN": worker_token,
        }
    )
    private_eval_data = ROOT / ".disttrainbench" / "evaluation-data"
    private_eval_data.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "metadata.json"
    events_path = run_dir / "events.jsonl"
    metadata: dict[str, Any] = {
        "benchmark": "DistTrainBench",
        "schema_version": 1,
        "run_id": run_id,
        "status": "initializing",
        "created_at_utc": utc_now(),
        "agent": args.agent or "custom",
        "model": args.model,
        "api_key_env": api_key_env,
        "agent_command_sha256": hashlib.sha256(command_for_hash.encode("utf-8")).hexdigest(),
        "time_limit_seconds": args.time_limit_seconds,
        "score_definition": "CIFAR-100 test top-1 accuracy after the agent exits",
        "host": host_metadata(args.benchmark_matrix_size, args.benchmark_threads, not args.skip_host_benchmark),
    }
    log_event(events_path, "run_created", run_id=run_id)
    write_json(metadata_path, metadata)
    up_started = False
    agent_return_code: int | None = None
    timed_out = False
    try:
        docker_version = run_checked([docker, "version", "--format", "{{.Server.Version}}"])
        metadata["docker_server_version"] = docker_version
        config_meta, _ = inspect_config(compose, compose_environment)
        metadata.update(config_meta)
        metadata["worker_count"] = len(config_meta["workers"])
        metadata["worker_api"] = {
            "port": 8765,
            "transport": "HTTP on the per-run Docker bridge network",
            "authentication": "random per-run bearer token",
        }
        write_json(metadata_path, metadata)

        up = compose + ["up", "-d"]
        if not args.no_build:
            up.append("--build")
        print(f"[{run_id}] building and starting containers", flush=True)
        log_event(events_path, "containers_starting")
        up_started = True
        run_checked(up, timeout=1800, env=compose_environment)
        metadata["actual_container_limits"] = container_limits(
            compose, ["coordinator", *config_meta["workers"]], compose_environment
        )
        metadata["software_versions"] = container_package_versions(compose, compose_environment)
        metadata["worker_count"] = len(config_meta["workers"])
        wait_for_worker_apis(compose, config_meta["workers"], compose_environment)
        write_json(metadata_path, metadata)
        log_event(events_path, "containers_ready", worker_count=metadata["worker_count"])

        checkpoint = "/runs/checkpoint.pt"
        exec_env = {
            "DISTTRAINBENCH_RUN_ID": run_id,
            "DISTTRAINBENCH_RUN_DIR": "/runs",
            "DISTTRAINBENCH_CHECKPOINT": checkpoint,
            "DISTTRAINBENCH_TIME_LIMIT_SECONDS": str(args.time_limit_seconds),
            "DISTTRAINBENCH_TRAIN_DATA_DIR": "/data",
            "DISTTRAINBENCH_WORKERS": ",".join(config_meta["workers"]),
        }
        prompt_text = (
            render_task_prompt(
                prompt_template_path,
                run_id=run_id,
                time_limit_seconds=args.time_limit_seconds,
                workers=config_meta["workers"],
            )
            if prompt_template_path
            else None
        )
        if prompt_text is not None:
            prompt_path = run_dir / "agent.prompt.txt"
            prompt_path.write_text(prompt_text, encoding="utf-8")
            metadata["agent_prompt"] = "agent.prompt.txt"
            metadata["agent_prompt_sha256"] = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()

        command = compose + ["exec", "-T"]
        command.append("-i" if prompt_text is not None else "--interactive=false")
        command.extend(["-w", "/workspace"])
        for key, value in exec_env.items():
            command.extend(["-e", f"{key}={value}"])
        if api_key_env:
            # Compose reads the value from the runner environment. The key is
            # set only on this exec process, not on the long-running container.
            command.extend(["-e", api_key_env])
        command += [
            "coordinator", "timeout", "--signal=TERM", "--kill-after=15s",
            f"{args.time_limit_seconds}s",
        ]
        if built_command:
            command.extend(built_command)
        else:
            command.extend(["bash", "-lc", args.agent_command])
        log_event(
            events_path,
            "agent_started",
            agent=args.agent or "custom",
            time_limit_seconds=args.time_limit_seconds,
        )
        print(f"[{run_id}] agent started; time limit is {args.time_limit_seconds}s", flush=True)
        metadata["agent_started_at_utc"] = utc_now()
        write_json(metadata_path, metadata)
        agent_started = time.monotonic()
        with (run_dir / "agent.stdout.log").open("wb") as stdout, (run_dir / "agent.stderr.log").open("wb") as stderr:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=compose_environment,
                stdin=subprocess.PIPE if prompt_text is not None else subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
            )
            try:
                process.communicate(
                    input=prompt_text.encode("utf-8") if prompt_text is not None else None,
                    timeout=args.time_limit_seconds + 30,
                )
                agent_return_code = process.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                process.terminate()
                try:
                    process.communicate(timeout=5)
                    agent_return_code = process.returncode
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
                    agent_return_code = process.returncode
                # A lost docker exec connection can leave its child running; stop coordinator to clear it.
                subprocess.run(
                    compose + ["stop", "-t", "2", "coordinator"],
                    cwd=ROOT,
                    env=compose_environment,
                    check=False,
                )
        duration = time.monotonic() - agent_started
        metadata["agent_finished_at_utc"] = utc_now()
        metadata["duration_seconds"] = round(duration, 3)
        metadata["agent_exit_code"] = agent_return_code
        metadata["timed_out"] = timed_out or agent_return_code == 124
        metadata["agent_trace"] = {
            "stdout": "agent.stdout.log",
            "stderr": "agent.stderr.log",
            "stdout_sha256": sha256(run_dir / "agent.stdout.log"),
            "stderr_sha256": sha256(run_dir / "agent.stderr.log"),
        }
        log_event(
            events_path,
            "agent_finished",
            exit_code=agent_return_code,
            timed_out=metadata["timed_out"],
            duration_seconds=metadata["duration_seconds"],
        )

        logs = subprocess.run(
            compose + ["logs", "--no-color", "--timestamps"],
            cwd=ROOT,
            env=compose_environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
            check=False,
        )
        (run_dir / "container.log").write_text(logs.stdout, encoding="utf-8")
        metadata["container_log"] = "container.log"
        write_json(metadata_path, metadata)

        run_checked(
            compose + ["down", "--remove-orphans"], timeout=120, env=compose_environment
        )
        up_started = False
        log_event(events_path, "training_containers_stopped")

        if public_run_dir.is_symlink() or not public_run_dir.is_dir():
            metadata["status"] = "checkpoint_missing"
            metadata["error"] = "Agent removed or replaced its run output directory"
            checkpoint_path = run_dir / "checkpoint.pt"
        else:
            agent_checkpoint = public_run_dir / "checkpoint.pt"
            checkpoint_path = run_dir / "checkpoint.pt"
            if agent_checkpoint.is_file() and not agent_checkpoint.is_symlink():
                shutil.copyfile(agent_checkpoint, checkpoint_path)
            clear_agent_outputs(public_run_dir)
        if checkpoint_path.is_file():
            print(f"[{run_id}] running final CIFAR-100 evaluation", flush=True)
            eval_container = f"dtb-eval-{run_id}"
            eval_command = [
                docker, "run", "--rm", "--name", eval_container,
                "--network", "bridge", "--cpus=1", "--memory=2g",
                "--volume", f"{run_dir}:/run:rw",
                "--volume", f"{private_eval_data}:/evaluation-data:rw",
                "--entrypoint", "python", IMAGE, "-m", "disttrainbench.evaluate",
                "--checkpoint", "/run/checkpoint.pt",
                "--data-root", "/evaluation-data",
                "--output", "/run/evaluation.json",
            ]
            log_event(events_path, "final_evaluation_started")
            eval_start = time.monotonic()
            try:
                eval_result = subprocess.run(
                    eval_command,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                subprocess.run([docker, "rm", "-f", eval_container], cwd=ROOT, check=False)
                (run_dir / "evaluation.log").write_bytes(error.stdout or b"")
                raise RuntimeError("Final evaluator exceeded its 30 minute limit") from error
            (run_dir / "evaluation.log").write_bytes(eval_result.stdout)
            metadata["evaluation_duration_seconds"] = round(time.monotonic() - eval_start, 3)
            metadata["evaluation_exit_code"] = eval_result.returncode
            metadata["evaluation_log"] = "evaluation.log"
            if eval_result.returncode == 0 and (run_dir / "evaluation.json").is_file():
                evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
                metadata["score"] = evaluation["top1_accuracy"]
                metadata["score_percent"] = evaluation["top1_accuracy_percent"]
                metadata["evaluation"] = "evaluation.json"
                metadata["status"] = "completed"
                log_event(events_path, "final_evaluation_completed", score=metadata["score"])
            else:
                metadata["status"] = "evaluation_failed"
                metadata["error"] = f"Evaluator exited with code {eval_result.returncode}"
                log_event(events_path, "final_evaluation_failed", exit_code=eval_result.returncode)
        else:
            metadata["status"] = "checkpoint_missing"
            metadata["error"] = f"Expected final checkpoint at {checkpoint}"
            log_event(events_path, "checkpoint_missing", expected=checkpoint)

        if metadata.get("status") == "completed" and metadata.get("timed_out"):
            metadata["status"] = "completed_time_limit"
        elif metadata.get("status") == "completed" and agent_return_code != 0:
            metadata["status"] = "completed_agent_nonzero_exit"
        write_json(metadata_path, metadata)
        print(
            f"[{run_id}] status={metadata['status']} "
            f"score={metadata.get('score_percent', 'unavailable')}",
            flush=True,
        )
        return 0 if metadata["status"].startswith("completed") else 1
    except (OSError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        metadata["status"] = "harness_failed"
        metadata["error"] = str(error)
        write_json(metadata_path, metadata)
        log_event(events_path, "harness_failed", error=str(error))
        return 1
    finally:
        if up_started:
            subprocess.run(
                compose + ["down", "--remove-orphans"],
                cwd=ROOT,
                env=compose_environment,
                check=False,
            )
        if workspace_snapshot.exists() and not (run_dir / "agent-workspace").exists():
            try:
                shutil.copytree(
                    workspace_snapshot,
                    run_dir / "agent-workspace",
                    ignore=ignore_symlinks,
                )
            except OSError as error:
                print(f"Could not snapshot agent workspace: {error}", file=sys.stderr, flush=True)
        records_sanitized = False
        try:
            secrets_found = configured_secrets(
                read_dotenv(ROOT / ".env"), os.environ, compose_environment
            )
            redact_record_tree(run_dir, secrets_found)
            trace = metadata.get("agent_trace")
            if isinstance(trace, dict):
                for stream in ("stdout", "stderr"):
                    trace_path = run_dir / f"agent.{stream}.log"
                    if trace_path.is_file():
                        trace[f"{stream}_sha256"] = sha256(trace_path)
            metadata["trace_redaction"] = {
                "applied": bool(secrets_found),
                "redacted_secret_count": len(secrets_found),
            }
            write_json(metadata_path, metadata)
            records_sanitized = True
        except (OSError, RuntimeError) as error:
            print(f"Could not sanitize run artifacts; leaving them in the private records directory: {error}",
                  file=sys.stderr, flush=True)
        if records_sanitized:
            try:
                publish_records(run_dir, public_run_dir)
            except (OSError, RuntimeError) as error:
                print(f"Could not publish run artifacts: {error}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
