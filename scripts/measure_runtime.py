"""Measure local API startup time, inference latency, and peak RSS memory."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]


def _process_tree_rss_mb(process: psutil.Process) -> float:
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except psutil.Error:
        pass

    rss_bytes = 0
    for child in processes:
        try:
            rss_bytes += child.memory_info().rss
        except psutil.Error:
            continue
    return round(rss_bytes / (1024 * 1024), 2)


def _wait_for_health(
    *,
    base_url: str,
    process: subprocess.Popen,
    ps_process: psutil.Process,
    timeout_seconds: float,
    sample_interval: float,
) -> tuple[float, float]:
    started_at = time.perf_counter()
    peak_rss_mb = 0.0
    health_url = f"{base_url.rstrip('/')}/health"

    while True:
        peak_rss_mb = max(peak_rss_mb, _process_tree_rss_mb(ps_process))
        if process.poll() is not None:
            raise RuntimeError(f"Server exited early with code {process.returncode}.")

        try:
            response = requests.get(health_url, timeout=2)
            if response.status_code == 200:
                return round(time.perf_counter() - started_at, 3), peak_rss_mb
        except requests.RequestException:
            pass

        if time.perf_counter() - started_at > timeout_seconds:
            raise TimeoutError(f"Server did not become healthy within {timeout_seconds}s.")
        time.sleep(sample_interval)


def _measure_request(
    *,
    base_url: str,
    endpoint: str,
    text: str,
    ps_process: psutil.Process,
) -> tuple[dict[str, Any], float]:
    peak_rss_mb = _process_tree_rss_mb(ps_process)
    started_at = time.perf_counter()
    response = requests.post(
        f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}",
        json={"text": text},
        timeout=120,
    )
    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
    peak_rss_mb = max(peak_rss_mb, _process_tree_rss_mb(ps_process))
    response.raise_for_status()
    return {
        "endpoint": f"/{endpoint.lstrip('/')}",
        "latency_ms": latency_ms,
        "response": response.json(),
    }, peak_rss_mb


def measure_runtime(args: argparse.Namespace) -> dict[str, Any]:
    env = os.environ.copy()
    env["TRANSLATOR_DEMO_MODE"] = "0" if args.full_model else "1"
    env["TRANSLATOR_DEMO_MEMORY"] = "1"
    env["TRANSLATOR_AUTO_DOWNLOAD_MODEL"] = "0"

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.serve:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    ps_process = psutil.Process(process.pid)
    base_url = f"http://{args.host}:{args.port}"

    try:
        startup_seconds, startup_peak_rss_mb = _wait_for_health(
            base_url=base_url,
            process=process,
            ps_process=ps_process,
            timeout_seconds=args.startup_timeout,
            sample_interval=args.sample_interval,
        )
        request_result, inference_peak_rss_mb = _measure_request(
            base_url=base_url,
            endpoint="institutional-review",
            text=args.text,
            ps_process=ps_process,
        )
        peak_rss_mb = max(startup_peak_rss_mb, inference_peak_rss_mb)
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)

    model_path = REPO_ROOT / "best_model.pth"
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "full_model" if args.full_model else "demo",
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "artifacts": {
            "model_size_bytes": model_path.stat().st_size if model_path.is_file() else None,
            "tokenizer_present": (REPO_ROOT / "data" / "tokenizer").is_dir(),
        },
        "startup_seconds_to_health": startup_seconds,
        "startup_peak_rss_mb": startup_peak_rss_mb,
        "inference_peak_rss_mb": inference_peak_rss_mb,
        "peak_rss_mb": peak_rss_mb,
        "peak_vram_mb": None,
        "request": request_result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure local API runtime metrics.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--text", default="The parliamentary session was adjourned.")
    parser.add_argument("--output", default="runtime_metrics.json")
    parser.add_argument("--full-model", action="store_true")
    args = parser.parse_args()

    report = measure_runtime(args)
    rendered = json.dumps(report, indent=2)
    Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
