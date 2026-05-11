"""Benchmark local translation API endpoints and write JSON results."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

DEFAULT_TEXTS = {
    "translate": "The parliamentary session was adjourned.",
    "institutional-review": "The committee approved the amendment.",
}


def percentile(values: list[float], p: float) -> float:
    """Return the percentile using linear interpolation."""
    if not values:
        raise ValueError("values must not be empty")
    if p < 0 or p > 100:
        raise ValueError("p must be between 0 and 100")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * (p / 100)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_latencies(latencies_ms: list[float], elapsed_seconds: float) -> dict[str, Any]:
    """Build the benchmark summary used by tests and CLI output."""
    if not latencies_ms:
        raise ValueError("latencies_ms must not be empty")

    return {
        "count": len(latencies_ms),
        "avg_ms": round(statistics.fmean(latencies_ms), 2),
        "p50_ms": round(percentile(latencies_ms, 50), 2),
        "p95_ms": round(percentile(latencies_ms, 95), 2),
        "min_ms": round(min(latencies_ms), 2),
        "max_ms": round(max(latencies_ms), 2),
        "throughput_rps": round(len(latencies_ms) / elapsed_seconds, 3)
        if elapsed_seconds > 0
        else None,
    }


def benchmark_endpoint(
    *,
    base_url: str,
    endpoint: str,
    text: str,
    iterations: int,
    timeout: float,
) -> dict[str, Any]:
    """Measure one API endpoint with sequential requests."""
    url = f"{base_url.rstrip('/')}/{endpoint}"
    latencies_ms = []
    started_at = time.perf_counter()

    for _ in range(iterations):
        request_started_at = time.perf_counter()
        response = requests.post(url, json={"text": text}, timeout=timeout)
        response.raise_for_status()
        latencies_ms.append(round((time.perf_counter() - request_started_at) * 1000, 2))

    elapsed_seconds = time.perf_counter() - started_at
    return {
        "endpoint": f"/{endpoint}",
        "text": text,
        "summary": summarize_latencies(latencies_ms, elapsed_seconds),
        "latencies_ms": latencies_ms,
    }


def build_benchmark_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Attach reproducibility metadata to endpoint benchmark results."""
    root = Path.cwd()
    model_path = root / "best_model.pth"
    tokenizer_path = root / "data" / "tokenizer"
    tokenizer_size_bytes = None
    if tokenizer_path.is_dir():
        tokenizer_size_bytes = sum(
            path.stat().st_size for path in tokenizer_path.rglob("*") if path.is_file()
        )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "dependency_mode": os.getenv("TRANSLATOR_DEPENDENCY_MODE", "unspecified"),
        "demo_mode": os.getenv("TRANSLATOR_DEMO_MODE", "0"),
        "artifacts": {
            "model_size_bytes": model_path.stat().st_size if model_path.is_file() else None,
            "tokenizer_size_bytes": tokenizer_size_bytes,
        },
        "metrics_not_measured": [
            "cold_start_time",
            "peak_ram",
            "peak_vram",
        ],
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark local translator API endpoints.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--endpoint",
        choices=["translate", "institutional-review", "both"],
        default="both",
    )
    parser.add_argument("--text", default=None)
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()

    endpoints = (
        ["translate", "institutional-review"]
        if args.endpoint == "both"
        else [args.endpoint]
    )
    results = []
    for endpoint in endpoints:
        text = args.text or DEFAULT_TEXTS[endpoint]
        results.append(
            benchmark_endpoint(
                base_url=args.base_url,
                endpoint=endpoint,
                text=text,
                iterations=args.iterations,
                timeout=args.timeout,
            )
        )

    report = build_benchmark_report(results)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
