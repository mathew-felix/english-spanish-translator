"""Preflight checks for local API/demo startup."""

from __future__ import annotations

import argparse
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import requests
from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_repo_on_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _load_local_env() -> None:
    _ensure_repo_on_path()
    from src.env import load_local_env

    load_local_env()


def _build_release_url(owner: str, repo: str, tag: str) -> str:
    _ensure_repo_on_path()
    from scripts.download_model import _build_release_api_url

    return _build_release_api_url(owner=owner, repo=repo, tag=tag)


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    message: str

    @property
    def failed(self) -> bool:
        return self.status == "FAIL"


def _demo_mode_enabled() -> bool:
    return os.getenv("TRANSLATOR_DEMO_MODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "demo",
    }


def check_python_version() -> CheckResult:
    version = platform.python_version()
    if sys.version_info >= (3, 12):
        return CheckResult("python", "PASS", f"Python {version}")
    return CheckResult("python", "FAIL", f"Python 3.12+ required; found {version}")


def check_model_artifacts() -> list[CheckResult]:
    root = REPO_ROOT
    model_path = root / "best_model.pth"
    tokenizer_path = root / "data" / "tokenizer"

    if _demo_mode_enabled():
        return [
            CheckResult(
                "artifacts",
                "WARN",
                "TRANSLATOR_DEMO_MODE is enabled; checkpoint/tokenizer are optional.",
            )
        ]

    results = []
    results.append(
        CheckResult(
            "checkpoint",
            "PASS" if model_path.is_file() else "FAIL",
            f"{model_path} {'found' if model_path.is_file() else 'missing'}",
        )
    )
    results.append(
        CheckResult(
            "tokenizer",
            "PASS" if tokenizer_path.is_dir() else "FAIL",
            f"{tokenizer_path} {'found' if tokenizer_path.is_dir() else 'missing'}",
        )
    )
    return results


def check_environment() -> list[CheckResult]:
    _load_local_env()
    wandb_configured = bool(os.getenv("WANDB_API_KEY")) or os.getenv("WANDB_MODE") == "offline"
    results = [
        CheckResult(
            "openai",
            "PASS" if os.getenv("OPENAI_API_KEY") else "WARN",
            "OPENAI_API_KEY configured"
            if os.getenv("OPENAI_API_KEY")
            else "OPENAI_API_KEY not set; GPT review will be skipped.",
        ),
        CheckResult(
            "wandb",
            "PASS" if wandb_configured else "WARN",
            "W&B configured for online or offline use."
            if wandb_configured
            else "W&B is not configured; set WANDB_MODE=offline for training runs.",
        ),
    ]
    return results


def check_release_connectivity(timeout: float = 10.0) -> CheckResult:
    artifact_source = os.getenv("MODEL_ARTIFACT_SOURCE", "github").strip().lower()
    if artifact_source == "huggingface":
        repo_id = os.getenv("HF_MODEL_REPO_ID", "mathew-felix/en-es-nmt-transformer")
        revision = os.getenv("HF_MODEL_REVISION", "main")
        try:
            HfApi().model_info(repo_id=repo_id, revision=revision, timeout=timeout)
        except Exception as exc:
            return CheckResult(
                "model_repo",
                "FAIL",
                f"Hugging Face model lookup failed with {type(exc).__name__}: {repo_id}@{revision}",
            )
        return CheckResult(
            "model_repo",
            "PASS",
            f"Hugging Face model repo reachable: {repo_id}@{revision}",
        )

    owner = os.getenv("MODEL_RELEASE_OWNER", "mathew-felix")
    repo = os.getenv("MODEL_RELEASE_REPO", "english-spanish-translator")
    tag = os.getenv("MODEL_RELEASE_TAG", "eng-sp-tranlate")
    url = _build_release_url(owner=owner, repo=repo, tag=tag)

    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        return CheckResult(
            "release",
            "FAIL",
            f"GitHub release lookup failed with {type(exc).__name__}: {url}",
        )

    if response.status_code == 200:
        return CheckResult("release", "PASS", f"Release metadata reachable: {url}")
    return CheckResult(
        "release",
        "FAIL",
        f"Release metadata returned HTTP {response.status_code}: {url}",
    )


def run_preflight(skip_network: bool = False) -> list[CheckResult]:
    _load_local_env()
    results = [check_python_version()]
    results.extend(check_model_artifacts())
    results.extend(check_environment())
    if skip_network:
        results.append(CheckResult("release", "WARN", "Release connectivity check skipped."))
    else:
        results.append(check_release_connectivity())
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local translator runtime readiness.")
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Skip GitHub release connectivity validation.",
    )
    args = parser.parse_args()

    results = run_preflight(skip_network=args.skip_network)
    for result in results:
        print(f"{result.status:4} {result.name}: {result.message}")
    return 1 if any(result.failed for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
