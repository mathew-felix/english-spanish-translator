import subprocess

import pytest


def test_no_large_generated_artifacts_are_tracked():
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("git is unavailable")

    tracked_paths = [
        path for path in result.stdout.decode("utf-8").split("\0") if path
    ]
    forbidden_suffixes = (".pth", ".pt", ".ckpt", ".onnx", ".bin")
    tracked_artifacts = [
        path for path in tracked_paths if path.lower().endswith(forbidden_suffixes)
    ]

    assert tracked_artifacts == []


def test_tracked_tree_has_no_obvious_live_secret_patterns():
    result = subprocess.run(
        [
            "git",
            "grep",
            "-nI",
            "-E",
            "wandb_v1_[A-Za-z0-9]+|sk-[A-Za-z0-9]{20,}|WANDB_API_KEY[[:space:]]*=[[:space:]]*wandb",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 128:
        pytest.skip("git is unavailable")

    assert result.returncode == 1, result.stdout
