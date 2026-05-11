import subprocess
import sys


def test_run_help_exits_without_loading_pipeline_modules():
    result = subprocess.run(
        [sys.executable, "-m", "src.run", "--help"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0
    assert "--step" in result.stdout
