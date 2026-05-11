"""Small `.env` loader for local development.

The project only needs simple KEY=VALUE parsing. Existing process environment
values always win so CI and deployment settings cannot be overwritten by a file.
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[1]


def parse_env_line(raw_line: str) -> tuple[str, str] | None:
    """Parse one `.env` line.

    Blank lines, comments, malformed keys, and lines without `=` are ignored.
    """
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None

    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip().strip("\"'")

    if not key or any(character.isspace() for character in key):
        return None
    return key, value


def load_local_env(env_path: str | os.PathLike[str] | None = None) -> list[str]:
    """Load local `.env` values without overriding existing environment values.

    Returns the keys loaded from the file. Empty values are treated as
    documentation placeholders and are not added to the environment.
    """
    path = Path(env_path) if env_path is not None else project_root() / ".env"
    loaded_keys: list[str] = []

    if not path.is_file():
        return loaded_keys

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parsed = parse_env_line(raw_line)
        if parsed is None:
            continue

        key, value = parsed
        if key not in os.environ and value:
            os.environ[key] = value
            loaded_keys.append(key)

    return loaded_keys
