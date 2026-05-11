import os

from src.env import load_local_env, parse_env_line


def test_parse_env_line_ignores_comments_and_malformed_lines():
    assert parse_env_line("# comment") is None
    assert parse_env_line("missing_equals") is None
    assert parse_env_line("BAD KEY=value") is None


def test_load_local_env_preserves_existing_values(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "TRANSLATOR_API_BASE_URL=http://from-file.test",
                "OPENAI_API_KEY=",
                "NEW_VALUE='loaded'",
                "malformed",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRANSLATOR_API_BASE_URL", "http://existing.test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NEW_VALUE", raising=False)

    loaded = load_local_env(env_path)

    assert os.environ["TRANSLATOR_API_BASE_URL"] == "http://existing.test"
    assert "OPENAI_API_KEY" not in os.environ
    assert os.environ["NEW_VALUE"] == "loaded"
    assert loaded == ["NEW_VALUE"]
