import pytest

from src import review


def test_institutional_review_keeps_draft_when_index_is_missing(monkeypatch):
    monkeypatch.delenv("TRANSLATOR_DEMO_MEMORY", raising=False)
    monkeypatch.setattr(review, "_translation_memory_dir_has_files", lambda: False)

    result = review.build_institutional_review(
        "The parliamentary session was adjourned.",
        translator=lambda text: "Se aplazó la sesión parlamentaria.",
    )

    assert result["draft_translation"] == "Se aplazó la sesión parlamentaria."
    assert result["final_translation"] == result["draft_translation"]
    assert result["context_status"] == "index_missing"
    assert result["reviewer_status"] == "gpt_skipped_no_context"
    assert result["retrieved_pairs"] == []


def test_institutional_review_uses_demo_memory_when_enabled(monkeypatch):
    monkeypatch.setenv("TRANSLATOR_DEMO_MEMORY", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(review, "load_local_env", lambda: [])
    monkeypatch.setattr(review, "_translation_memory_dir_has_files", lambda: False)

    result = review.build_institutional_review(
        "The parliamentary session was adjourned.",
        translator=lambda text: "Se suspendio la sesion parlamentaria.",
    )

    assert result["context_status"] == "demo_fixture"
    assert result["retrieved_pairs"][0]["corpus"] == "curated_demo_fixture"
    assert result["reviewer_status"] == "gpt_not_configured"


def test_institutional_review_keeps_draft_when_openai_is_not_configured(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(review, "load_local_env", lambda: [])

    def fake_retriever(text, k):
        return [
            {
                "english": "The session is adjourned.",
                "spanish": "Se aplaza la sesión.",
                "distance": 0.2,
            }
        ]

    result = review.build_institutional_review(
        "The parliamentary session was adjourned.",
        translator=lambda text: "Se aplazó la sesión parlamentaria.",
        retriever=fake_retriever,
    )

    assert result["context_status"] == "available"
    assert result["reviewer_status"] == "gpt_not_configured"
    assert result["decision"] == "KEEP"
    assert result["final_translation"] == "Se aplazó la sesión parlamentaria."


def test_parse_revision_response_infers_edit_when_decision_is_missing():
    decision, translation = review._parse_revision_response(
        "TRANSLATION: Se interrumpe la sesión parlamentaria.",
        "Se aplazó la sesión parlamentaria.",
    )

    assert decision == "EDIT"
    assert translation == "Se interrumpe la sesión parlamentaria."


def test_institutional_review_rejects_blank_text():
    with pytest.raises(ValueError, match="Text must not be empty"):
        review.build_institutional_review("   ", translator=lambda text: "x")
