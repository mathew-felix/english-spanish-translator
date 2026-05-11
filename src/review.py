"""Core institutional translation-review service.

This module deliberately avoids importing ChromaDB, sentence transformers, or
OpenAI at import time. The FastAPI path can run with only the custom model and
add optional retrieval/GPT behavior when those pieces are installed and
configured.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from src.env import load_local_env, project_root

DEFAULT_TOP_K = 3

Translator = Callable[[str], str]
Retriever = Callable[[str, int], list[dict[str, Any]]]


def _translation_memory_dir_has_files() -> bool:
    """Return whether a local generated Chroma index appears to exist."""
    index_path = project_root() / "rag" / "chroma_db"
    if not index_path.is_dir():
        return False
    return any(index_path.iterdir())


def _demo_memory_enabled() -> bool:
    """Return whether curated demo evidence should be used without ChromaDB."""
    return os.getenv("TRANSLATOR_DEMO_MEMORY", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "demo",
    }


def _format_retrieved_context(retrieved_pairs: list[dict[str, Any]]) -> str:
    """Format bilingual evidence for GPT prompting and UI output."""
    lines = []
    for index, pair in enumerate(retrieved_pairs, start=1):
        lines.append(
            f"{index}. EN: {pair['english']} | ES: {pair['spanish']} | "
            f"distance={pair['distance']}"
        )
    return "\n".join(lines)


def _parse_revision_response(response_text: str, draft_translation: str) -> tuple[str, str]:
    """Parse the GPT review response into a decision and final translation."""
    cleaned_response = response_text.strip()
    if cleaned_response.startswith("```"):
        cleaned_lines = [
            line
            for line in cleaned_response.splitlines()
            if not line.strip().startswith("```")
        ]
        cleaned_response = "\n".join(cleaned_lines).strip()

    decision = ""
    final_translation = ""
    for line in cleaned_response.splitlines():
        stripped_line = line.strip()
        lowered_line = stripped_line.lower()
        if lowered_line.startswith("decision:"):
            decision = stripped_line.split(":", 1)[1].strip().upper()
        elif lowered_line.startswith("translation:"):
            final_translation = stripped_line.split(":", 1)[1].strip()

    if not final_translation:
        final_translation = draft_translation
    if decision not in {"KEEP", "EDIT"}:
        decision = "KEEP" if final_translation == draft_translation else "EDIT"
    return decision, final_translation


def _translate_with_custom_model(text: str) -> str:
    """Call the cached local model only when the review service needs it."""
    from source.inference import translate

    return translate(text)


def _retrieve_review_context(
    text: str,
    k: int = DEFAULT_TOP_K,
    retriever: Retriever | None = None,
) -> tuple[list[dict[str, Any]], str, str]:
    """Retrieve institutional bilingual examples when an index is available."""
    if retriever is not None:
        retrieved_pairs = retriever(text, k)
        if retrieved_pairs:
            return (
                retrieved_pairs,
                "available",
                "Retrieved bilingual examples from the configured retriever.",
            )
        return (
            [],
            "empty",
            "The configured retriever returned no bilingual examples.",
        )

    if not _translation_memory_dir_has_files():
        if _demo_memory_enabled():
            from src.demo_memory import retrieve_demo_translations

            retrieved_pairs = retrieve_demo_translations(text, k=k)
            if retrieved_pairs:
                return (
                    retrieved_pairs,
                    "demo_fixture",
                    "Retrieved curated bilingual examples from the demo fixture.",
                )
        return (
            [],
            "index_missing",
            "No local translation-memory index was found. Build one with "
            "`python rag/build_index.py` or run in demo mode.",
        )

    try:
        from rag.retriever import retrieve_similar_translations
    except ImportError:
        return (
            [],
            "rag_dependency_missing",
            "RAG optional dependencies are not installed. Install "
            "`requirements-rag.txt` to enable live retrieval.",
        )

    retrieved_pairs = retrieve_similar_translations(text, k=k)
    if not retrieved_pairs:
        return (
            [],
            "empty",
            "The translation memory returned no matching bilingual examples.",
        )

    return (
        retrieved_pairs,
        "available",
        "Retrieved bilingual examples from the local translation memory.",
    )


def _chat_with_openai(system_prompt: str, user_prompt: str) -> tuple[str | None, str, str]:
    """Run an optional GPT review call and return fallback status details."""
    load_local_env()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return (
            None,
            "gpt_not_configured",
            "OPENAI_API_KEY is not configured; kept the custom model draft.",
        )

    try:
        from openai import OpenAI
    except ImportError:
        return (
            None,
            "gpt_dependency_missing",
            "The OpenAI package is not installed; kept the custom model draft.",
        )

    try:
        client = OpenAI(api_key=api_key, timeout=30.0, max_retries=2)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:
        return (
            None,
            "gpt_error",
            f"GPT review failed with {type(exc).__name__}; kept the custom model draft.",
        )

    content = response.choices[0].message.content
    return (content or "").strip(), "gpt_reviewed", "GPT reviewed the draft with retrieved context."


def _review_translation_with_context(
    text: str,
    draft_translation: str,
    formatted_context: str,
) -> tuple[str, str, str, str]:
    """Review a draft only when retrieved bilingual context is available."""
    if not formatted_context.strip():
        return (
            "KEEP",
            draft_translation,
            "gpt_skipped_no_context",
            "No retrieved examples were available, so GPT review was skipped.",
        )

    system_prompt = (
        "You review a custom English-to-Spanish machine translation draft for "
        "institutional, legal, policy, and parliamentary text. Use the retrieved "
        "bilingual examples to preserve terminology and phrasing evidence. If "
        "the draft is already correct and consistent, keep it. Otherwise make "
        "the smallest necessary edit. Return exactly two lines:\n"
        "DECISION: KEEP or EDIT\n"
        "TRANSLATION: <final Spanish translation>"
    )
    user_prompt = (
        f"English source: {text}\n"
        f"Custom model draft: {draft_translation}\n\n"
        f"Retrieved bilingual evidence:\n{formatted_context}"
    )

    response_text, status, explanation = _chat_with_openai(system_prompt, user_prompt)
    if not response_text:
        return "KEEP", draft_translation, status, explanation

    decision, final_translation = _parse_revision_response(
        response_text=response_text,
        draft_translation=draft_translation,
    )
    return decision, final_translation, status, explanation


def build_institutional_review(
    text: str,
    *,
    translator: Translator | None = None,
    retriever: Retriever | None = None,
    k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Build a staged institutional translation-review result.

    The no-GPT/no-RAG path is intentional: reviewers still see the custom draft
    and a clear reason why evidence or GPT assistance was not used.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Text must not be empty.")

    translate_fn = translator or _translate_with_custom_model
    draft_translation = translate_fn(cleaned_text)
    retrieved_pairs, context_status, context_message = _retrieve_review_context(
        cleaned_text,
        k=k,
        retriever=retriever,
    )
    formatted_context = _format_retrieved_context(retrieved_pairs)
    decision, final_translation, reviewer_status, reviewer_explanation = (
        _review_translation_with_context(
            text=cleaned_text,
            draft_translation=draft_translation,
            formatted_context=formatted_context,
        )
    )

    return {
        "input": cleaned_text,
        "draft_translation": draft_translation,
        "decision": decision,
        "final_translation": final_translation,
        "retrieved_pairs": retrieved_pairs,
        "formatted_context": formatted_context,
        "context_status": context_status,
        "context_message": context_message,
        "reviewer_status": reviewer_status,
        "reviewer_explanation": reviewer_explanation,
    }
