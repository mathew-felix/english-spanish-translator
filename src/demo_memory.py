"""Curated bilingual evidence for lightweight hosted demos.

This fixture is deliberately tiny and deterministic. It is not a replacement for
the generated ChromaDB translation memory; it lets public demos show the review
workflow when a full RAG index is unavailable.
"""

from __future__ import annotations

from typing import Any

DEMO_TRANSLATION_MEMORY = [
    {
        "english": "The parliamentary session was adjourned.",
        "spanish": "Se suspendio la sesion parlamentaria.",
        "keywords": {"parliamentary", "session", "adjourned"},
    },
    {
        "english": "The committee approved the amendment.",
        "spanish": "El comite aprobo la enmienda.",
        "keywords": {"committee", "approved", "amendment"},
    },
    {
        "english": "The council voted on the motion.",
        "spanish": "El Consejo voto sobre la mocion.",
        "keywords": {"council", "voted", "motion"},
    },
    {
        "english": "The agency published the legal notice.",
        "spanish": "La agencia publico el aviso legal.",
        "keywords": {"agency", "published", "legal", "notice"},
    },
    {
        "english": "The policy report was submitted for review.",
        "spanish": "El informe de politica se presento para revision.",
        "keywords": {"policy", "report", "submitted", "review"},
    },
]


def retrieve_demo_translations(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Return keyword-ranked curated examples for public demos."""
    cleaned_tokens = {
        token.strip(".,;:!?()[]{}\"'").lower()
        for token in query.split()
        if token.strip(".,;:!?()[]{}\"'")
    }
    scored_rows = []
    for row in DEMO_TRANSLATION_MEMORY:
        overlap = cleaned_tokens.intersection(row["keywords"])
        if not overlap:
            continue
        distance = round(1.0 / (len(overlap) + 1), 6)
        scored_rows.append(
            (
                -len(overlap),
                distance,
                {
                    "english": row["english"],
                    "spanish": row["spanish"],
                    "corpus": "curated_demo_fixture",
                    "source_index": None,
                    "distance": distance,
                },
            )
        )

    scored_rows.sort(key=lambda item: (item[0], item[1]))
    return [row for _, _, row in scored_rows[:k]]
