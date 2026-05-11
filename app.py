"""Hugging Face Spaces entry point for the institutional review demo.

The hosted demo defaults to checkpoint-free mode because the full checkpoint is
large enough to make free CPU Spaces cold starts unpredictable. Set
`TRANSLATOR_DEMO_MODE=0` only on a Space with model artifacts available.
"""

from __future__ import annotations

import os
import time
from typing import Any

import gradio as gr

os.environ.setdefault("TRANSLATOR_DEMO_MODE", "1")
os.environ.setdefault("TRANSLATOR_DEMO_MEMORY", "1")


def _truthy_env(name: str, default: str = "") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _demo_mode_enabled() -> bool:
    return os.getenv("TRANSLATOR_DEMO_MODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "demo",
    }


def _ensure_full_model_artifacts() -> None:
    if _demo_mode_enabled() or not _truthy_env("TRANSLATOR_AUTO_DOWNLOAD_MODEL", "0"):
        return

    import sys

    from scripts.download_model import main as download_model_main

    original_argv = sys.argv[:]
    sys.argv = [
        "download_model.py",
        "--source",
        os.getenv("MODEL_ARTIFACT_SOURCE", "huggingface"),
    ]
    try:
        download_model_main()
    finally:
        sys.argv = original_argv


_ensure_full_model_artifacts()

from src.review import build_institutional_review  # noqa: E402

EXAMPLES = [
    "The parliamentary session was adjourned.",
    "The committee approved the amendment.",
    "The council voted on the motion.",
    "The agency published the legal notice.",
    "The policy report was submitted for review.",
]


def _format_examples(pairs: list[dict[str, Any]]) -> str:
    if not pairs:
        return "No retrieved bilingual examples were available."

    lines = []
    for index, pair in enumerate(pairs, start=1):
        lines.append(
            f"{index}. EN: {pair['english']}\n"
            f"   ES: {pair['spanish']}\n"
            f"   distance: {pair['distance']}"
        )
    return "\n\n".join(lines)


def review_text(text: str) -> tuple[str, str, str, str, str, str]:
    cleaned_text = text.strip()
    if not cleaned_text:
        raise gr.Error("Enter an English institutional sentence.")

    started_at = time.perf_counter()
    result = build_institutional_review(cleaned_text)
    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
    mode = "demo samples" if os.getenv("TRANSLATOR_DEMO_MODE") == "1" else "full model"

    status = (
        f"Mode: {mode}\n"
        f"Context: {result['context_status']} - {result['context_message']}\n"
        f"Reviewer: {result['reviewer_status']} - {result['reviewer_explanation']}\n"
        f"Latency: {latency_ms} ms"
    )
    if mode == "full model":
        transparency_notice = (
            "Hosted demo note: this Space is running full-model inference. "
            "Retrieved examples may come from a small curated demo-memory "
            "fixture, not a full production translation-memory index."
        )
    else:
        transparency_notice = (
            "Hosted demo note: sample mode uses deterministic demo translations, "
            "not live full-checkpoint inference. Enable the full model only when "
            "artifacts and storage are configured."
        )

    return (
        result["draft_translation"],
        result["decision"],
        result["final_translation"],
        _format_examples(result["retrieved_pairs"]),
        status,
        transparency_notice,
    )


with gr.Blocks(title="Institutional Translation Review") as demo:
    gr.Markdown("# Institutional Translation Review")
    gr.Markdown(
        "Inspect a custom English-to-Spanish draft, optional bilingual evidence, "
        "review status, and final wording. The hosted Space can run the full "
        "model and may use a curated demo-memory fixture when no full index is "
        "available."
    )

    source = gr.Textbox(
        label="English institutional source",
        value=EXAMPLES[0],
        lines=3,
        max_lines=6,
    )
    run_button = gr.Button("Review Translation", variant="primary")

    with gr.Row():
        draft = gr.Textbox(label="Custom draft", lines=3)
        final = gr.Textbox(label="Final reviewed translation", lines=3)

    decision = gr.Textbox(label="Review decision")
    evidence = gr.Textbox(label="Retrieved bilingual evidence", lines=8)
    status = gr.Textbox(label="Runtime and fallback status", lines=5)
    notice = gr.Textbox(label="Hosted demo transparency", lines=3)

    gr.Examples(examples=EXAMPLES, inputs=source)
    run_button.click(
        review_text,
        inputs=source,
        outputs=[draft, decision, final, evidence, status, notice],
    )


if __name__ == "__main__":
    demo.launch()
