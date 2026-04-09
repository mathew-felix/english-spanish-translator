"""Tool implementations for the LangGraph agent.
Each tool returns plain text so ToolNode can feed results back into the graph.
"""

import os
from typing import Optional

import requests
from langchain_core.tools import tool
from openai import OpenAI

from rag.retriever import retrieve_similar_translations

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


def load_local_env() -> None:
    """Load `.env` values into the process environment if present.
    Existing environment variables take precedence over file values.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(repo_root, ".env")

    if not os.path.isfile(env_path):
        return

    with open(env_path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ and value:
                os.environ[key] = value


def get_api_base_url() -> str:
    """Return the FastAPI base URL used by the translation tool.
    The value is configurable through `TRANSLATOR_API_BASE_URL`.
    """
    load_local_env()
    return os.getenv("TRANSLATOR_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def _build_openai_client() -> Optional[OpenAI]:
    """Create an OpenAI client when an API key is available.
    Returns `None` so the agent can fall back to offline behavior.
    """
    load_local_env()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAI(api_key=api_key, timeout=30.0, max_retries=2)


def _chat_with_openai(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Run one small GPT-4o-mini call for tool-generated text.
    Returns `None` when no API key is available for online inference.
    """
    client = _build_openai_client()
    if client is None:
        return None

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    return (content or "").strip()


def _format_retrieved_context(retrieved_pairs: list[dict]) -> str:
    """Format retrieved bilingual examples for logging and GPT prompting.
    The format is intentionally compact so the tool output stays readable.
    """
    lines = []
    for index, pair in enumerate(retrieved_pairs, start=1):
        lines.append(
            f"{index}. EN: {pair['english']} | ES: {pair['spanish']} | distance={pair['distance']}"
        )
    return "\n".join(lines)


def _parse_revision_response(response_text: str, draft_translation: str) -> tuple[str, str]:
    """Parse the GPT review response into a decision and final translation.
    Falls back to the custom-model draft when the response format is incomplete.
    """
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


def _review_translation_with_context(
    text: str,
    draft_translation: str,
    formatted_context: str,
) -> tuple[str, str]:
    """Review a custom-model draft with GPT-4o-mini and retrieved memory context.
    The fallback path keeps the draft unchanged when no OpenAI key is configured.
    """
    system_prompt = (
        "You review a custom English-to-Spanish machine translation draft for "
        "parliamentary and institutional text. Use the retrieved bilingual examples "
        "to preserve terminology. If the draft is already correct and consistent, "
        "keep it. Otherwise make the smallest necessary edit. Return exactly two "
        "lines in this format:\n"
        "DECISION: KEEP or EDIT\n"
        "TRANSLATION: <final Spanish translation>"
    )
    user_prompt = (
        f"English source: {text}\n"
        f"Custom model draft: {draft_translation}\n\n"
        f"Retrieved translation memory:\n{formatted_context}"
    )

    response_text = _chat_with_openai(system_prompt, user_prompt)
    if not response_text:
        return "KEEP (offline fallback)", draft_translation
    return _parse_revision_response(response_text, draft_translation)


def build_rag_translation_review(text: str) -> dict:
    """Build a structured institutional-translation review result.
    The custom model drafts first, then retrieval and GPT review refine that draft.
    """
    draft_translation = translate_with_custom_model.invoke({"text": text})
    retrieved_pairs = retrieve_similar_translations(text, k=3)
    formatted_context = _format_retrieved_context(retrieved_pairs)
    decision, final_translation = _review_translation_with_context(
        text=text,
        draft_translation=draft_translation,
        formatted_context=formatted_context,
    )
    return {
        "input": text,
        "draft_translation": draft_translation,
        "decision": decision,
        "final_translation": final_translation,
        "retrieved_pairs": retrieved_pairs,
        "formatted_context": formatted_context,
    }


@tool
def translate_with_custom_model(text: str) -> str:
    """Translate one English sentence through the FastAPI service.
    The tool expects the translation API to be reachable on the configured base URL.
    """
    api_url = f"{get_api_base_url()}/translate"
    try:
        response = requests.post(
            api_url,
            json={"text": text},
            timeout=10,
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        raise RuntimeError("Translation API request timed out.") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Translation API request failed: {exc}") from exc

    payload = response.json()
    translation = payload.get("translation", "").strip()
    if not translation:
        raise RuntimeError("Translation API returned an empty translation.")
    return translation


@tool
def rag_translate(text: str) -> str:
    """Translate with translation-memory context retrieved from ChromaDB.
    Retrieved bilingual examples are included in the tool output for traceability.
    """
    review_result = build_rag_translation_review(text)

    return (
        f"Decision: {review_result['decision']}\n"
        f"Custom model draft: {review_result['draft_translation']}\n"
        f"Translation: {review_result['final_translation']}\n"
        f"Retrieved context:\n{review_result['formatted_context']}"
    )


TOOLS = [
    translate_with_custom_model,
    rag_translate,
]
