"""Tool implementations for the LangGraph agent.
Each tool returns plain text so ToolNode can feed results back into the graph.
"""

import os
import re
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


def _offline_grammar_response(question: str) -> str:
    """Provide a small offline fallback for common grammar prompts.
    This keeps `agent/run.py` runnable without an OpenAI API key.
    """
    lowered = question.lower()
    if "ser" in lowered and "estar" in lowered:
        return (
            "Spanish uses 'ser' for identity, origin, and lasting characteristics, "
            "while 'estar' is used for temporary states, conditions, and locations."
        )
    return (
        "Grammar explanations require OPENAI_API_KEY for full detail. "
        "The routing worked, but the repo is currently in offline fallback mode."
    )


def _extract_spanish_word(question: str) -> str:
    """Extract a likely Spanish target word from a vocabulary question.
    Falls back to the raw question when no quoted or isolated word is found.
    """
    quoted_match = re.search(r"['\"]([^'\"]+)['\"]", question)
    if quoted_match:
        return quoted_match.group(1).strip().lower()

    word_match = re.search(
        r"\bwhat does\s+([a-záéíóúñü]+)\s+mean\b",
        question.lower(),
    )
    if word_match:
        return word_match.group(1).strip().lower()

    return question.strip().lower()


def _offline_word_info_response(question: str) -> str:
    """Provide a small offline fallback for common vocabulary prompts.
    This avoids blocking the routing test when no OpenAI key is configured.
    """
    glossary = {
        "biblioteca": (
            "'biblioteca' means 'library' in English. "
            "It is a feminine noun in Spanish: 'la biblioteca'."
        ),
    }
    word = _extract_spanish_word(question)
    if word in glossary:
        return glossary[word]
    return (
        "Vocabulary explanations require OPENAI_API_KEY for full detail. "
        "The routing worked, but the repo is currently in offline fallback mode."
    )


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
def explain_spanish_grammar(question: str) -> str:
    """Explain a Spanish grammar concept with GPT-4o-mini.
    Falls back to a deterministic offline answer when no API key is configured.
    """
    system_prompt = (
        "You explain Spanish grammar clearly and concisely for learners. "
        "Answer in 3 to 5 sentences with one concrete example."
    )
    response_text = _chat_with_openai(system_prompt, question)
    if response_text:
        return response_text
    return _offline_grammar_response(question)


@tool
def get_spanish_word_info(question: str) -> str:
    """Explain the meaning and usage of a Spanish word with GPT-4o-mini.
    Falls back to a small built-in glossary when no API key is configured.
    """
    system_prompt = (
        "You explain Spanish vocabulary clearly for English speakers. "
        "Give the English meaning, part of speech, and a short example sentence."
    )
    response_text = _chat_with_openai(system_prompt, question)
    if response_text:
        return response_text
    return _offline_word_info_response(question)


@tool
def rag_translate(text: str) -> str:
    """Translate with translation-memory context retrieved from ChromaDB.
    Retrieved bilingual examples are included in the tool output for traceability.
    """
    retrieved_pairs = retrieve_similar_translations(text, k=3)
    formatted_context = _format_retrieved_context(retrieved_pairs)

    system_prompt = (
        "You translate English into Spanish for parliamentary and institutional text. "
        "Use the retrieved bilingual examples to keep terminology consistent. "
        "Return only the final Spanish translation."
    )
    user_prompt = (
        f"English text: {text}\n\n"
        f"Retrieved translation memory:\n{formatted_context}"
    )
    response_text = _chat_with_openai(system_prompt, user_prompt)

    if not response_text:
        response_text = translate_with_custom_model.invoke({"text": text})

    return (
        f"Translation: {response_text}\n"
        f"Retrieved context:\n{formatted_context}"
    )


TOOLS = [
    translate_with_custom_model,
    rag_translate,
    explain_spanish_grammar,
    get_spanish_word_info,
]
