# LangGraph Routing Report

Updated: 2026-04-09

## Overview

This report documents the focused routing layer used in the English-to-Spanish translation project.

The routing layer now supports one clear purpose:

- route ordinary English-to-Spanish requests to the custom translation model
- route institutional translation requests to the RAG-backed review path

The graph shape remains fixed:

`START -> agent -> tools -> agent -> END`

## Implemented Files

| Path | Purpose |
| --- | --- |
| `agent/__init__.py` | package marker |
| `agent/tools.py` | focused translation and review tools |
| `agent/graph.py` | LangGraph state, routing logic, and compiled graph |
| `agent/run.py` | smoke test for focused routing |

## Tool Layer

The focused agent exposes two tools.

### 1. `translate_with_custom_model`

Purpose:

- handle ordinary English-to-Spanish translation requests

Behavior:

- calls the local FastAPI `POST /translate` endpoint
- returns the direct translation from the custom model

### 2. `rag_translate`

Purpose:

- handle parliamentary and institutional translation requests

Behavior:

1. call the custom model first for the draft translation
2. retrieve the top 3 similar Europarl bilingual examples from ChromaDB
3. send the source, draft, and retrieved context to GPT-4o-mini
4. return the decision, the draft, the final translation, and the retrieved context

This is the key focused interaction in the project because GPT is no longer a side feature. It depends on the custom model draft to do useful work.

## Routing Logic

The agent uses:

- `translate_with_custom_model` for normal translation requests
- `rag_translate` for institutional phrases such as:
  - `parliament`
  - `parliamentary`
  - `session`
  - `adjourned`
  - `motion`
  - `committee`
  - `council`
  - `amendment`

The conditional edge logic was not changed:

- if the latest model message has non-empty `tool_calls`, route to `tools`
- otherwise route to `END`

## OpenAI Integration

The implementation uses the OpenAI Python SDK directly.

Current online path:

- `OpenAI(api_key=..., timeout=30.0, max_retries=2)`
- `client.chat.completions.create(...)`
- model: `gpt-4o-mini`

Current offline path:

- direct translation still works
- `rag_translate` still returns the custom-model draft and retrieved context
- the decision becomes `KEEP (offline fallback)` when GPT is unavailable

## Focused Smoke Test

Verification command:

```bash
venv/bin/python agent/run.py
```

The smoke test now checks only the focused translation routes:

| Query | Expected tool |
| --- | --- |
| `Translate 'I need a doctor' to Spanish` | `translate_with_custom_model` |
| `How do you say 'the train is late'?` | `translate_with_custom_model` |
| `Translate 'The parliamentary session was adjourned.' to Spanish` | `rag_translate` |
| `Translate 'The committee approved the amendment.' to Spanish` | `rag_translate` |

## Verified Hybrid Result

Observed `rag_translate` output after OpenAI billing was enabled:

```text
Decision: EDIT
Custom model draft: Se suspendió la sesión parlamentaria.
Translation: Se interrumpe la sesión parlamentaria.
```

This is the important project result because it proves:

- the custom model drafts
- retrieval adds similar domain examples
- GPT reviews and improves the draft

## Honest Assessment

The focused routing layer is now stronger than the earlier broad routing version.

Why:

- it is tied to one application instead of many side features
- it supports the core translation story directly
- it strengthens the connection between the custom model and GPT instead of diluting it

This is still lightweight orchestration, not a long-horizon autonomous system. In this project, its job is routing and review, not broad automation.
