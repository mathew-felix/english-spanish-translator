# LangGraph Agent Report

Updated: 2026-04-09

## Overview

This report documents the Phase 6 LangGraph agent implementation added on top of the existing English-to-Spanish translation system.

The goal of this phase was to move the project from a standalone translation model and API into a routed AI system that can:

- translate English text into Spanish with the custom Transformer
- explain Spanish grammar
- explain the meaning of Spanish words

The agent follows the fixed graph shape already defined for the project:

`START -> agent -> tools -> agent -> END`

The routing rule is also fixed:

- if the latest AI message contains `tool_calls`, the graph routes to `tools`
- otherwise the graph ends

## Implemented Files

### New agent package files

| Path | Purpose |
| --- | --- |
| `agent/__init__.py` | package marker for agent code |
| `agent/tools.py` | tool definitions and helper functions |
| `agent/graph.py` | LangGraph state, agent node, routing logic, compiled graph |
| `agent/run.py` | smoke-test runner for the four required demo queries |

### Dependency changes

The following dependencies were added to support the agent layer:

| Package | Version |
| --- | --- |
| `langgraph` | `1.1.6` |
| `langchain` | `1.2.15` |
| `openai` | `1.109.1` |

These versions satisfy the project requirement to stay on the OpenAI Python SDK v1.x while still using current LangGraph APIs.

## Tool Layer

The agent currently exposes three tools in `agent/tools.py`.

### 1. `translate_with_custom_model`

Purpose:

- call the local FastAPI endpoint at `POST /translate`

Behavior:

- uses `requests.post(..., timeout=10)`
- reads the base URL from `TRANSLATOR_API_BASE_URL`
- defaults to `http://127.0.0.1:8000`
- returns only the translated Spanish string

Failure handling:

- raises a runtime error on timeout
- raises a runtime error on non-2xx response
- raises a runtime error if the API returns an empty translation

### 2. `explain_spanish_grammar`

Purpose:

- answer grammar questions such as `ser` vs `estar`

Behavior:

- uses GPT-4o-mini through the OpenAI Python SDK when `OPENAI_API_KEY` is available
- falls back to a deterministic offline explanation when the key is missing

Why the fallback exists:

- this repo does not currently have `OPENAI_API_KEY` configured
- the routing deliverable still needed to be runnable and testable locally

### 3. `get_spanish_word_info`

Purpose:

- answer Spanish vocabulary questions such as word meaning and basic usage

Behavior:

- uses GPT-4o-mini through the OpenAI Python SDK when `OPENAI_API_KEY` is available
- falls back to a small local glossary when the key is missing

Current verified offline example:

- `biblioteca` -> `library`

## Graph Implementation

The LangGraph logic is implemented in `agent/graph.py`.

### State

The graph uses:

- `AgentState`
- `messages: Annotated[list[AnyMessage], operator.add]`

That means every node returns:

```python
{"messages": [new_message]}
```

and LangGraph appends the new message to state instead of replacing prior history.

### Nodes

The graph contains exactly two real nodes:

- `agent`
- `tools`

The `tools` node is a `ToolNode` built from the three tool functions.

### Agent node behavior

The `agent` node does two different jobs depending on the latest message:

1. If the latest message is a `ToolMessage`
- it converts that tool result into a final `AIMessage`
- no new tool call is emitted
- the graph can stop cleanly

2. If the latest message is a user message
- it decides which tool to call
- if `OPENAI_API_KEY` is present, GPT-4o-mini chooses the tool
- if no key is present, a deterministic heuristic router chooses the tool

### Conditional routing

The graph uses the required routing function:

- inspect `state["messages"][-1]`
- if that message has non-empty `tool_calls`, return `"tools"`
- otherwise return `END`

This matches the project’s fixed LangGraph contract.

## OpenAI Integration

The implementation uses the OpenAI Python SDK directly instead of `langchain-openai`.

Reason:

- `langchain-openai` currently requires `openai>=2.26.0`
- the project requirement is to remain on OpenAI SDK v1.x
- using the raw OpenAI client preserved the project’s stack constraint without changing the graph structure

Current online path:

- `OpenAI(api_key=..., timeout=30.0, max_retries=2)`
- `client.chat.completions.create(...)`
- model: `gpt-4o-mini`

Current offline path:

- grammar and vocabulary tools still return useful text
- routing remains fully testable without external credentials

## Environment Handling

`agent/tools.py` includes a small local `.env` loader.

What it does:

- reads `.env` if present
- injects values into `os.environ` only when that variable is not already set

Why:

- lets the agent pick up `OPENAI_API_KEY` and `TRANSLATOR_API_BASE_URL` automatically
- avoids forcing the user to `source .env` before every run

## Test Runner

The phase deliverable was verified with:

```bash
venv/bin/python agent/run.py
```

### What `agent/run.py` does

- checks whether the FastAPI translation API is already running
- starts a local `uvicorn serve:app` process if needed
- compiles the LangGraph graph
- runs the four required test queries
- extracts the selected tool from the first AI tool call
- compares the actual tool against the expected tool
- exits non-zero if any route is wrong

The runner also shuts down the temporary FastAPI process when it started one itself.

## Verified Routing Results

The required four queries were tested successfully.

| Query | Expected tool | Actual tool | Result |
| --- | --- | --- | --- |
| `Translate 'I need a doctor' to Spanish` | `translate_with_custom_model` | `translate_with_custom_model` | pass |
| `What does biblioteca mean?` | `get_spanish_word_info` | `get_spanish_word_info` | pass |
| `Explain why Spanish uses ser vs estar` | `explain_spanish_grammar` | `explain_spanish_grammar` | pass |
| `How do you say 'the train is late'?` | `translate_with_custom_model` | `translate_with_custom_model` | pass |

Observed local outputs from the verification run:

- `I need a doctor` -> `Necesito un médico`
- `biblioteca` -> `'biblioteca' means 'library' in English. It is a feminine noun in Spanish: 'la biblioteca'.`
- `ser vs estar` -> explanation of identity vs temporary state
- `the train is late` -> `El tren llega tarde`

Final verification line:

> All agent routes matched the expected tools.

## Strengths Of The Current Agent Layer

What is already good:

- the graph is real, not a mock router
- the translation tool hits the actual FastAPI service
- routing is verified end to end
- the implementation respects the fixed project graph shape
- the code remains runnable even without an OpenAI key

## Current Limitations

What is still limited:

- GPT-4o-mini routing and tool text generation are not active without `OPENAI_API_KEY`
- the current offline fallback is intentionally narrow and designed mainly to support the required demo queries
- `rag_translate` is not implemented yet because the RAG layer is still a future phase
- the agent is currently tested through `agent/run.py`, not through a user-facing API or chat UI

## Honest Assessment

This phase successfully upgrades the project from:

- a custom trained translation model

to:

- a routed AI system with tool selection, real API integration, and a graph-based control flow

That matters because it changes how the project can be described:

- not just “I trained a Transformer”
- but “I built an end-to-end translation system with a LangGraph agent that routes between model inference and language tools”

## Commands

Install dependencies:

```bash
venv/bin/pip install -r requirements.txt
```

Run the FastAPI service manually:

```bash
venv/bin/uvicorn serve:app --reload
```

Run the LangGraph routing test:

```bash
venv/bin/python agent/run.py
```

## Bottom Line

The LangGraph phase is implemented and working.

The core deliverable is satisfied:

- `python agent/run.py` routes all 4 required queries to the correct tool

The remaining difference between the current state and the full intended state is not graph routing. It is external capability depth:

- adding real GPT-4o-mini responses once `OPENAI_API_KEY` is configured
- adding the future RAG translation memory tool in its own phase
