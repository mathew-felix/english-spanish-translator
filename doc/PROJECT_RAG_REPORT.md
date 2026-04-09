# RAG Translation Memory Report

Updated: 2026-04-09

## Overview

This report documents the Phase 7 Retrieval-Augmented Generation implementation added to the English-to-Spanish translation project.

The goal of this phase was to add a translation memory over 50,000 Europarl sentence pairs so the system can retrieve similar institutional wording before producing a translation. This improves terminology consistency for parliamentary and formal-domain text without changing the underlying custom Transformer architecture.

This review path should be described carefully:

- it is meant for institutional language
- it is not a claim that GPT or retrieval improves every sentence
- it is not a replacement for honest baseline comparison

The RAG layer follows the project design exactly:

- vector store: ChromaDB persistent client
- collection name: `translation_memory`
- distance metric: cosine
- embeddings: `all-MiniLM-L6-v2`
- retrieval count: `k=3`
- agent tool: `rag_translate`

## Implemented Files

### New RAG package files

| Path | Purpose |
| --- | --- |
| `rag/__init__.py` | package marker for RAG code |
| `rag/build_index.py` | builds a persistent ChromaDB index from Europarl rows |
| `rag/retriever.py` | lazy-loaded retrieval API for similar bilingual pairs |

### Updated agent files

| Path | Purpose |
| --- | --- |
| `agent/tools.py` | adds the `rag_translate` tool |
| `agent/graph.py` | routes parliamentary and institutional translation requests to `rag_translate` |

### Supporting file changes

| Path | Purpose |
| --- | --- |
| `.gitignore` | ignores the generated `rag/chroma_db/` index |
| `requirements.txt` | adds ChromaDB and sentence-transformers dependencies |

## Dependency Additions

The following packages were added for the RAG phase:

| Package | Version |
| --- | --- |
| `chromadb` | `0.5.23` |
| `sentence-transformers` | `3.4.1` |

The project also aligned the Hugging Face stack used by the embedding layer:

| Package | Version |
| --- | --- |
| `transformers` | `4.46.3` |
| `tokenizers` | `0.20.3` |

These versions work with the local RAG implementation and the existing Python 3.12 environment.

## Index Builder

The translation-memory builder is implemented in `rag/build_index.py`.

### Data source

The builder reads from:

- `data/train.csv`

It keeps only rows where:

- `Corpus == "Europarl"`
- English text is non-empty
- Spanish text is non-empty

### Build scope

The builder indexes:

- up to `50,000` Europarl pairs

This is intentionally capped so the translation memory stays focused and practical while still being large enough to support useful retrieval.

### Storage details

The persistent vector index is written to:

- `rag/chroma_db/`

This directory is gitignored because it is generated locally and can be rebuilt from the training CSV.

### Embedding behavior

English source sentences are embedded with:

- `all-MiniLM-L6-v2`

The builder uses normalized embeddings and batches the Chroma `add()` calls so the full 50K-row build completes cleanly.

## Retriever

The retrieval interface is implemented in `rag/retriever.py`.

### Public function

`retrieve_similar_translations(query, k=3) -> list[dict]`

Each returned row contains:

- `english`
- `spanish`
- `corpus`
- `source_index`
- `distance`

### Lazy loading

The retriever uses lazy process-wide caching for:

- the Chroma persistent client
- the `translation_memory` collection
- the sentence-transformer embedding model

This keeps startup lighter and avoids reopening the database or reloading the embedding model on every query.

### Error behavior

The retriever raises:

- `ValueError` for empty queries
- `ValueError` when `k < 1`
- `RuntimeError` if the Chroma index has not been built yet

The missing-index error tells the user to run:

```bash
venv/bin/python rag/build_index.py
```

## Agent Integration

The new agent tool is implemented in `agent/tools.py` as `rag_translate`.

### What `rag_translate` does

1. calls the custom FastAPI translation model first to get a draft translation
2. retrieves the top 3 similar English-Spanish Europarl pairs
3. formats those pairs into compact bilingual context
4. sends the source text, the custom-model draft, and the retrieved context to GPT-4o-mini
5. returns the decision, the original draft, the final translation, and the retrieved context

### Prompting behavior

The GPT review prompt tells GPT-4o-mini to:

- review a custom English-to-Spanish draft
- preserve parliamentary and institutional terminology
- use the retrieved examples for consistency
- keep the draft when it is already correct
- make the smallest necessary edit when the draft should be improved
- return both a `KEEP` or `EDIT` decision and the final translation

### Offline fallback

If `OPENAI_API_KEY` is not available, `rag_translate` does not fail.

Instead, it:

- still gets the custom-model draft
- still performs retrieval
- still logs the retrieved context
- keeps the custom-model draft as the final translation

This keeps the RAG pipeline demonstrable even when OpenAI access is unavailable locally.

## Graph Routing Update

The LangGraph structure was not changed.

The graph still follows:

`START -> agent -> tools -> agent -> END`

The only RAG-related update in `agent/graph.py` is the tool-selection logic.

Institutional translation requests containing terms such as:

- `parliament`
- `parliamentary`
- `session`
- `adjourned`
- `motion`
- `committee`
- `commission`
- `council`
- `amendment`
- `rapporteur`
- `plenary`

are now routed to:

- `rag_translate`

General translation requests still route to:

- `translate_with_custom_model`

This preserved the project’s fixed graph architecture while adding domain-aware routing.

## Verified Build And Retrieval

The RAG phase was verified locally with the following commands:

```bash
venv/bin/python rag/build_index.py
venv/bin/python agent/run.py
```

The index build completed successfully and wrote:

- `50,000` Europarl sentence pairs

Observed build result:

```text
Built translation memory with 50000 Europarl sentence pairs.
```

### Retrieval example

Query:

```text
The parliamentary session was adjourned.
```

Top retrieved rows:

1. `The session is adjourned.` → `Se interrumpe el periodo de sesiones.`
2. `Adjournment of the session` → `Interrupción del periodo de sesiones`
3. `I declare adjourned the session of the European Parliament.` → `Declaro interrumpido el período de sesiones del Parlamento Europeo.`

### Initial offline `rag_translate` output example

Observed tool output:

```text
Decision: KEEP (offline fallback)
Custom model draft: Se suspendió la sesión parlamentaria.
Translation: Se suspendió la sesión parlamentaria.
Retrieved context:
1. EN: The session is adjourned. | ES: Se interrumpe el periodo de sesiones. | distance=0.177841
2. EN: Adjournment of the session | ES: Interrupción del periodo de sesiones | distance=0.225302
3. EN: I declare adjourned the session of the European Parliament. | ES: Declaro interrumpido el período de sesiones del Parlamento Europeo. | distance=0.298554
```

This shows that the tool returns:

- a review decision
- the custom-model draft
- a final Spanish translation
- the exact retrieved bilingual context used for that decision

### Verified GPT-backed review output

After `OPENAI_API_KEY` was configured with working billing, the same tool path was re-tested successfully.

Observed output:

```text
Decision: EDIT
Custom model draft: Se suspendió la sesión parlamentaria.
Translation: Se interrumpe la sesión parlamentaria.
Retrieved context:
1. EN: The session is adjourned. | ES: Se interrumpe el periodo de sesiones. | distance=0.177841
2. EN: Adjournment of the session | ES: Interrupción del periodo de sesiones | distance=0.225302
3. EN: I declare adjourned the session of the European Parliament. | ES: Declaro interrumpido el período de sesiones del Parlamento Europeo. | distance=0.298554
```

This is the important dependency result for the project:

- the custom model generated the first-pass translation
- retrieval supplied domain memory
- GPT revised the draft using the retrieved examples

So the RAG path is no longer just "GPT translates with context." It is now "custom model drafts, GPT reviews and edits when needed."

## When The Review Path Helps

This flow is strongest when:

- the sentence is close to Europarl style
- terminology consistency matters
- a similar institutional phrasing likely already exists in the memory

Examples:

- parliamentary sessions
- committee actions
- council decisions
- motions and amendments

## When The Review Path Is Not The Right Story

This flow is weaker when:

- the sentence is casual or conversational
- the topic is far from parliamentary language
- the retrieved Europarl examples are only loosely related

That is why the project should not claim that this path is a universal translation improvement layer. It is a domain-aware revision layer.

## Validation Outcome

The RAG implementation is working as intended.

What is confirmed:

- the index builds from local training data
- the retriever returns sensible Europarl neighbors
- the agent routes institutional translation requests to `rag_translate`
- the custom model is now a required first-pass step in `rag_translate`
- GPT-backed review successfully edited a real custom-model draft after OpenAI billing was enabled
- the original four-agent smoke-test queries still pass
- the graph structure was preserved exactly

## Known Limitations

### 1. GPT review path depends on `OPENAI_API_KEY`

Without an OpenAI key:

- retrieval still works
- routing still works
- context is still shown
- the custom-model draft is kept unchanged

So the RAG layer is operational, but the full draft-review path is only active when `OPENAI_API_KEY` is configured and the account has available quota.

### 2. Chroma telemetry warnings still appear

Local runs still print Chroma telemetry warnings even though telemetry is disabled in code.

These warnings did not block:

- index creation
- retrieval
- agent execution

So this currently looks like a library-side warning rather than a project logic bug.

### 3. Memory scope is intentionally narrow

The current index uses only:

- Europarl rows

That is correct for the project goal of parliamentary translation memory, but it means the RAG context is optimized for institutional language rather than broad everyday conversation.

## Practical Commands

Build the index:

```bash
venv/bin/python rag/build_index.py
```

Test retrieval directly:

```bash
venv/bin/python - <<'PY'
from rag.retriever import retrieve_similar_translations
print(retrieve_similar_translations("The parliamentary session was adjourned.", k=3))
PY
```

Run the agent smoke test:

```bash
venv/bin/python agent/run.py
```

Test the full GPT-backed review path directly:

```bash
venv/bin/python - <<'PY'
from agent.tools import rag_translate
print(rag_translate.invoke({"text": "The parliamentary session was adjourned."}))
PY
```

## Final Assessment

This phase successfully turns the project from a plain translation API into a translation system with domain memory.

The important outcome is not just that ChromaDB was added. The important outcome is that the system can now:

- recognize institutional translation requests
- retrieve similar known bilingual wording
- expose that context for traceability
- force the custom model to produce the first draft
- let GPT-4o-mini review that draft against translation-memory examples

That makes the translation stack more defensible in interviews because it shows retrieval, orchestration, and domain adaptation on top of the core custom model.
