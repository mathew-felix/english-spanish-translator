# Project Report

Updated: 2026-04-09

## Snapshot

This repository contains an English-to-Spanish translation project built around a custom Transformer implemented from scratch in PyTorch.

Current scope:

- custom translation model and training pipeline
- API and Docker packaging for inference
- baseline comparison against MarianMT
- translation memory and revision step for institutional language

What is currently implemented:

- CLI pipeline for download, preprocessing, training, and evaluation
- OPUS-based English-Spanish dataset pipeline
- custom Transformer model
- training loop with W&B tracking
- Colab training notebook
- FastAPI inference endpoint
- Hugging Face baseline comparison against MarianMT
- Docker packaging
- LangGraph routing layer
- ChromaDB RAG translation memory
- Swagger UI screenshot and API documentation

## Implemented Components

| Area | Current state |
| --- | --- |
| CLI | `run.py` supports `--step download|preprocess|train|evaluate` |
| Data download | `source/DatasetDownload.py` downloads OPUS corpora |
| Preprocessing | `source/DatasetPreprocessing.py` merges and filters multi-corpus data |
| Dataset class | `source/DatasetTranslation.py` builds encoder/decoder/target tensors |
| Model | `source/Model.py` contains the custom Transformer |
| Training | `source/Train.py` trains, logs to W&B, and saves checkpoints |
| Evaluation | `source/Evaluate.py` computes corpus BLEU with `sacrebleu` |
| Inference | `source/inference.py` loads the model once and exposes `translate(text)` |
| API | `serve.py` exposes `/health` and `/translate` with FastAPI |
| Docker | `Dockerfile` and `docker-compose.yml` run the API in a container |
| Agent | `agent/` contains the LangGraph routing layer and tool runner |
| RAG | `rag/` contains the ChromaDB translation-memory builder and retriever |

## Current File Inventory

### Core project files

| Path | Purpose |
| --- | --- |
| `run.py` | CLI entrypoint |
| `serve.py` | FastAPI application |
| `source/Config.py` | central hyperparameters and paths |
| `source/DatasetDownload.py` | OPUS corpus download |
| `source/DatasetPreprocessing.py` | corpus merge, filtering, inspection, splitting |
| `source/DatasetTranslation.py` | PyTorch translation dataset |
| `source/Model.py` | custom Transformer implementation |
| `source/Train.py` | training loop and W&B logging |
| `source/Evaluate.py` | checkpoint-aware full test-set evaluation |
| `source/inference.py` | checkpoint-aware inference singleton |
| `templates/index.html` | browser UI for the institutional review process |
| `assets/ui.js` | step-by-step institutional review frontend logic |
| `assets/ui.css` | styling for the institutional review page |
| `Dockerfile` | container build for the FastAPI service |
| `docker-compose.yml` | local multi-file container launch for the API |
| `agent/graph.py` | LangGraph agent loop and conditional routing |
| `agent/tools.py` | direct translation and RAG review tools |
| `agent/run.py` | local routing smoke test |
| `rag/build_index.py` | builds the persistent Chroma translation memory |
| `rag/retriever.py` | lazy-loaded retrieval over the translation memory |
| `finetune/baseline_hf.py` | Hugging Face comparison runner |
| `finetune/manual_comparison_test_set.csv` | hand-written 50-row comparison benchmark |
| `assets/swagger_demo.png` | FastAPI Swagger screenshot |

## Data Pipeline

The dataset pipeline has moved away from the earlier Kaggle-only Europarl approach.

The current code downloads and preprocesses:

- `Europarl`
- `News-Commentary`
- `TED2020`
- `OpenSubtitles`

`source/DatasetPreprocessing.py` currently:

- locates `.en` and `.es` files
- normalizes punctuation and whitespace
- preserves Spanish punctuation such as `¿` and `¡`
- filters noisy subtitle data
- de-duplicates bilingual pairs
- writes a merged dataset
- splits train and test files

## Latest Verified Training Run

The latest full verified run completed in Colab with the settings and results listed below.

### Run configuration

| Item | Value |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Epochs | 30 |
| Batch size | 640 |
| Max sequence length | 60 |
| Learning rate | `4.5e-4` |
| Patience | 5 |
| Warmup steps | 4000 |
| Beam width | 4 |

### Dataset sizes from the completed run

| Corpus | Kept pairs |
| --- | --- |
| Europarl | 1,940,734 |
| News-Commentary | 46,904 |
| TED2020 | 403,752 |
| OpenSubtitles | 2,000,000 |
| Total | 4,391,390 |

Split used in the run:

- `3,512,826` train
- `878,564` test

### Training outcome

Key verified metrics from the completed run:

- epoch 1 validation loss: `4.2375`
- epoch 15 validation loss: `2.6032`
- best checkpoint: epoch 29 with validation loss `2.5055`
- final training-run bounded BLEU at epoch 29: `0.3167`
- full test-set `sacreBLEU`: `31.41`
- full evaluation wall time: `2:33:37`

W&B run:

- https://wandb.ai/relixmatrix-texas-state-university/english-spanish-translator/runs/acxn0hti

Important note:

- the trailing three console example lines at the very bottom of the original Colab console log were identified by the user as stale output from a previous run and are not part of the verified results for this training run

## API Layer

The FastAPI layer is implemented and verified locally.

Current endpoints:

- `GET /health`
- `POST /translate`
- `POST /institutional-review`

The API:

- loads the model once at startup
- uses checkpoint-aware config loading
- retries with a narrower beam when wide-beam decoding returns empty output
- validates input with Pydantic v2
- returns translation text and `latency_ms`
- exposes the structured institutional review flow for the browser UI

Observed local translation response:

```json
{
  "input": "Where is the nearest hospital?",
  "translation": "¿Dónde está el hospital más cercano?",
  "latency_ms": 21467.58
}
```

The exact latency is local-runtime dependent. This response was re-verified on 2026-04-09 after restoring the exported Colab artifacts into the project paths used by `Config`.

## Docker Layer

Docker packaging is implemented and verified locally.

Current files:

- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`

Verified local result:

- `docker compose up --build` starts the API successfully
- `GET /health` returns `{"status":"ok"}`
- `POST /translate` returns the same translation path as the local FastAPI process

This means the translation API can now be started from a clean container build instead of depending on the local Python environment.

## Institutional Translation Path

The repository includes a second translation path for institutional text:

- institutional English sentence in
- custom model creates the first Spanish draft
- retrieval memory surfaces similar Europarl examples
- GPT reviews the draft against those examples
- final Spanish translation is returned

This flow is available through:

- the browser UI at `/`
- the structured API endpoint `POST /institutional-review`

Verified structured result:

```json
{
  "input": "The parliamentary session was adjourned.",
  "draft_translation": "Se suspendió la sesión parlamentaria.",
  "decision": "EDIT",
  "final_translation": "Se interrumpe la sesión parlamentaria."
}
```

Use this path for:

- parliamentary wording
- committee and council language
- amendments, motions, and other institutional text

Do not use this path as the default for:

- general everyday translation
- casual conversation
- broad consumer translation quality claims

## Revision Path Implementation

The repository now includes both:

- a lightweight routing layer
- a ChromaDB-based translation-memory layer

The focused agent now exposes two tools:

- `translate_with_custom_model`
- `rag_translate`

The graph structure remains:

`START -> agent -> tools -> agent -> END`

The most important verified hybrid behavior is now in `rag_translate`:

- the custom model generates the first translation draft
- ChromaDB retrieves the top 3 similar Europarl pairs
- GPT-4o-mini reviews the draft against the retrieved context
- GPT either keeps the draft or edits it

Verified example after OpenAI billing was enabled:

```text
Decision: EDIT
Custom model draft: Se suspendió la sesión parlamentaria.
Translation: Se interrumpe la sesión parlamentaria.
```

This creates a dependency between the custom model draft and the GPT revision step instead of running them as separate features.

## Hugging Face Comparison Layer

The repository now includes a baseline comparison against `Helsinki-NLP/opus-mt-en-es`.

Comparison artifacts:

- `finetune/baseline_hf.py`
- `finetune/manual_comparison_test_set.csv`

The generated comparison result files are kept locally and are not part of the public repo.

Verified comparison setup:

- hand-written 50-row benchmark with clean English/Spanish references
- ten everyday domains with five rows each
- same references for both outputs
- custom model translated through the local inference runtime
- MarianMT translated through the pretrained Hugging Face model

Observed local CPU comparison summary:

- custom average latency: `6518.67 ms`
- MarianMT average latency: `470.43 ms`
- exact reference matches: `11 / 50` for the custom model vs `20 / 50` for MarianMT
- MarianMT was stronger overall on fluency and lexical accuracy
- the custom Transformer still produced grammatically valid Spanish on many everyday rows

## Model Status

The core model remains a genuine custom Transformer, not `torch.nn.Transformer`.

Implemented model properties:

- learned embeddings
- sinusoidal positional encoding
- encoder-decoder architecture
- weight tying
- causal masking
- padding masks in attention
- beam-search style generation

The training pipeline currently uses `bert-base-multilingual-cased` with:

- `<PAD>`
- `<UNK>`
- `<SOS>`
- `<END>`

## Project Status

Currently verified:

- end-to-end custom model training pipeline works
- large multi-corpus preprocessing pipeline works
- W&B experiment tracking works
- full evaluation works
- FastAPI serving works
- Dockerized serving works
- LangGraph routing works
- ChromaDB retrieval works
- GPT-backed draft review works on the RAG translation path
- the browser demo now presents one coherent application instead of multiple unrelated side features
- MarianMT comparison artifacts now exist for interview discussion
- exported Colab artifacts can now be reloaded locally for evaluation and serving
- the Colab run notebook works on high-memory GPU hardware

## Remaining Gaps

The project is functional, but several practical gaps remain:

- a full production deployment setup
- broader human evaluation beyond corpus BLEU

## Bottom Line

This project currently includes:

- a custom Transformer
- a real multi-corpus training run
- verified W&B tracking
- a FastAPI inference layer
- Docker packaging
- a focused institutional translation path
- a ChromaDB translation memory
- Colab-based reproducible training

Current short description:

> a custom English-to-Spanish Transformer system trained end to end on a large OPUS corpus mix, evaluated at `31.41 sacreBLEU`, served through FastAPI, and extended with a translation-memory-based revision path for institutional language
