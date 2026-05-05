# English-to-Spanish Translator

Custom PyTorch Transformer for English-to-Spanish translation with a FastAPI
serving layer, RAG-assisted institutional review path, and reproducible training
pipeline.

[![CI](https://github.com/mathew-felix/english-spanish-translator/actions/workflows/ci.yml/badge.svg)](https://github.com/mathew-felix/english-spanish-translator/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![W&B Run](https://img.shields.io/badge/W%26B-run%20acxn0hti-FFBE00?logo=weightsandbiases&logoColor=black)](https://wandb.ai/relixmatrix-texas-state-university/english-spanish-translator/runs/acxn0hti)

## Demo

![Demo GIF](assets/demo.gif)

The demo walks through the browser UI: English institutional text in, staged draft/context/decision/final Spanish out (recorded via `scripts/render_ui_demo_gif.py` against a running API). To re-record: start the stack, then run `python scripts/render_ui_demo_gif.py` (requires dev deps including Playwright; run `playwright install chromium` once).

---

## Table of Contents

- [Demo](#demo)
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Reproducing Results](#reproducing-results)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository contains an end-to-end English-to-Spanish machine translation
project. The core model is a custom encoder-decoder Transformer implemented in
PyTorch and trained on OPUS English-Spanish corpora. The serving layer exposes a
direct translation endpoint and an institutional review endpoint that can combine
the custom model, Europarl retrieval examples, and an optional GPT revision step.

The intended users are ML practitioners, students, and reviewers who need a
cloneable project that demonstrates data download, preprocessing, training,
evaluation, model serving, and deployment hygiene.

## Features

- Custom PyTorch Transformer in `source/Model.py`.
- OPUS corpus downloader and preprocessing pipeline.
- Reported 30-epoch training run over 4,391,390 aligned sentence pairs.
- Final reported held-out test score: 31.41 sacreBLEU.
- FastAPI endpoints for health checks, direct translation, and institutional
  translation review.
- Optional ChromaDB translation memory over Europarl examples.
- Optional GPT-4o-mini review when `OPENAI_API_KEY` is configured.
- Docker runtime that can download pinned release model artifacts on startup.
- CI-ready lint, tests, and dependency-audit commands.

## Requirements

- Python 3.12 for local development and CI.
- Git for cloning the repository.
- Internet access for downloading release model artifacts and OPUS datasets.
- Docker Desktop or Docker Engine for containerized API serving.
- CUDA-capable GPU recommended for training; CPU works for API smoke tests but is
  slower for model inference.

## Installation

Clean local setup from clone to running API:

```bash
git clone https://github.com/mathew-felix/english-spanish-translator.git; cd english-spanish-translator
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts\download_model.py --tag eng-sp-tranlate
.venv\Scripts\python -m uvicorn src.serve:app --host 127.0.0.1 --port 8000
```

On macOS or Linux, replace `.venv\Scripts\python` with `.venv/bin/python`.

After `make dev-install`, you can run the API with `make api` (same as the
`uvicorn` line above).

Optional environment configuration:

```bash
copy .env.example .env
```

Use `.env` for local values only. Keep real keys out of git.

Docker setup:

```bash
docker compose up --build
```

Expected health check:

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status":"ok"}
```

## Usage

Direct translation:

```bash
curl -X POST http://127.0.0.1:8000/translate ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"Where is the nearest hospital?\"}"
```

Expected response shape:

```json
{
  "input": "Where is the nearest hospital?",
  "translation": "<Spanish translation>",
  "latency_ms": 12.34
}
```

Institutional review:

```bash
curl -X POST http://127.0.0.1:8000/institutional-review ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"The parliamentary session was adjourned.\"}"
```

Build the local retrieval index before relying on institutional examples:

```bash
.venv\Scripts\python rag\build_index.py
```

Run the full ML pipeline:

```bash
.venv\Scripts\python -m src.run --step download
.venv\Scripts\python -m src.run --step preprocess
.venv\Scripts\python -m src.run --step train
.venv\Scripts\python -m src.run --step evaluate
```

## Reproducing Results

See [REPRODUCE.md](REPRODUCE.md) for the numbered reproduction procedure. The
reported run used:

- hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition
- epochs: 30
- batch size: 640
- max sequence length: 60
- learning rate: `4.5e-4`
- train split: 3,512,826 pairs
- test split: 878,564 pairs
- best validation loss: `2.5055` at epoch 29
- final test sacreBLEU: `31.41`

Generated datasets, checkpoints, tokenizer files, W&B runs, and RAG indexes are
ignored by git. Download or regenerate them locally.

## Project Structure

```text
english-spanish-translator/
|-- .github/                 # CI workflow, issue templates, PR template
|-- agent/                   # LangGraph translation router
|-- assets/                  # Browser UI assets
|-- doc/                     # Original project report
|-- docs/                    # Production deployment and security notes
|-- finetune/                # MarianMT comparison script and manual test set
|-- rag/                     # ChromaDB index builder and retriever
|-- notebooks/               # Jupyter notebooks (outputs cleared before commit)
|-- scripts/                 # Model download, demo recording, utilities
|-- source/                  # Core data, model, training, evaluation, inference code
|-- src/                     # FastAPI app (`src.serve`) and pipeline CLI (`src.run`)
|-- templates/               # FastAPI HTML template
|-- tests/                   # CI-friendly contract and security tests
|-- Dockerfile               # Containerized API runtime
|-- docker-compose.yml       # Local container orchestration
|-- requirements.txt         # Pinned runtime dependencies
`-- requirements-dev.txt     # Pinned development dependencies
```

## Running Tests

Install dev dependencies:

```bash
.venv\Scripts\python -m pip install -r requirements-dev.txt
```

Run lint, tests, and dependency audit:

```bash
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m pytest tests -v --cov=src --cov=scripts.download_model --cov-fail-under=60
.venv\Scripts\python -m pip_audit -r requirements.txt
```

The smoke tests stub heavyweight model dependencies so they can run in CI without
downloading the checkpoint or tokenizer.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Before opening a pull request, run lint,
tests, audit, and confirm notebook outputs are cleared.

## License

This project is licensed under the [MIT License](LICENSE).
