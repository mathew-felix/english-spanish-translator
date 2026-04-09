# Project Report

Updated: 2026-04-09

## Snapshot

This repository now contains a working English-to-Spanish translation project built around a custom Transformer implemented from scratch in PyTorch.

What is currently implemented:

- CLI pipeline for download, preprocessing, training, and evaluation
- OPUS-based English-Spanish dataset pipeline
- custom Transformer model
- training loop with W&B tracking
- Colab training notebook
- FastAPI inference endpoint
- Hugging Face baseline comparison against MarianMT
- Swagger UI screenshot and API documentation

What is not implemented yet:

- Docker packaging
- LangGraph agent layer
- ChromaDB RAG layer
- `CURRENT_PROJECT_STATUS.md`

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
| Colab | `colab_training.ipynb` supports setup, training, evaluation, and artifact export |

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
| `finetune/baseline_hf.py` | Hugging Face comparison runner |
| `finetune/manual_comparison_test_set.csv` | hand-written 50-row comparison benchmark |
| `finetune/custom_model_results_manual.json` | custom-model outputs on the manual benchmark |
| `finetune/baseline_results_manual.json` | MarianMT outputs on the manual benchmark |
| `colab_training.ipynb` | Colab workflow |
| `assets/swagger_demo.png` | FastAPI Swagger screenshot |

### Reports

| Path | Purpose |
| --- | --- |
| `doc/PROJECT_REPORT.md` | project-wide status |
| `doc/PROJECT_FASTAPI_REPORT.md` | FastAPI phase report |
| `doc/TRAINING_REPORT.md` | completed training run report |
| `doc/HF_COMPARISON_REPORT.md` | MarianMT baseline comparison report |
| `doc/MODEL_SPOTCHECK_REPORT.md` | exported checkpoint spot-check results |

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

The latest full verified run is the Colab run captured in `output.txt`.

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

Key verified metrics from `output.txt`:

- epoch 1 validation loss: `4.2375`
- epoch 15 validation loss: `2.6032`
- best checkpoint: epoch 29 with validation loss `2.5055`
- final training-run bounded BLEU at epoch 29: `0.3167`
- full test-set `sacreBLEU`: `31.41`
- full evaluation wall time: `2:33:37`

W&B run:

- https://wandb.ai/relixmatrix-texas-state-university/english-spanish-translator/runs/acxn0hti

Important note:

- the trailing three console example lines at the very bottom of `output.txt` were identified by the user as stale output from a previous run and are not part of the verified results for this training run

## API Layer

The FastAPI layer is implemented and verified locally.

Current endpoints:

- `GET /health`
- `POST /translate`

The API:

- loads the model once at startup
- uses checkpoint-aware config loading
- retries with a narrower beam when wide-beam decoding returns empty output
- validates input with Pydantic v2
- returns translation text and `latency_ms`

Observed local translation response:

```json
{
  "input": "Where is the nearest hospital?",
  "translation": "¿Dónde está el hospital más cercano?",
  "latency_ms": 21467.58
}
```

The exact latency is local-runtime dependent. This response was re-verified on 2026-04-09 after restoring the exported Colab artifacts into the project paths used by `Config`.

## Hugging Face Comparison Layer

The repository now includes a baseline comparison against `Helsinki-NLP/opus-mt-en-es`.

Comparison artifacts:

- `finetune/baseline_hf.py`
- `finetune/manual_comparison_test_set.csv`
- `finetune/custom_model_results_manual.json`
- `finetune/baseline_results_manual.json`

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

## Project Strengths

What is genuinely strong in the current project:

- end-to-end custom model training pipeline works
- large multi-corpus preprocessing pipeline works
- W&B experiment tracking works
- full evaluation works
- FastAPI serving works
- MarianMT comparison artifacts now exist for interview discussion
- exported Colab artifacts can now be reloaded locally for evaluation and serving
- Colab workflow works on high-memory GPU hardware

## Remaining Gaps

The project is functional, but several planned layers are still missing:

- Docker / container deployment
- LangGraph orchestration
- RAG translation memory

Also still missing:

- a cleaned, fully current `CURRENT_PROJECT_STATUS.md`
- a true production deployment story
- broader human evaluation beyond corpus BLEU

## Bottom Line

This project has moved beyond a simple training script. It is now a working ML systems project with:

- a custom Transformer
- a real multi-corpus training run
- verified W&B tracking
- a FastAPI inference layer
- Colab-based reproducible training

The strongest current claim is:

> a custom English-to-Spanish Transformer system trained end to end on a large OPUS corpus mix, evaluated at `31.41 sacreBLEU`, tracked in W&B, and served through FastAPI
