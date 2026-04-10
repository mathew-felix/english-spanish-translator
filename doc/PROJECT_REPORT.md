# Project Report

Updated: 2026-04-10

## Snapshot

This repository contains an English-to-Spanish translation system built around a custom Transformer written from scratch in PyTorch. The main work covers dataset preparation, model training, evaluation, and comparison against a MarianMT baseline. The same repository also includes FastAPI serving, Docker packaging, and a second translation path for institutional text that uses retrieved Europarl examples and a GPT revision step.

## Implemented Scope

- CLI pipeline for download, preprocessing, training, and evaluation
- OPUS-based English-Spanish dataset pipeline
- custom Transformer model in `source/Model.py`
- training loop with Weights & Biases tracking
- Colab training notebook
- FastAPI endpoints for direct translation and institutional review
- Docker packaging for the API
- MarianMT comparison runner in `finetune/baseline_hf.py`
- translation memory over `50K` Europarl pairs with ChromaDB
- institutional translation flow and browser demo

## Data Pipeline

The completed preprocessing run merged four corpora:

| Corpus | Kept pairs |
| --- | --- |
| Europarl | 1,940,734 |
| News-Commentary | 46,904 |
| TED2020 | 403,752 |
| OpenSubtitles | 2,000,000 |
| Total | 4,391,390 |

Train/test split used in the completed run:

- `3,512,826` train pairs
- `878,564` test pairs

`source/DatasetPreprocessing.py` handles file discovery, text cleanup, duplicate removal, subtitle filtering, and train/test split generation.

## Training Run

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

### Run results

- epoch 1 validation loss: `4.2375`
- epoch 15 validation loss: `2.6032`
- best checkpoint: epoch 29 with validation loss `2.5055`
- final training-run bounded BLEU at epoch 29: `0.3167`
- full test-set `sacreBLEU`: `31.41`
- full evaluation wall time: `2:33:37`

W&B run:

- https://wandb.ai/relixmatrix-texas-state-university/english-spanish-translator/runs/acxn0hti

Late-epoch console samples from the completed run:

- `How are you? -> ¿Cómo estás?`
- `Where is the hospital? -> ¿Dónde está el hospital?`
- `I need help with my homework. -> Necesito ayuda con mis deberes.`

Run note:

- the three trailing sample lines at the very bottom of the original Colab console log came from an earlier run and are not part of the results above

## Evaluation and Testing

### BLEU evaluation

The full held-out evaluation completed on `878,564` test pairs.

- corpus metric: `31.41 sacreBLEU`
- BLEU score distribution plot saved as `bleu_score_distribution.png`
- training curve plot saved as `loss_plot.png`

### FastAPI checks

Local API checks were run on a MacBook M4 laptop. These timings are local CPU checks and should not be read as the training-time GPU performance.

Endpoints checked:

- `GET /health`
- `POST /translate`
- `POST /institutional-review`

Observed `/translate` response:

```json
{
  "input": "Where is the nearest hospital?",
  "translation": "¿Dónde está el hospital más cercano?",
  "latency_ms": 21467.58
}
```

Observed `/institutional-review` response:

```json
{
  "input": "The parliamentary session was adjourned.",
  "draft_translation": "Se suspendió la sesión parlamentaria.",
  "decision": "EDIT",
  "final_translation": "Se interrumpe la sesión parlamentaria.",
  "retrieved_examples": [
    {
      "english": "The session is adjourned.",
      "spanish": "Se interrumpe el periodo de sesiones.",
      "distance": 0.177841
    },
    {
      "english": "Adjournment of the session",
      "spanish": "Interrupción del periodo de sesiones",
      "distance": 0.225302
    },
    {
      "english": "I declare adjourned the session of the European Parliament.",
      "spanish": "Declaro interrumpido el período de sesiones del Parlamento Europeo.",
      "distance": 0.298554
    }
  ],
  "latency_ms": 10508.06
}
```

### Docker checks

Docker verification completed locally.

- `docker compose up --build` started the API
- `GET /health` returned `{"status":"ok"}`
- `POST /translate` returned the same translation path as the local FastAPI process

### Institutional review checks

The institutional path uses the custom-model draft first, then retrieval, then a GPT revision step.

Structured result:

```json
{
  "input": "The parliamentary session was adjourned.",
  "draft_translation": "Se suspendió la sesión parlamentaria.",
  "decision": "EDIT",
  "final_translation": "Se interrumpe la sesión parlamentaria."
}
```

This path is intended for parliamentary and committee language. The direct `/translate` path remains the default for ordinary sentences.

## MarianMT Comparison

The repository includes a comparison against `Helsinki-NLP/opus-mt-en-es` using a hand-written 50-sentence benchmark in `finetune/manual_comparison_test_set.csv`.

Comparison command:

```bash
venv/bin/python finetune/baseline_hf.py \
  --csv-path finetune/manual_comparison_test_set.csv \
  --limit 50 \
  --custom-output custom_model_results_manual.json \
  --baseline-output baseline_results_manual.json
```

The generated JSON result files are kept locally.

The local comparison run was also measured on a MacBook M4 laptop.

- custom model average latency: `6518.67 ms`
- MarianMT average latency: `470.43 ms`
- exact reference matches: `11 / 50` for the custom model
- exact reference matches: `20 / 50` for MarianMT
- MarianMT was stronger overall on fluency and lexical accuracy

Sample rows from the benchmark:

| English | Custom Transformer | MarianMT |
| --- | --- | --- |
| Where can I buy a train ticket to Madrid? | ¿Dónde puedo comprar un billete de tren a Madrid? | ¿Dónde puedo comprar un billete de tren a Madrid? |
| We need an ambulance right away. | Necesitamos una ambulancia enseguida. | Necesitamos una ambulancia de inmediato. |
| Did you remember to back up the files? | ¿Recuerdas retrasar los archivos? | ¿Te acordaste de hacer copias de seguridad de los archivos? |
| I need to reset my password again. | Necesito reanudar mi contraseña otra vez. | Necesito restablecer mi contraseña de nuevo. |
| The washing machine stopped working this morning. | La lavadora dejó de trabajar esta mañana. | La lavadora dejó de funcionar esta mañana. |

## Remaining Gaps

- production deployment beyond local and container verification
- broader human evaluation beyond corpus BLEU and the 50-sentence manual benchmark

## Summary

The current project state includes a trained custom Transformer, a completed large-scale run with `31.41 sacreBLEU`, FastAPI serving, Docker packaging, a MarianMT comparison, and an institutional translation path built on retrieved Europarl examples and a GPT revision step.
