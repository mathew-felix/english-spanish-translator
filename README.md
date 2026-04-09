# English-to-Spanish Translator

[![Weights & Biases](https://img.shields.io/badge/W%26B-Latest%20Run-FFBE00?logo=weightsandbiases&logoColor=black)](https://wandb.ai/relixmatrix-texas-state-university/english-spanish-translator/runs/acxn0hti)

This project implements an English-to-Spanish translation system around a custom Transformer built from raw PyTorch modules. The repository now includes the full training pipeline, evaluation, Weights & Biases tracking, a Colab training notebook, and a FastAPI inference layer.

## Latest Verified Run

The latest full end-to-end training run was completed in Colab and captured in `output.txt`.

| Item | Value |
| --- | --- |
| Hardware | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Epochs | 30 |
| Batch size | 640 |
| Max sequence length | 60 |
| Learning rate | `4.5e-4` |
| Train split | 3,512,826 pairs |
| Test split | 878,564 pairs |
| Best validation loss | `2.5055` at epoch 29 |
| Final test sacreBLEU | `31.41` |
| W&B run | `acxn0hti` (`silvery-galaxy-1`) |
| Full evaluation time | `2:33:37` |

Run links:

- Project: https://wandb.ai/relixmatrix-texas-state-university/english-spanish-translator
- Latest run: https://wandb.ai/relixmatrix-texas-state-university/english-spanish-translator/runs/acxn0hti

## Features

- Custom encoder-decoder Transformer implemented from scratch in PyTorch
- OPUS English-Spanish corpus mix: `Europarl + News-Commentary + TED2020 + filtered OpenSubtitles`
- Tokenization with `bert-base-multilingual-cased` plus custom special tokens
- Training with AMP, warmup scheduling, label smoothing, gradient clipping, and W&B logging
- Corpus-level evaluation with `sacrebleu`
- Colab notebook for GPU training and Google Drive artifact export
- FastAPI inference endpoint with `/health`, `/translate`, and Swagger docs

## Tech Stack

- Python 3.12
- PyTorch
- Hugging Face `transformers`
- FastAPI
- Pydantic v2
- Weights & Biases
- pandas / NumPy / matplotlib / tqdm / sacrebleu

## Installation

```bash
git clone https://github.com/mathew-felix/english-spanish-translator.git
cd english-spanish-translator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Pipeline Usage

```bash
python run.py --step download
python run.py --step preprocess
python run.py --step train
python run.py --step evaluate
```

## Dataset Summary

The completed OPUS preprocessing run kept:

- `Europarl`: 1,940,734 pairs
- `News-Commentary`: 46,904 pairs
- `TED2020`: 403,752 pairs
- `OpenSubtitles`: 2,000,000 pairs

Total merged dataset:

- `4,391,390` aligned sentence pairs

Train/test split from the verified run:

- `3,512,826` train
- `878,564` test

## Training Results

The completed 30-epoch run showed stable training and steady validation improvement:

- epoch 1 validation loss: `4.2375`
- epoch 15 validation loss: `2.6032`
- epoch 29 validation loss: `2.5055`
- final full-test sacreBLEU: `31.41`

BLEU score distribution from the full evaluation run:

![BLEU Score Distribution](bleu_score_distribution.png)

Late-epoch qualitative samples from the training log:

- `How are you? -> ¿Cómo estás?`
- `Where is the hospital? -> ¿Dónde está el hospital?`
- `I need help with my homework. -> Necesito ayuda con mis deberes.`

Detailed run analysis is in `doc/TRAINING_REPORT.md`.

## API

Run the FastAPI server locally:

```bash
uvicorn serve:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Translation request:

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Where is the nearest hospital?"}'
```

Observed local response from the current checkpoint:

```json
{
  "input": "Where is the nearest hospital?",
  "translation": "¿Dónde está el hospital más cercano?",
  "latency_ms": 21467.58
}
```

The exact latency depends on local hardware. The response above is from the latest local verification run on CPU.

Swagger UI screenshot:

![Swagger UI](assets/swagger_demo.png)

## Weights & Biases

Set your API key before training if you want online tracking:

```bash
export WANDB_API_KEY=your_api_key
python run.py --step train
```

Latest verified run:

- https://wandb.ai/relixmatrix-texas-state-university/english-spanish-translator/runs/acxn0hti

## Reports

- `doc/PROJECT_REPORT.md`: current project-wide status
- `doc/PROJECT_FASTAPI_REPORT.md`: FastAPI inference layer report
- `doc/TRAINING_REPORT.md`: completed training run report
- `doc/MODEL_SPOTCHECK_REPORT.md`: exported checkpoint spot-check results

## License

This project is licensed under the MIT License.
