# Contributing

## Development Setup

Use Python 3.12 and install both runtime and development dependencies:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt
```

On macOS or Linux, replace `.venv\Scripts\python` with `.venv/bin/python`.

## Quality Gates

Run these before opening a pull request:

```bash
python -m ruff check .
python -m pytest tests -v --cov=src --cov=scripts.download_model --cov-fail-under=60
python -m pip_audit -r requirements.txt
```

## Secrets

Never commit real API keys, W&B keys, model service tokens, `.env` files, model
checkpoints, generated datasets, or local RAG indexes. Use `.env.example` for
documented placeholders only.

## Pull Requests

Each pull request should include:

- a short summary of user-visible behavior changes
- tests or a clear reason tests are not applicable
- reproduction notes for ML or data changes
- confirmation that generated artifacts were not committed
