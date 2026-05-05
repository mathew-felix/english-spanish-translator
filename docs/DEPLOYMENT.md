# Deployment

## Local API

Download the pinned model artifacts and start Uvicorn:

```bash
python scripts/download_model.py --tag eng-sp-tranlate
python -m uvicorn src.serve:app --host 127.0.0.1 --port 8000
```

Check readiness:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

## Docker

The Docker image does not require local model files at build time. At container
startup, `scripts/docker-entrypoint.sh` downloads the pinned release artifacts
when `TRANSLATOR_AUTO_DOWNLOAD_MODEL=1`.

```bash
docker compose up --build
```

Set `TRANSLATOR_AUTO_DOWNLOAD_MODEL=0` only when `best_model.pth` and
`data/tokenizer/` are already present inside the image or mounted volume.

## Configuration

Use environment variables rather than editing source code:

- `MODEL_RELEASE_OWNER`
- `MODEL_RELEASE_REPO`
- `MODEL_RELEASE_TAG`
- `OPENAI_API_KEY`
- `TRANSLATOR_API_BASE_URL`
- `TRANSLATOR_AUTO_DOWNLOAD_MODEL`
- `TRANSLATOR_HOST`
- `TRANSLATOR_PORT`
- `WANDB_API_KEY`
- `WANDB_MODE`
- `WANDB_PROJECT`

## Production Notes

- Keep `.env` out of git.
- Rotate the previously exposed W&B key before publishing this repository.
- Purge git history containing the old notebook secret before pushing to a
  shared remote.
- Build the RAG index with `python rag/build_index.py` only after
  `data/train.csv` exists.
