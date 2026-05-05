#!/bin/sh
set -eu

if [ "${TRANSLATOR_AUTO_DOWNLOAD_MODEL:-1}" = "1" ]; then
    python scripts/download_model.py --tag "${MODEL_RELEASE_TAG:-eng-sp-tranlate}"
fi

if [ ! -f "best_model.pth" ] || [ ! -d "data/tokenizer" ]; then
    echo "Model artifacts are missing. Run scripts/download_model.py or set TRANSLATOR_AUTO_DOWNLOAD_MODEL=1." >&2
    exit 1
fi

exec python -m uvicorn src.serve:app --host "${TRANSLATOR_HOST:-0.0.0.0}" --port "${TRANSLATOR_PORT:-8000}"
