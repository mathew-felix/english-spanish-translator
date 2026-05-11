#!/bin/sh
set -eu

case "${TRANSLATOR_DEMO_MODE:-}" in
    1|true|TRUE|yes|YES|demo|DEMO)
        demo_mode=1
        ;;
    *)
        demo_mode=0
        ;;
esac

if [ "$demo_mode" = "0" ] && [ "${TRANSLATOR_AUTO_DOWNLOAD_MODEL:-1}" = "1" ]; then
    python scripts/download_model.py \
        --source "${MODEL_ARTIFACT_SOURCE:-github}" \
        --tag "${MODEL_RELEASE_TAG:-eng-sp-tranlate}"
fi

if [ "$demo_mode" = "0" ] && { [ ! -f "best_model.pth" ] || [ ! -d "data/tokenizer" ]; }; then
    echo "Model artifacts are missing. Run scripts/download_model.py or set TRANSLATOR_AUTO_DOWNLOAD_MODEL=1." >&2
    exit 1
fi

exec python -m uvicorn src.serve:app --host "${TRANSLATOR_HOST:-0.0.0.0}" --port "${TRANSLATOR_PORT:-8000}"
