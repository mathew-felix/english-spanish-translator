# Changelog

All notable changes to this project are tracked here.

## Unreleased

- Removed the hardcoded W&B key from the Colab notebook and documented environment
  variable configuration.
- Added production scaffolding: `.env.example`, dev requirements, pytest smoke
  tests, Ruff configuration, Makefile commands, Docker entrypoint, CI workflow,
  contribution guidance, and release/reproduction docs.
- Changed model downloads to default to the pinned `eng-sp-tranlate` release tag
  instead of `latest`.
- Updated Docker startup to download release model artifacts at runtime when they
  are not already present in the image.

## 1.0.0

- Custom PyTorch Transformer training pipeline for English-to-Spanish translation.
- FastAPI endpoints for direct translation and institutional review.
- RAG-based translation memory path with optional GPT revision.
- Reported 30-epoch run with 31.41 sacreBLEU.
