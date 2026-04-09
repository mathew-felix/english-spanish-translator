# FastAPI Inference Report

Updated: 2026-04-09

## Overview

The FastAPI inference layer has been implemented for the English-to-Spanish translator project.

The serving layer wraps the existing trained custom Transformer model and exposes two HTTP endpoints:

- `GET /health`
- `POST /translate`

The API is designed to load the model once at startup and reuse it for subsequent translation requests.

## Files Implemented

| Path | Purpose |
| --- | --- |
| `source/inference.py` | Loads tokenizer and checkpoint once, applies saved checkpoint config, and exposes `translate(text)` |
| `serve.py` | FastAPI application with startup lifecycle, request/response validation, `/health`, and `/translate` |
| `assets/swagger_demo.png` | Screenshot of the live Swagger UI generated from `/docs` |

## Dependencies Added

The following API-serving dependencies are now part of the project requirements:

- `fastapi==0.115.12`
- `uvicorn==0.34.0`
- `pydantic==2.11.3`

## Architecture Of The Inference Layer

### 1. Startup model loading

`source/inference.py` implements a cached singleton runtime:

- `_INFERENCE_ENGINE`
- `_INFERENCE_LOCK`
- `InferenceEngine`
- `get_inference_engine()`

This means:

- the model is loaded once per process
- the tokenizer is loaded once per process
- repeated API requests do not reload the checkpoint

This matches the intended FastAPI serving pattern for the project.

### 2. Checkpoint-aware config loading

The inference layer does not rely only on the default values in `source/Config.py`.

Instead, it:

- loads the saved checkpoint
- reads the saved `config` dict inside the checkpoint
- overlays those values onto the runtime `Config`
- rebuilds the model with the saved training settings

This is important because the trained checkpoint currently uses values different from the default config, including:

- `max_seq_length = 60`
- `batch_size = 640`
- `learning_rate = 4.5e-4`

Without this step, inference could fail with weight-shape mismatches.

### 3. Path resolution

The inference loader resolves repo-relative paths safely.

It supports:

- `best_model.pth` at repo root
- fallback to `weights/best_model.pth`
- tokenizer directory under `data/tokenizer/`

If required files are missing, it raises a clear `FileNotFoundError`.

### 4. Translation flow

For each translation request, the inference layer:

1. strips and validates the input text
2. lowercases the English input to match current dataset preprocessing
3. tokenizes the input with the saved tokenizer
4. adds `<SOS>` and `<END>`
5. pads to `config.max_seq_length`
6. runs model generation with `beam_width=4`
7. retries with `beam_width=2` if wide-beam decoding returns no tokens
8. decodes the output with the tokenizer
9. cleans tokenizer spacing artifacts such as `¿ Dónde` into `¿Dónde`

## FastAPI App Details

`serve.py` implements the REST API.

### App lifecycle

The app uses a FastAPI lifespan handler:

- `get_inference_engine()` is called during startup
- the model is ready before the first request

### Request model

`TranslationRequest` uses Pydantic v2:

- field: `text: str`
- min length: `1`
- max length: `500`
- extra fields are forbidden
- whitespace-only input is rejected

### Response model

`TranslationResponse` returns:

- `input`
- `translation`
- `latency_ms`

### Error handling

The API maps failures into HTTP responses:

- `400` for invalid input such as empty text
- `500` for model/tokenizer/checkpoint runtime errors

## Endpoint Inventory

### `GET /health`

Purpose:

- lightweight service health response

Current response:

```json
{
  "status": "ok"
}
```

### `POST /translate`

Purpose:

- translate one English sentence into Spanish

Request body:

```json
{
  "text": "Where is the nearest hospital?"
}
```

Observed local response:

```json
{
  "input": "Where is the nearest hospital?",
  "translation": "¿Dónde está el hospital más cercano?",
  "latency_ms": 21467.58
}
```

## Swagger UI

The FastAPI docs are available at:

- `/docs`

A screenshot of the generated Swagger UI has been saved at:

- `assets/swagger_demo.png`

## Commands Used

Run the API locally:

```bash
venv/bin/uvicorn serve:app --reload
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

## Verification Performed

The FastAPI inference layer was verified locally with the current repository state after the exported Colab artifacts were restored into the paths used by the runtime config.

Verified items:

- `serve.py` compiles
- `source/inference.py` compiles
- `GET /health` returns `{"status":"ok"}`
- `POST /translate` returns a valid JSON response with translation output
- `/openapi.json` exposes `/health` and `/translate`
- `/docs` loads successfully
- Swagger screenshot was captured and saved to `assets/swagger_demo.png`

## Current Behavior Notes

### 1. Model loading strategy

The model is loaded once at startup, not per request.

This is correct for performance and avoids repeated checkpoint deserialization.

### 2. CPU latency

The measured local response time for the tested request was about `21467.58 ms`.

That reflects the current local runtime conditions and checkpoint size. It is not a guaranteed fixed latency.

### 3. Input validation

Whitespace-only input is rejected before inference.

Example invalid request body:

```json
{
  "text": "   "
}
```

This produces a validation error instead of running the model.

## What This Phase Completed

This FastAPI phase is now functionally implemented:

- inference module created
- model singleton loader implemented
- checkpoint-aware runtime config handling implemented
- REST API app created
- `/health` endpoint created
- `/translate` endpoint created
- request/response validation added
- latency reporting added
- Swagger UI screenshot generated
- README API section updated

## Remaining Caveats

The serving layer is implemented, but a few surrounding project items are still outside this phase:

- there is no Docker packaging yet
- there is no deployment configuration yet
- there is no batching or async queue for high-throughput inference
- the current README still contains older non-API sections that do not fully match the latest repo state

## Bottom Line

The project now has a working FastAPI inference layer over the trained custom Transformer checkpoint.

It can be started with Uvicorn, responds correctly on `/health`, translates on `/translate`, validates request payloads with Pydantic v2, reports latency, and exposes live Swagger docs with a saved screenshot.
