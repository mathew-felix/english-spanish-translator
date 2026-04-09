import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from source.inference import get_inference_engine, translate


class TranslationRequest(BaseModel):
    """Validate incoming translation requests.
    Text must be non-empty after stripping whitespace and capped at 500 characters.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=500)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value):
        """Reject empty or whitespace-only input text.
        The API should fail fast before touching the model.
        """
        if not value.strip():
            raise ValueError("Text must not be empty.")
        return value


class TranslationResponse(BaseModel):
    """Return the translated string and request timing.
    Latency is measured in milliseconds for simple local benchmarking.
    """

    model_config = ConfigDict(extra="forbid")

    input: str
    translation: str
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the API process starts.
    This avoids reloading the checkpoint on every `/translate` request.
    """
    get_inference_engine()
    yield


app = FastAPI(
    title="English-to-Spanish Translator API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Return a lightweight health response.
    This endpoint does not perform inference work per request.
    """
    return {"status": "ok"}


@app.post("/translate", response_model=TranslationResponse)
def translate_endpoint(payload: TranslationRequest):
    """Translate one English sentence through the cached model.
    Inference errors are surfaced as HTTP responses instead of raw tracebacks.
    """
    started_at = time.perf_counter()

    try:
        translated_text = translate(payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
    return TranslationResponse(
        input=payload.text,
        translation=translated_text,
        latency_ms=latency_ms,
    )
