import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field, field_validator

from source.inference import get_inference_engine, translate
from src.env import load_local_env
from src.review import build_institutional_review

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATES = Jinja2Templates(directory=os.path.join(PROJECT_ROOT, "templates"))
logger = logging.getLogger(__name__)


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


class ReviewContextExample(BaseModel):
    """Return one retrieved bilingual example used during review.
    Distances are included so the UI can show why an example was surfaced.
    """

    model_config = ConfigDict(extra="forbid")

    english: str
    spanish: str
    distance: float


class InstitutionalReviewResponse(BaseModel):
    """Return the staged institutional-translation review output.
    The response is structured so the UI can reveal each step separately.
    """

    model_config = ConfigDict(extra="forbid")

    input: str
    draft_translation: str
    decision: str
    final_translation: str
    retrieved_examples: list[ReviewContextExample]
    context_status: str
    context_message: str
    reviewer_status: str
    reviewer_explanation: str
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the API process starts.
    This avoids reloading the checkpoint on every `/translate` request.
    """
    load_local_env()
    get_inference_engine()
    yield


app = FastAPI(
    title="Institutional Translation Review API",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount(
    "/assets",
    StaticFiles(directory=os.path.join(PROJECT_ROOT, "assets")),
    name="assets",
)


@app.middleware("http")
async def log_request_summary(request: Request, call_next):
    """Log route-level request telemetry without recording input text."""
    started_at = time.perf_counter()
    status_code = 500
    error_class = "-"

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as exc:
        error_class = type(exc).__name__
        raise
    finally:
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.info(
            "request_complete method=%s path=%s status_code=%s latency_ms=%s error_class=%s",
            request.method,
            request.url.path,
            status_code,
            latency_ms,
            error_class,
        )


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Render the browser UI for the translator demo.
    The page focuses on institutional translation review as the main application path.
    """
    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {},
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


@app.post("/institutional-review", response_model=InstitutionalReviewResponse)
def institutional_review_endpoint(payload: TranslationRequest):
    """Run the institutional translation-review process.
    This endpoint exposes the custom draft, retrieval context, and review decision separately.
    """
    started_at = time.perf_counter()

    try:
        review_result = build_institutional_review(payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
    return InstitutionalReviewResponse(
        input=review_result["input"],
        draft_translation=review_result["draft_translation"],
        decision=review_result["decision"],
        final_translation=review_result["final_translation"],
        retrieved_examples=[
            ReviewContextExample(
                english=pair["english"],
                spanish=pair["spanish"],
                distance=pair["distance"],
            )
            for pair in review_result["retrieved_pairs"]
        ],
        context_status=review_result["context_status"],
        context_message=review_result["context_message"],
        reviewer_status=review_result["reviewer_status"],
        reviewer_explanation=review_result["reviewer_explanation"],
        latency_ms=latency_ms,
    )
