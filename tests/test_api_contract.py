import importlib
import sys
import types

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def serve_module(monkeypatch):
    fake_agent_tools = types.ModuleType("agent.tools")
    fake_agent_tools.build_rag_translation_review = lambda text: {
        "input": text,
        "draft_translation": "borrador",
        "decision": "KEEP",
        "final_translation": "borrador",
        "retrieved_pairs": [
            {
                "english": "The session is open.",
                "spanish": "Se abre la sesion.",
                "distance": 0.123,
            }
        ],
    }

    fake_inference = types.ModuleType("source.inference")
    fake_inference.get_inference_engine = lambda: object()
    fake_inference.translate = lambda text: f"traducido: {text}"

    monkeypatch.setitem(sys.modules, "agent.tools", fake_agent_tools)
    monkeypatch.setitem(sys.modules, "source.inference", fake_inference)
    sys.modules.pop("src.serve", None)

    module = importlib.import_module("src.serve")
    yield module
    sys.modules.pop("src.serve", None)


def test_health_endpoint_is_lightweight(serve_module):
    with TestClient(serve_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_translate_endpoint_returns_contract(serve_module):
    with TestClient(serve_module.app) as client:
        response = client.post("/translate", json={"text": "Where is the station?"})

    payload = response.json()
    assert response.status_code == 200
    assert payload["input"] == "Where is the station?"
    assert payload["translation"] == "traducido: Where is the station?"
    assert isinstance(payload["latency_ms"], float)


def test_translate_endpoint_rejects_blank_text(serve_module):
    with TestClient(serve_module.app) as client:
        response = client.post("/translate", json={"text": "   "})

    assert response.status_code == 422


def test_institutional_review_endpoint_returns_context(serve_module):
    with TestClient(serve_module.app) as client:
        response = client.post(
            "/institutional-review",
            json={"text": "The parliamentary session is open."},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["decision"] == "KEEP"
    assert payload["final_translation"] == "borrador"
    assert payload["retrieved_examples"][0]["english"] == "The session is open."
