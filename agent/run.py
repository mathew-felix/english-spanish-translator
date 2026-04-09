"""Smoke-test runner for the focused LangGraph translation router.
It validates routing between direct translation and institutional review.
"""

import os
import subprocess
import sys
import time
from typing import Optional
from urllib.parse import urlparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import requests
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import build_graph
from agent.tools import get_api_base_url, load_local_env


TEST_CASES = [
    ("Translate 'I need a doctor' to Spanish", "translate_with_custom_model"),
    ("How do you say 'the train is late'?", "translate_with_custom_model"),
    ("Translate 'The parliamentary session was adjourned.' to Spanish", "rag_translate"),
    ("Translate 'The committee approved the amendment.' to Spanish", "rag_translate"),
]


def _health_url() -> str:
    """Return the FastAPI health endpoint URL for agent smoke tests.
    The base URL is shared with the translation tool configuration.
    """
    return f"{get_api_base_url()}/health"


def _api_is_healthy() -> bool:
    """Check whether the FastAPI translation service is already reachable.
    This avoids spawning a duplicate local server when one is already running.
    """
    try:
        response = requests.get(_health_url(), timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def _start_local_api_if_needed() -> Optional[subprocess.Popen]:
    """Start the FastAPI server for translation tool tests when needed.
    Auto-start only works for localhost-style URLs backed by the current repo.
    """
    if _api_is_healthy():
        return None

    parsed_url = urlparse(get_api_base_url())
    host = parsed_url.hostname or "127.0.0.1"
    port = parsed_url.port or 8000
    if host not in {"127.0.0.1", "localhost"}:
        raise RuntimeError(
            "Translation API is not reachable and TRANSLATOR_API_BASE_URL is not local."
        )

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "serve:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.time() + 60
    while time.time() < deadline:
        if _api_is_healthy():
            return process
        if process.poll() is not None:
            raise RuntimeError("Local FastAPI server exited before becoming healthy.")
        time.sleep(1)

    process.terminate()
    raise RuntimeError("Timed out waiting for the FastAPI server to become healthy.")


def _cleanup_process(process: Optional[subprocess.Popen]) -> None:
    """Stop the temporary FastAPI server started by this runner.
    Existing external servers are left untouched because `process` will be `None`.
    """
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _selected_tool_name(messages: list) -> str:
    """Extract the first tool name selected by the agent.
    The first model message with `tool_calls` defines the route for the query.
    """
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            return message.tool_calls[0]["name"]
    return ""


def main() -> None:
    """Run the focused routing checks through the compiled graph.
    The script exits with a non-zero status if any query chooses the wrong tool.
    """
    load_local_env()
    server_process = _start_local_api_if_needed()

    try:
        graph = build_graph()
        failures = []

        for query, expected_tool in TEST_CASES:
            result = graph.invoke({"messages": [HumanMessage(content=query)]})
            messages = result["messages"]
            actual_tool = _selected_tool_name(messages)
            final_response = messages[-1].content

            print(f"Query: {query}")
            print(f"Expected tool: {expected_tool}")
            print(f"Actual tool:   {actual_tool}")
            print(f"Response:      {final_response}")
            print()

            if actual_tool != expected_tool:
                failures.append((query, expected_tool, actual_tool))

        if failures:
            raise SystemExit(f"Routing failures detected: {failures}")

        print("All agent routes matched the expected tools.")
    finally:
        _cleanup_process(server_process)


if __name__ == "__main__":
    main()
