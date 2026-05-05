"""Capture the institutional-review UI and render it as an animated GIF.
The GIF focuses on one process step at a time instead of whole-page states.
"""

import os
import tempfile
import time

import requests
from PIL import Image
from playwright.sync_api import sync_playwright

BASE_URL = os.getenv("UI_DEMO_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
OUTPUT_GIF = os.getenv("UI_DEMO_OUTPUT_GIF", os.path.join("assets", "demo.gif"))
CANVAS_SIZE = (1600, 1000)
VIEWPORT = {"width": 1660, "height": 1320}
FRAME_BACKGROUND = "#efe6da"


def _repo_root() -> str:
    """Return the repository root for stable output paths.
    The script may be run from any working directory.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _wait_for_app(timeout_seconds: int = 45) -> None:
    """Wait for the UI server to become reachable.
    Raises a clear error if nothing is listening on the configured base URL.
    """
    started_at = time.time()
    health_url = f"{BASE_URL}/health"
    while time.time() - started_at < timeout_seconds:
        try:
            response = requests.get(health_url, timeout=3)
            if response.status_code == 200:
                return
        except requests.RequestException:
            time.sleep(1)
            continue
        time.sleep(1)
    raise RuntimeError(
        f"FastAPI app is not reachable on {BASE_URL}. "
        "Start the app first with `docker compose up -d` or "
        "`venv/bin/python -m uvicorn src.serve:app --reload`."
    )


def _compose_frame(image_path: str) -> Image.Image:
    """Place one captured screenshot on a fixed-size background canvas.
    This keeps the GIF stable even when the captured content height changes.
    """
    canvas = Image.new("RGB", CANVAS_SIZE, FRAME_BACKGROUND)
    screenshot = Image.open(image_path).convert("RGB")

    max_width = CANVAS_SIZE[0] - 80
    max_height = CANVAS_SIZE[1] - 80
    scale = min(max_width / screenshot.width, max_height / screenshot.height, 1.0)
    resized = screenshot.resize(
        (int(screenshot.width * scale), int(screenshot.height * scale)),
        Image.Resampling.LANCZOS,
    )

    x_offset = (CANVAS_SIZE[0] - resized.width) // 2
    y_offset = (CANVAS_SIZE[1] - resized.height) // 2
    canvas.paste(resized, (x_offset, y_offset))
    return canvas.convert("P", palette=Image.ADAPTIVE)


def _save_locator_screenshot(page, selector: str, output_path: str) -> str:
    """Capture one focused panel from the live browser UI.
    The GIF should frame the active process instead of the entire page.
    """
    page.locator(selector).scroll_into_view_if_needed(timeout=10000)
    page.wait_for_timeout(500)
    page.locator(selector).screenshot(path=output_path)
    return output_path


def _capture_ui_states(output_dir: str) -> list[tuple[str, int]]:
    """Capture the step-by-step UI sequence for the review page.
    Each screenshot isolates the current process so the GIF reads as a sequence.
    """
    screenshots: list[tuple[str, int]] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
        page.goto(BASE_URL, wait_until="networkidle", timeout=120000)

        page.fill("#review-text", "The parliamentary session was adjourned.")
        screenshots.append(
            (
                _save_locator_screenshot(
                    page,
                    ".input-panel",
                    os.path.join(output_dir, "01-input.png"),
                ),
                3400,
            )
        )

        page.click("#review-form button[type='submit']")

        page.wait_for_function(
            "document.getElementById('step-draft').classList.contains('visible')",
            timeout=120000,
        )
        screenshots.append(
            (
                _save_locator_screenshot(
                    page,
                    ".steps-panel",
                    os.path.join(output_dir, "02-draft.png"),
                ),
                3600,
            )
        )

        page.wait_for_function(
            "document.getElementById('step-context').classList.contains('visible')",
            timeout=120000,
        )
        screenshots.append(
            (
                _save_locator_screenshot(
                    page,
                    ".steps-panel",
                    os.path.join(output_dir, "03-context.png"),
                ),
                4600,
            )
        )

        page.wait_for_function(
            "document.getElementById('step-decision').classList.contains('visible')",
            timeout=120000,
        )
        screenshots.append(
            (
                _save_locator_screenshot(
                    page,
                    ".steps-panel",
                    os.path.join(output_dir, "04-decision.png"),
                ),
                3600,
            )
        )

        page.wait_for_function(
            "document.getElementById('step-final').classList.contains('visible') && document.getElementById('final-output').textContent.length > 0",
            timeout=120000,
        )
        screenshots.append(
            (
                _save_locator_screenshot(
                    page,
                    ".steps-panel",
                    os.path.join(output_dir, "05-final.png"),
                ),
                5600,
            )
        )

        browser.close()

    return screenshots


def _render_gif(screenshots: list[tuple[str, int]], output_path: str) -> None:
    """Render the final GIF with slower holds for each walkthrough step.
    The result is intended to be readable without pausing on every frame.
    """
    frames = [_compose_frame(image_path) for image_path, _ in screenshots]
    durations = [duration for _, duration in screenshots]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=durations,
        optimize=True,
        disposal=2,
    )


def main() -> None:
    """Capture the real browser review process and save the GIF artifact.
    The application must already be running before this script starts.
    """
    repo_root = _repo_root()
    os.chdir(repo_root)
    _wait_for_app()

    with tempfile.TemporaryDirectory(prefix="ui-demo-") as temp_dir:
        screenshots = _capture_ui_states(temp_dir)
        output_path = os.path.join(repo_root, OUTPUT_GIF)
        _render_gif(screenshots, output_path)
        print(f"Saved UI demo GIF to {output_path}")


if __name__ == "__main__":
    main()
