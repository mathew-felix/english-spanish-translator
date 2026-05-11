"""Prepare a minimal Hugging Face Space upload folder.

The Space should not receive generated model artifacts from the git checkout.
Full-model mode downloads `best_model.pth` and tokenizer files from the
configured Hugging Face model repo at startup.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "hf-space"

FILES_TO_COPY = [
    "app.py",
    ".gitattributes",
]

DIRECTORIES_TO_COPY = [
    "source",
    "src",
    "scripts",
    "rag",
]

EXCLUDED_DIRECTORY_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "chroma_db",
}

EXCLUDED_FILE_SUFFIXES = {
    ".pyc",
    ".pyo",
}

SPACE_REQUIREMENTS = [
    "fastapi==0.136.1",
    "Jinja2==3.1.6",
    "pydantic==2.11.3",
    "requests==2.33.1",
    "huggingface-hub==1.14.0",
    "torch==2.11.0",
    "transformers==5.7.0",
    "uvicorn==0.34.0",
    "gradio==5.50.0",
    "pillow==11.3.0",
]


def _ignore_generated_files(directory: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        path = Path(directory) / name
        if name in EXCLUDED_DIRECTORY_NAMES:
            ignored.add(name)
        elif path.is_file() and path.suffix in EXCLUDED_FILE_SUFFIXES:
            ignored.add(name)
    return ignored


def _copy_file(relative_path: str, output_dir: Path) -> None:
    source_path = REPO_ROOT / relative_path
    if not source_path.exists():
        return
    destination_path = output_dir / relative_path
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)


def _copy_directory(relative_path: str, output_dir: Path) -> None:
    source_path = REPO_ROOT / relative_path
    if not source_path.exists():
        return
    destination_path = output_dir / relative_path
    shutil.copytree(
        source_path,
        destination_path,
        ignore=_ignore_generated_files,
        dirs_exist_ok=True,
    )


def prepare_space(output_dir: Path, force: bool = False) -> Path:
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"{output_dir} already exists. Re-run with --force to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)
    for relative_path in FILES_TO_COPY:
        _copy_file(relative_path, output_dir)
    for relative_path in DIRECTORIES_TO_COPY:
        _copy_directory(relative_path, output_dir)

    (output_dir / "requirements.txt").write_text(
        "\n".join(SPACE_REQUIREMENTS) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "---\n"
        "title: Institutional Translation Review\n"
        "emoji: 🌐\n"
        "colorFrom: blue\n"
        "colorTo: green\n"
        "sdk: gradio\n"
        "sdk_version: 5.50.0\n"
        "app_file: app.py\n"
        "pinned: false\n"
        "---\n\n"
        "# Institutional Translation Review Space\n\n"
        "This Space runs the Gradio hosted demo for the English-Spanish "
        "institutional translation review project. Full-model mode downloads "
        "artifacts from `mathew-felix/en-es-nmt-transformer`. The UI can use "
        "`TRANSLATOR_DEMO_MEMORY=1` to show curated bilingual evidence when a "
        "full Chroma translation-memory index is not available.\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Hugging Face Space files.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the Space upload bundle should be written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output directory.",
    )
    args = parser.parse_args()

    output_dir = prepare_space(Path(args.output_dir), force=args.force)
    print(f"Prepared Hugging Face Space files at {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
