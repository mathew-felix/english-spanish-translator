import argparse
import csv
import json
import os
import sys
import time

import torch
from transformers import MarianMTModel, MarianTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from source.Config import Config
from source.inference import get_inference_engine


DEFAULT_MODEL_NAME = "Helsinki-NLP/opus-mt-en-es"


def load_test_rows(csv_path, limit):
    """Load a fixed held-out slice for model comparison.
    Rows are read in file order so the custom and HF models see the exact same inputs.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            rows.append(
                {
                    "index": index,
                    "source": row["English"],
                    "reference": row["Spanish"],
                    "corpus": row.get("Corpus", ""),
                }
            )
            if len(rows) >= limit:
                break
    return rows


def normalise_text(text):
    """Collapse whitespace in generated text for stable JSON output.
    This keeps comparison output readable without changing the model prediction itself.
    """
    return " ".join(text.split()).strip()


def translate_with_custom(engine, text):
    """Run one sentence through the cached custom-model runtime.
    Timing excludes model load because the engine is initialised once up front.
    """
    started_at = time.perf_counter()
    translated_text = engine.translate(text)
    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
    return normalise_text(translated_text), latency_ms


def load_hf_runtime(model_name, device):
    """Load the Marian tokenizer and model on the chosen device.
    The model is reused across all sentences to keep timing comparable.
    """
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return tokenizer, model


def translate_with_hf(tokenizer, model, text, device):
    """Run one sentence through the pretrained Marian baseline.
    Generation uses beam search to reflect the pretrained model's normal inference quality.
    """
    batch = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    batch = {key: value.to(device) for key, value in batch.items()}

    started_at = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            **batch,
            num_beams=4,
            max_new_tokens=128,
        )
    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
    translated_text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return normalise_text(translated_text), latency_ms


def build_results(rows, model_name):
    """Generate JSON-ready comparison outputs for both models.
    The same input rows are used for the custom model and the Marian baseline.
    """
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    custom_engine = get_inference_engine()
    hf_tokenizer, hf_model = load_hf_runtime(model_name, device)

    custom_results = []
    baseline_results = []

    for row in rows:
        custom_translation, custom_latency_ms = translate_with_custom(
            custom_engine,
            row["source"],
        )
        hf_translation, hf_latency_ms = translate_with_hf(
            hf_tokenizer,
            hf_model,
            row["source"],
            device,
        )

        custom_results.append(
            {
                "index": row["index"],
                "source": row["source"],
                "reference": row["reference"],
                "corpus": row["corpus"],
                "translation": custom_translation,
                "latency_ms": custom_latency_ms,
                "model": "custom_transformer",
            }
        )
        baseline_results.append(
            {
                "index": row["index"],
                "source": row["source"],
                "reference": row["reference"],
                "corpus": row["corpus"],
                "translation": hf_translation,
                "latency_ms": hf_latency_ms,
                "model": model_name,
            }
        )

    metadata = {
        "source_csv": "",
        "num_sentences": len(rows),
        "device": str(device),
        "baseline_model": model_name,
    }
    return metadata, custom_results, baseline_results


def average_latency(results):
    """Compute mean latency across one model's comparison outputs.
    Returns a rounded millisecond value for README-friendly reporting.
    """
    if not results:
        return 0.0
    total_latency = sum(item["latency_ms"] for item in results)
    return round(total_latency / len(results), 2)


def save_results(path, metadata, results):
    """Write one comparison result file to JSON.
    The output includes metadata and a simple average latency summary.
    """
    payload = {
        "metadata": metadata,
        "summary": {
            "average_latency_ms": average_latency(results),
            "num_sentences": len(results),
        },
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main():
    """Generate Marian baseline and custom-model comparison outputs.
    Results are saved under `finetune/` for README comparison and interview discussion.
    """
    parser = argparse.ArgumentParser(
        description="Run a 50-sentence comparison between the custom model and MarianMT."
    )
    parser.add_argument("--limit", type=int, default=50, help="Number of test rows to compare.")
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name for the pretrained baseline.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Optional CSV path for the shared comparison set.",
    )
    parser.add_argument(
        "--custom-output",
        type=str,
        default="custom_model_results.json",
        help="Output JSON file for custom-model results.",
    )
    parser.add_argument(
        "--baseline-output",
        type=str,
        default="baseline_results.json",
        help="Output JSON file for Hugging Face baseline results.",
    )
    args = parser.parse_args()

    repo_root = REPO_ROOT
    finetune_dir = os.path.join(repo_root, "finetune")
    os.makedirs(finetune_dir, exist_ok=True)

    config = Config()
    if args.csv_path:
        if os.path.isabs(args.csv_path):
            test_csv_path = args.csv_path
        else:
            test_csv_path = os.path.join(repo_root, args.csv_path.lstrip("./"))
    else:
        test_csv_path = os.path.join(repo_root, config.test_csv.lstrip("./"))
    rows = load_test_rows(test_csv_path, args.limit)
    metadata, custom_results, baseline_results = build_results(rows, args.model_name)
    metadata["source_csv"] = test_csv_path

    save_results(
        os.path.join(finetune_dir, args.custom_output),
        metadata,
        custom_results,
    )
    save_results(
        os.path.join(finetune_dir, args.baseline_output),
        metadata,
        baseline_results,
    )

    print(f"Saved {len(custom_results)} custom-model results.")
    print(f"Saved {len(baseline_results)} baseline results.")
    print(
        "Average latency (ms): "
        f"custom={average_latency(custom_results)}, "
        f"baseline={average_latency(baseline_results)}"
    )


if __name__ == "__main__":
    main()
