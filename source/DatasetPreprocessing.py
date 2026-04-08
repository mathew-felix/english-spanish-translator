import csv
import hashlib
import html
import os
import random
import re
import unicodedata

import pandas as pd


RAW_DATA_DIR = os.path.join(".", "data", "raw")
MERGED_DATASET_PATH = os.path.join(".", "data", "english_spanish.csv")
TRAIN_DATASET_PATH = os.path.join(".", "data", "train.csv")
TEST_DATASET_PATH = os.path.join(".", "data", "test.csv")
INSPECT_SAMPLE_ROWS = 50000
MAX_SEQUENCE_TOKENS = 80
MAX_LENGTH_RATIO = 3.0
OPEN_SUBTITLES_MAX_ROWS = 2_000_000
CORPUS_LIMITS = {
    "Europarl": None,
    "News-Commentary": None,
    "TED2020": None,
    "OpenSubtitles": OPEN_SUBTITLES_MAX_ROWS,
}

VALID_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "áéíóúüñÁÉÍÓÚÜÑ¿¡"
    " .,!?;:'\"-()0123456789"
)


def get_unique_characters(column):
    """Return the distinct characters present in a pandas Series.
    Null values are ignored so inspection works on partially cleaned data.
    """
    unique_chars = set()
    for text in column.dropna():
        unique_chars.update(text)
    return unique_chars


def InspectDataset(file_path, sample_rows=INSPECT_SAMPLE_ROWS):
    """Inspect a bounded sample of the merged bilingual dataset.
    Sampling keeps the report fast even when the underlying corpus is very large.
    """
    if not os.path.isfile(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return

    try:
        data = pd.read_csv(file_path, nrows=sample_rows)
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty.")
        return
    except OSError as exc:
        print(f"Error reading '{file_path}': {exc}")
        return

    print(
        f"Loaded inspection sample with {len(data)} rows and "
        f"{len(data.columns)} columns from '{file_path}'.\n"
    )
    print("Dataset Information:")
    print(data.info())
    print("\nMissing Values:")
    print(data.isnull().sum())

    duplicate_count = data.duplicated(subset=["English", "Spanish"]).sum()
    print(f"\nDuplicate Rows in sample: {duplicate_count}")

    def contains_unwanted_characters(text):
        pattern = r"[^a-zA-Z0-9\sáéíóúüñÁÉÍÓÚÜÑ¿¡.,!?;:'\"()\-]"
        return bool(re.search(pattern, text))

    data["English_Issues"] = data["English"].astype(str).apply(contains_unwanted_characters)
    data["Spanish_Issues"] = data["Spanish"].astype(str).apply(contains_unwanted_characters)

    unique_chars_en = get_unique_characters(data["English"])
    unique_chars_es = get_unique_characters(data["Spanish"])
    print("\nUnique characters in English column:", sorted(unique_chars_en))
    print("\nUnique characters in Spanish column:", sorted(unique_chars_es))

    issues = data[data["English_Issues"] | data["Spanish_Issues"]]
    print(f"\nRows with unwanted characters in sample: {len(issues)}")
    if not issues.empty:
        print(issues.head())


def _find_parallel_files(corpus_dir, source_lang="en", target_lang="es"):
    """Locate the extracted source and target text files for one OPUS corpus.
    The expected format is the moses layout with matching `.en` and `.es` files.
    """
    source_files = [
        name for name in os.listdir(corpus_dir) if name.endswith(f".{source_lang}")
    ]
    target_files = set(
        name for name in os.listdir(corpus_dir) if name.endswith(f".{target_lang}")
    )

    for source_name in sorted(source_files):
        base_name = source_name[: -(len(source_lang) + 1)]
        target_name = f"{base_name}.{target_lang}"
        if target_name in target_files:
            return (
                os.path.join(corpus_dir, source_name),
                os.path.join(corpus_dir, target_name),
            )

    raise FileNotFoundError(
        f"Could not find matching '.{source_lang}' and '.{target_lang}' files in "
        f"'{corpus_dir}'."
    )


def _normalise_text(text):
    """Normalise punctuation and whitespace for one sentence.
    HTML tags are stripped before invalid characters are removed.
    """
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\xa0", " ", text)
    text = re.sub(r"[–—―]", "-", text)
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    text = "".join(char for char in text if char in VALID_CHARS)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _has_enough_language_content(text):
    """Check that a sentence contains enough alphabetic content.
    This removes empty, punctuation-only, and near-empty fragments.
    """
    return sum(char.isalpha() for char in text) >= 2


def _passes_length_filters(english_text, spanish_text):
    """Filter sentence pairs by token count and length ratio.
    Large imbalances are strong indicators of noisy alignment.
    """
    english_tokens = len(english_text.split())
    spanish_tokens = len(spanish_text.split())

    if not (1 <= english_tokens <= MAX_SEQUENCE_TOKENS):
        return False
    if not (1 <= spanish_tokens <= MAX_SEQUENCE_TOKENS):
        return False

    shorter = min(english_tokens, spanish_tokens)
    longer = max(english_tokens, spanish_tokens)
    if shorter == 0:
        return False
    return (longer / shorter) <= MAX_LENGTH_RATIO


def _looks_like_subtitle_noise(raw_text, cleaned_text):
    """Detect subtitle-specific artifacts before keeping a line.
    Speaker labels, URLs, and stage directions are dropped.
    """
    stripped = raw_text.strip()
    if not stripped:
        return True

    if re.search(r"https?://|www\.", stripped, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"[\[\(].*[\]\)]", stripped):
        return True
    if re.match(r"^[A-Z][A-Z\s\-]{1,30}:\s", stripped):
        return True
    if re.search(r"\b(subtitles?|opensubtitles)\b", stripped, flags=re.IGNORECASE):
        return True
    if not _has_enough_language_content(cleaned_text):
        return True
    return False


def _passes_pair_filters(corpus_name, raw_english, raw_spanish):
    """Clean and validate one bilingual sentence pair.
    Returns the cleaned pair when it passes corpus-aware quality checks.
    """
    english_text = _normalise_text(raw_english)
    spanish_text = _normalise_text(raw_spanish)

    if not english_text or not spanish_text:
        return None
    if not _has_enough_language_content(english_text):
        return None
    if not _has_enough_language_content(spanish_text):
        return None
    if not _passes_length_filters(english_text, spanish_text):
        return None

    if corpus_name == "OpenSubtitles":
        if _looks_like_subtitle_noise(raw_english, english_text):
            return None
        if _looks_like_subtitle_noise(raw_spanish, spanish_text):
            return None

    return english_text, spanish_text


def _pair_digest(english_text, spanish_text):
    """Create a compact digest for de-duplicating sentence pairs.
    Hashing keeps the global seen-set smaller than storing raw strings.
    """
    payload = f"{english_text}\t{spanish_text}".encode("utf-8")
    return hashlib.sha1(payload).digest()


def BuildCombinedDataset(raw_data_dir, output_path):
    """Merge the selected OPUS corpora into one bilingual CSV.
    OpenSubtitles is filtered more aggressively and capped so it does not dominate.
    """
    if not os.path.isdir(raw_data_dir):
        raise FileNotFoundError(
            f"Raw data directory '{raw_data_dir}' was not found. Run --step download first."
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    seen_pairs = set()

    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=["English", "Spanish", "Corpus"]
        )
        writer.writeheader()

        for corpus_name, max_rows in CORPUS_LIMITS.items():
            corpus_dir = os.path.join(raw_data_dir, corpus_name)
            if not os.path.isdir(corpus_dir):
                raise FileNotFoundError(
                    f"Expected extracted corpus directory '{corpus_dir}' was not found."
                )

            english_path, spanish_path = _find_parallel_files(corpus_dir)
            total_rows = 0
            kept_rows = 0

            with open(english_path, "r", encoding="utf-8", errors="replace") as english_file, \
                open(spanish_path, "r", encoding="utf-8", errors="replace") as spanish_file:
                for raw_english, raw_spanish in zip(english_file, spanish_file):
                    total_rows += 1
                    cleaned_pair = _passes_pair_filters(
                        corpus_name, raw_english.rstrip("\n"), raw_spanish.rstrip("\n")
                    )
                    if cleaned_pair is None:
                        continue

                    english_text, spanish_text = cleaned_pair
                    pair_key = _pair_digest(english_text, spanish_text)
                    if pair_key in seen_pairs:
                        continue

                    writer.writerow(
                        {
                            "English": english_text,
                            "Spanish": spanish_text,
                            "Corpus": corpus_name,
                        }
                    )
                    seen_pairs.add(pair_key)
                    kept_rows += 1

                    if max_rows is not None and kept_rows >= max_rows:
                        break

            print(
                f"{corpus_name}: kept {kept_rows} pairs from {total_rows} processed rows."
            )

    print(f"Merged dataset written to '{output_path}'.")


def SmallDataset(input_csv, output_csv, percent):
    """Sample a fraction of a cleaned dataset into a smaller CSV.
    This helper is intended for smoke tests, not the main training pipeline.
    """
    if not os.path.isfile(input_csv):
        print(f"Error: The file '{input_csv}' does not exist.")
        return False

    df = pd.read_csv(input_csv)
    sample_size = max(1, int(percent * len(df)))
    df_sampled = df.sample(n=sample_size, random_state=42)
    df_sampled.to_csv(output_csv, index=False)
    print(f"Sampled {sample_size} rows ({percent * 100:.0f}%) saved to {output_csv}")
    return True


def Split_data(file_path, test_size=0.2, seed=42):
    """Split a merged CSV into train and test files without loading it fully.
    Streaming keeps preprocessing practical on multi-million-row corpora.
    """
    if not os.path.isfile(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return False

    rng = random.Random(seed)
    train_rows = 0
    test_rows = 0

    with open(file_path, "r", newline="", encoding="utf-8") as input_file, \
        open(TRAIN_DATASET_PATH, "w", newline="", encoding="utf-8") as train_file, \
        open(TEST_DATASET_PATH, "w", newline="", encoding="utf-8") as test_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError("Dataset must contain a CSV header.")

        train_writer = csv.DictWriter(train_file, fieldnames=reader.fieldnames)
        test_writer = csv.DictWriter(test_file, fieldnames=reader.fieldnames)
        train_writer.writeheader()
        test_writer.writeheader()

        for row in reader:
            if rng.random() < test_size:
                test_writer.writerow(row)
                test_rows += 1
            else:
                train_writer.writerow(row)
                train_rows += 1

    print(f"Split complete — train: {train_rows}, test: {test_rows}")
    return True


def DatasetPreprocessing():
    """Build, inspect, and split the English-Spanish corpus mix for training.
    The merged dataset uses Europarl, News Commentary, TED2020, and filtered OpenSubtitles.
    """
    try:
        BuildCombinedDataset(RAW_DATA_DIR, MERGED_DATASET_PATH)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Dataset preprocessing failed: {exc}")
        return

    InspectDataset(MERGED_DATASET_PATH)
    Split_data(MERGED_DATASET_PATH)
