import re
import os
import unicodedata
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def get_unique_characters(column):
    """Return set of all unique characters across all non-null values in a Series."""
    unique_chars = set()
    for text in column.dropna():
        unique_chars.update(text)
    return unique_chars


def InspectDataset(file_path):
    """
    Inspect the dataset for basic statistics, missing values, duplicates,
    and unwanted characters.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Loaded dataset with {len(data)} rows and {len(data.columns)} columns.\n")
        print("Dataset Information:")
        print(data.info())
        print("\nMissing Values:")
        print(data.isnull().sum())

        duplicate_count = data.duplicated().sum()
        print(f"\nDuplicate Rows: {duplicate_count}")

        def contains_unwanted_characters(text):
            pattern = r"[^a-zA-Z0-9\sáéíóúüñÁÉÍÓÚÜÑ¿¡.,!?;'\"()\-\u2013\u2014\u2015]"
            return bool(re.search(pattern, text))

        tqdm.pandas(desc="Checking unwanted characters")
        data["English_Issues"] = data["English"].astype(str).progress_apply(contains_unwanted_characters)
        data["Spanish_Issues"] = data["Spanish"].astype(str).progress_apply(contains_unwanted_characters)

        unique_chars_en = get_unique_characters(data["English"])
        unique_chars_es = get_unique_characters(data["Spanish"])
        print("\nUnique characters in English column:", sorted(unique_chars_en))
        print("\nUnique characters in Spanish column:", sorted(unique_chars_es))

        issues = data[data["English_Issues"] | data["Spanish_Issues"]]
        print(f"\nRows with unwanted characters: {len(issues)}")
        if len(issues) > 0:
            print(issues.head())

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
    except Exception as exc:
        print(f"An unexpected error occurred: {exc}")


def CleanDataset(file_path, output_path="./data/cleaned_dataset.csv"):
    """
    Clean the dataset: drop NaN, remove duplicates, normalise text,
    and filter rows where both columns are empty after cleaning.
    """
    if not os.path.isfile(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    try:
        data = pd.read_csv(file_path)
        print(f"Loaded dataset with {len(data)} rows.")

        # Step 1: drop NaN
        data = data.dropna(subset=["English", "Spanish"])

        # Step 2: drop duplicates
        data = data.drop_duplicates()

        # Step 3: valid character set and cleaning
        # FIX Bug #11: closing ) was missing on set() call
        valid_chars = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "áéíóúüñÁÉÍÓÚÜÑ¿¡"
            " .,!?;'\"-()0123456789"
        )

        def clean_text(text):
            """Normalise Unicode, replace special punctuation, strip invalid chars."""
            text = unicodedata.normalize("NFKC", text)
            text = re.sub(r"\xa0", " ", text)
            text = re.sub(r"[–—―]", "-", text)
            text = re.sub(r"[‘’]", "'", text)
            text = re.sub(r"[“”]", '"', text)
            return "".join(c for c in text if c in valid_chars).strip()

        print("Cleaning text columns...")
        data["English"] = data["English"].astype(str).apply(clean_text)
        data["Spanish"] = data["Spanish"].astype(str).apply(clean_text)

        # FIX Bug #12: | (OR) → & (AND) — both columns must be non-empty
        data = data[
            (data["English"].str.strip() != "") &
            (data["Spanish"].str.strip() != "")
        ]
        print(f"Rows after cleaning: {len(data)}")

        # Step 5: save
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        data.to_csv(output_path, index=False)
        print(f"Cleaned dataset saved to '{output_path}' with {len(data)} rows.")

    except Exception as exc:
        print(f"An error occurred: {exc}")


def SmallDataset(input_csv, output_csv, percent):
    """
    Sample a fraction of the cleaned dataset into a smaller CSV.
    Returns True when the sampled file is written so later steps can continue.
    FIX Bug #13: caller should pass percent=0.1 not 1 (100%).
    """
    if not os.path.isfile(input_csv):
        print(f"Error: The file '{input_csv}' does not exist.")
        return False

    df = pd.read_csv(input_csv)
    sample_size = int(percent * len(df))
    df_sampled = df.sample(n=sample_size, random_state=42)
    df_sampled.to_csv(output_csv, index=False)
    print(f"Sampled {sample_size} rows ({percent*100:.0f}%) saved to {output_csv}")
    return True


def Split_data(file):
    """
    Split the sampled dataset into train.csv and test.csv (80/20).
    Returns True on success.
    """
    if not os.path.isfile(file):
        print(f"Error: The file '{file}' does not exist.")
        return False

    data = pd.read_csv(file)
    if "English" not in data.columns or "Spanish" not in data.columns:
        raise ValueError("Dataset must contain 'English' and 'Spanish' columns.")

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data.to_csv("./data/train.csv", index=False)
    test_data.to_csv("./data/test.csv", index=False)
    print(f"Split complete — train: {len(train_data)}, test: {len(test_data)}")
    return True


def DatasetPreprocessing():
    """
    End-to-end preprocessing pipeline. Stops early if a prerequisite file is missing.
    FIX Bug #14: existence check runs BEFORE InspectDataset to avoid silent failures.
    """
    raw_path     = "./data/english_spanish.csv"
    cleaned_path = "./data/cleaned_dataset.csv"
    sampled_path = "./data/small_dataset.csv"

    # FIX Bug #14: check before inspect
    if not os.path.isfile(raw_path):
        print(f"Error: raw dataset not found at '{raw_path}'. Run --step download first.")
        return

    InspectDataset(raw_path)
    CleanDataset(raw_path, cleaned_path)

    if not os.path.isfile(cleaned_path):
        return

    InspectDataset(cleaned_path)

    # FIX Bug #13: 0.1 = 10% sample (~190K pairs), not 1 = 100% (full 1.9M)
    if not SmallDataset(cleaned_path, sampled_path, 0.1):
        return

    Split_data(sampled_path)
