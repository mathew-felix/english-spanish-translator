import os
import subprocess
import sys
import zipfile


def datasetDownload():
    """Download the Kaggle CSV and unzip it into ./data."""
    dataset   = "djonafegnem/europarl-parallel-corpus-19962011"
    file_name = "english_spanish.csv"
    output_dir = "./data"

    os.makedirs(output_dir, exist_ok=True)

    # Use the Kaggle CLI module from the active Python environment.
    command = [
        sys.executable, "-m", "kaggle.cli",
        "datasets", "download",
        "-d", dataset,
        "-f", file_name,
        "-p", output_dir,
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Dataset download failed with exit code {exc.returncode}.")
        return

    print(f"{file_name} downloaded to {output_dir}")
    datasetUnzip(output_dir, file_name)


def datasetUnzip(output_dir, file_name):
    """
    Extract the downloaded Kaggle archive into output_dir.
    Keeps the raw CSV beside the zip for later preprocessing.
    """
    zip_path = os.path.join(output_dir, f"{file_name}.zip")

    if os.path.exists(zip_path) and zipfile.is_zipfile(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
        print(f"File unzipped successfully to {output_dir}")
    else:
        print(f"The file '{zip_path}' does not exist or is not a valid zip file.")
