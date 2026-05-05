import json
import os
import shutil
import ssl
import urllib.parse
import urllib.request
import zipfile

import certifi

OPUS_API_URL = "https://opus.nlpl.eu/opusapi/"
OPUS_CORPORA = (
    "Europarl",
    "News-Commentary",
    "TED2020",
    "OpenSubtitles",
)


def _create_ssl_context():
    """Create an SSL context backed by certifi CA roots.
    This avoids local certificate-store issues when calling OPUS from Python.
    """
    return ssl.create_default_context(cafile=certifi.where())


def _fetch_opus_metadata(corpus, source_lang="en", target_lang="es"):
    """Fetch the latest OPUS metadata for one language pair.
    Returns the moses-preprocessed corpus descriptor used for download.
    """
    query = urllib.parse.urlencode(
        {
            "corpus": corpus,
            "source": source_lang,
            "target": target_lang,
            "preprocessing": "moses",
            "version": "latest",
        }
    )
    api_url = f"{OPUS_API_URL}?{query}"
    with urllib.request.urlopen(
        api_url, timeout=30, context=_create_ssl_context()
    ) as response:
        payload = json.load(response)

    corpora = payload.get("corpora", [])
    if not corpora:
        raise ValueError(
            f"No OPUS metadata returned for corpus={corpus}, "
            f"source={source_lang}, target={target_lang}."
        )
    return corpora[0]


def _download_file(url, output_path):
    """Download one corpus archive to disk.
    The file is streamed so large corpora do not sit fully in memory.
    """
    with urllib.request.urlopen(
        url, timeout=120, context=_create_ssl_context()
    ) as response, open(output_path, "wb") as output_file:
        shutil.copyfileobj(response, output_file)


def _extract_archive(zip_path, extract_dir):
    """Extract a downloaded OPUS zip into its corpus directory.
    Existing extracted files are preserved unless the directory is empty.
    """
    os.makedirs(extract_dir, exist_ok=True)
    if os.listdir(extract_dir):
        print(f"Using existing extracted files in '{extract_dir}'.")
        return

    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"The file '{zip_path}' is not a valid zip archive.")

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)
    print(f"Extracted '{zip_path}' to '{extract_dir}'.")


def datasetDownload():
    """Download the English-Spanish corpora used by the training pipeline.
    Fetches Europarl, News Commentary, TED2020, and OpenSubtitles from OPUS.
    """
    raw_data_dir = os.path.join(".", "data", "raw")
    os.makedirs(raw_data_dir, exist_ok=True)

    for corpus in OPUS_CORPORA:
        try:
            metadata = _fetch_opus_metadata(corpus)
            corpus_url = metadata["url"]
            corpus_version = metadata["version"]
            pair_count = metadata["alignment_pairs"]
        except (
            KeyError,
            TimeoutError,
            urllib.error.HTTPError,
            urllib.error.URLError,
            ValueError,
        ) as exc:
            print(f"Failed to fetch metadata for '{corpus}': {exc}")
            return

        zip_path = os.path.join(raw_data_dir, f"{corpus}.zip")
        extract_dir = os.path.join(raw_data_dir, corpus)

        print(
            f"Preparing {corpus} ({corpus_version}) with {pair_count} aligned pairs."
        )
        if not os.path.isfile(zip_path):
            try:
                _download_file(corpus_url, zip_path)
            except (
                TimeoutError,
                urllib.error.HTTPError,
                urllib.error.URLError,
            ) as exc:
                print(f"Dataset download failed for '{corpus}': {exc}")
                return
            print(f"Downloaded '{corpus}' to '{zip_path}'.")
        else:
            print(f"Using existing archive '{zip_path}'.")

        try:
            _extract_archive(zip_path, extract_dir)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            print(f"Failed to extract '{corpus}': {exc}")
            return
