import argparse
import os
import shutil
import tempfile
import zipfile

import requests

DEFAULT_OWNER = "mathew-felix"
DEFAULT_REPO = "english-spanish-translator"
DEFAULT_TAG = "eng-sp-tranlate"
DEFAULT_CHECKPOINT_ASSET = "best_model.pth"
DEFAULT_TOKENIZER_ASSET = "tokenizer.zip"


def _get_repo_root():
    """Return the repository root for path-safe downloads.
    The script is expected to live inside the repo `scripts/` directory.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _build_release_api_url(owner, repo, tag):
    """Build the GitHub Releases API URL for one repository.
    `latest` uses the latest-release endpoint; other values are treated as tags.
    """
    base_url = f"https://api.github.com/repos/{owner}/{repo}/releases"
    if tag == "latest":
        return f"{base_url}/latest"
    return f"{base_url}/tags/{tag}"


def _fetch_release_metadata(owner, repo, tag):
    """Fetch release metadata from the GitHub API.
    A timeout is used so the script fails fast on network problems.
    """
    response = requests.get(
        _build_release_api_url(owner, repo, tag),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _find_asset_download_url(release_metadata, asset_name):
    """Return the public download URL for one release asset.
    The error lists available asset names when the requested asset is missing.
    """
    for asset in release_metadata.get("assets", []):
        if asset.get("name") == asset_name:
            return asset.get("browser_download_url")

    available_assets = ", ".join(
        asset.get("name", "<unknown>") for asset in release_metadata.get("assets", [])
    )
    raise ValueError(
        f"Asset '{asset_name}' was not found in the selected release. "
        f"Available assets: {available_assets or 'none'}"
    )


def _download_file(download_url, destination_path):
    """Download one release asset to disk.
    Parent directories are created first and the file is streamed in chunks.
    """
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    with requests.get(download_url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with open(destination_path, "wb") as destination_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    destination_file.write(chunk)


def _resolve_extracted_root(extracted_dir):
    """Return the tokenizer root inside an extracted archive.
    Some archives contain one top-level folder while others contain files directly.
    """
    entries = [
        os.path.join(extracted_dir, entry_name)
        for entry_name in os.listdir(extracted_dir)
    ]
    directories = [entry for entry in entries if os.path.isdir(entry)]
    files = [entry for entry in entries if os.path.isfile(entry)]

    if len(directories) == 1 and not files:
        return directories[0]
    return extracted_dir


def _extract_tokenizer_archive(archive_path, tokenizer_dir, force):
    """Extract the tokenizer archive into `data/tokenizer`.
    Existing tokenizer files are replaced only when `force` is enabled.
    """
    if os.path.isdir(tokenizer_dir):
        if not force:
            print(f"Tokenizer already exists at {tokenizer_dir}. Skipping.")
            return
        shutil.rmtree(tokenizer_dir)

    os.makedirs(os.path.dirname(tokenizer_dir), exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(archive_path, "r") as archive_file:
            archive_file.extractall(temp_dir)

        extracted_root = _resolve_extracted_root(temp_dir)
        shutil.copytree(extracted_root, tokenizer_dir)


def _parse_args():
    """Parse command-line arguments for release asset downloads.
    Defaults target the public GitHub release for this repository.
    """
    parser = argparse.ArgumentParser(
        description="Download the model checkpoint and tokenizer from a GitHub Release."
    )
    parser.add_argument(
        "--owner",
        default=os.getenv("MODEL_RELEASE_OWNER", DEFAULT_OWNER),
        help="GitHub owner or org name.",
    )
    parser.add_argument(
        "--repo",
        default=os.getenv("MODEL_RELEASE_REPO", DEFAULT_REPO),
        help="GitHub repository name.",
    )
    parser.add_argument(
        "--tag",
        default=os.getenv("MODEL_RELEASE_TAG", DEFAULT_TAG),
        help="Release tag to use. Defaults to the latest release.",
    )
    parser.add_argument(
        "--checkpoint-asset",
        default=DEFAULT_CHECKPOINT_ASSET,
        help="Checkpoint asset name in the release.",
    )
    parser.add_argument(
        "--tokenizer-asset",
        default=DEFAULT_TOKENIZER_ASSET,
        help="Tokenizer archive asset name in the release.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing checkpoint and tokenizer files.",
    )
    return parser.parse_args()


def main():
    """Download the release checkpoint and tokenizer into repo-local paths.
    The checkpoint is saved to the repo root and the tokenizer is extracted to `data/tokenizer`.
    """
    args = _parse_args()
    repo_root = _get_repo_root()
    checkpoint_path = os.path.join(repo_root, "best_model.pth")
    tokenizer_dir = os.path.join(repo_root, "data", "tokenizer")

    if os.path.isfile(checkpoint_path) and not args.force:
        print(f"Checkpoint already exists at {checkpoint_path}. Skipping.")
    else:
        release_metadata = _fetch_release_metadata(args.owner, args.repo, args.tag)
        checkpoint_url = _find_asset_download_url(release_metadata, args.checkpoint_asset)
        print(f"Downloading checkpoint from {checkpoint_url}")
        _download_file(checkpoint_url, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    if os.path.isdir(tokenizer_dir) and not args.force:
        print(f"Tokenizer already exists at {tokenizer_dir}. Skipping.")
        return

    release_metadata = _fetch_release_metadata(args.owner, args.repo, args.tag)
    tokenizer_url = _find_asset_download_url(release_metadata, args.tokenizer_asset)

    with tempfile.TemporaryDirectory() as temp_dir:
        tokenizer_archive_path = os.path.join(temp_dir, args.tokenizer_asset)
        print(f"Downloading tokenizer from {tokenizer_url}")
        _download_file(tokenizer_url, tokenizer_archive_path)
        _extract_tokenizer_archive(tokenizer_archive_path, tokenizer_dir, args.force)

    print(f"Saved tokenizer to {tokenizer_dir}")


if __name__ == "__main__":
    main()
