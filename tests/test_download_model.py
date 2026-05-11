import pytest

from scripts import download_model


def test_build_release_api_url_uses_pinned_tag_endpoint():
    url = download_model._build_release_api_url(
        owner="owner",
        repo="repo",
        tag="eng-sp-tranlate",
    )

    assert url == "https://api.github.com/repos/owner/repo/releases/tags/eng-sp-tranlate"


def test_build_release_api_url_supports_latest_endpoint():
    url = download_model._build_release_api_url(
        owner="owner",
        repo="repo",
        tag="latest",
    )

    assert url == "https://api.github.com/repos/owner/repo/releases/latest"


def test_find_asset_download_url_returns_exact_asset():
    release_metadata = {
        "assets": [
            {"name": "tokenizer.zip", "browser_download_url": "https://example.test/t.zip"},
            {"name": "best_model.pth", "browser_download_url": "https://example.test/m.pth"},
        ]
    }

    assert (
        download_model._find_asset_download_url(release_metadata, "best_model.pth")
        == "https://example.test/m.pth"
    )


def test_find_asset_download_url_reports_available_assets():
    release_metadata = {"assets": [{"name": "tokenizer.zip"}]}

    with pytest.raises(ValueError, match="tokenizer.zip"):
        download_model._find_asset_download_url(release_metadata, "best_model.pth")


def test_extract_tokenizer_archive_handles_nested_root(tmp_path):
    archive_path = tmp_path / "tokenizer.zip"
    tokenizer_dir = tmp_path / "data" / "tokenizer"

    import zipfile

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("tokenizer/config.json", "{}")

    download_model._extract_tokenizer_archive(
        archive_path=str(archive_path),
        tokenizer_dir=str(tokenizer_dir),
        force=False,
    )

    assert (tokenizer_dir / "config.json").read_text(encoding="utf-8") == "{}"


def test_extract_tokenizer_archive_skips_existing_without_force(tmp_path):
    archive_path = tmp_path / "tokenizer.zip"
    tokenizer_dir = tmp_path / "data" / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    existing_file = tokenizer_dir / "config.json"
    existing_file.write_text("existing", encoding="utf-8")

    import zipfile

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("config.json", "new")

    download_model._extract_tokenizer_archive(
        archive_path=str(archive_path),
        tokenizer_dir=str(tokenizer_dir),
        force=False,
    )

    assert existing_file.read_text(encoding="utf-8") == "existing"


def test_find_hf_tokenizer_dir_prefers_configured_path(tmp_path):
    snapshot_dir = tmp_path / "snapshot"
    tokenizer_dir = snapshot_dir / "custom_tokenizer"
    tokenizer_dir.mkdir(parents=True)

    result = download_model._find_hf_tokenizer_dir(
        snapshot_dir=str(snapshot_dir),
        tokenizer_path="custom_tokenizer",
    )

    assert result == str(tokenizer_dir)


def test_find_hf_tokenizer_dir_supports_data_tokenizer_fallback(tmp_path):
    snapshot_dir = tmp_path / "snapshot"
    tokenizer_dir = snapshot_dir / "data" / "tokenizer"
    tokenizer_dir.mkdir(parents=True)

    result = download_model._find_hf_tokenizer_dir(
        snapshot_dir=str(snapshot_dir),
        tokenizer_path="missing",
    )

    assert result == str(tokenizer_dir)


def test_find_hf_tokenizer_dir_reports_checked_paths(tmp_path):
    with pytest.raises(FileNotFoundError, match="data"):
        download_model._find_hf_tokenizer_dir(
            snapshot_dir=str(tmp_path),
            tokenizer_path="missing",
        )


def test_download_from_huggingface_copies_checkpoint_and_tokenizer(monkeypatch, tmp_path):
    checkpoint_source = tmp_path / "cache" / "best_model.pth"
    tokenizer_source = tmp_path / "snapshot" / "tokenizer"
    checkpoint_source.parent.mkdir()
    tokenizer_source.mkdir(parents=True)
    checkpoint_source.write_bytes(b"checkpoint")
    (tokenizer_source / "config.json").write_text("{}", encoding="utf-8")

    class Args:
        hf_repo_id = "mathew-felix/en-es-nmt-transformer"
        hf_revision = "main"
        hf_checkpoint_path = "best_model.pth"
        hf_tokenizer_path = "tokenizer"
        force = False

    monkeypatch.setattr(
        download_model,
        "hf_hub_download",
        lambda **kwargs: str(checkpoint_source),
    )
    monkeypatch.setattr(
        download_model,
        "snapshot_download",
        lambda **kwargs: str(tmp_path / "snapshot"),
    )

    download_model._download_from_huggingface(
        Args(),
        checkpoint_path=str(tmp_path / "repo" / "best_model.pth"),
        tokenizer_dir=str(tmp_path / "repo" / "data" / "tokenizer"),
    )

    assert (tmp_path / "repo" / "best_model.pth").read_bytes() == b"checkpoint"
    assert (tmp_path / "repo" / "data" / "tokenizer" / "config.json").exists()


def test_main_downloads_checkpoint_and_tokenizer(monkeypatch, tmp_path):
    release_metadata = {
        "assets": [
            {"name": "best_model.pth", "browser_download_url": "checkpoint-url"},
            {"name": "tokenizer.zip", "browser_download_url": "tokenizer-url"},
        ]
    }

    def fake_download(download_url, destination_path):
        if download_url == "checkpoint-url":
            with open(destination_path, "wb") as handle:
                handle.write(b"checkpoint")
            return

        import zipfile

        with zipfile.ZipFile(destination_path, "w") as archive:
            archive.writestr("tokenizer/config.json", "{}")

    monkeypatch.setattr(download_model, "_get_repo_root", lambda: str(tmp_path))
    monkeypatch.setattr(
        download_model,
        "_fetch_release_metadata",
        lambda owner, repo, tag: release_metadata,
    )
    monkeypatch.setattr(download_model, "_download_file", fake_download)
    monkeypatch.setattr("sys.argv", ["download_model.py"])

    download_model.main()

    assert (tmp_path / "best_model.pth").read_bytes() == b"checkpoint"
    assert (tmp_path / "data" / "tokenizer" / "config.json").exists()
