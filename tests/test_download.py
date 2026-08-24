"""Tests for broadway.data.download — URL fetch into the raw data directory.

The network boundary (requests.get) is replaced with a fake streaming
response; everything else (filename derivation incl. percent-decoding,
directory creation, chunked write via env.download_chunk_size, HTTP-error
propagation) is exercised for real.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from broadway.config.schema import EnvironmentConfig
from broadway.data.download import download


def _env(tmp_path: Path, chunk_size: int = 8) -> EnvironmentConfig:
    return EnvironmentConfig(
        log_level="INFO",
        data_dir=str(tmp_path / "data"),
        raw_subdir="raw",
        processed_subdir="processed",
        download_chunk_size=chunk_size,
        mlflow_tracking_uri=str(tmp_path / "mlruns"),
        database_user="u",
        database_password="p",
        database_name="db",
        database_host="localhost",
        database_port=5432,
        sample_size_ci=1000,
        sample_size_stats=10000,
        api_replicas_min=1,
        api_replicas_max=2,
        api_hpa_cpu_threshold=70,
    )


class _FakeResponse:
    def __init__(self, chunks: list[bytes], status_ok: bool = True) -> None:
        self._chunks = chunks
        self._status_ok = status_ok

    def raise_for_status(self) -> None:
        if not self._status_ok:
            raise requests.HTTPError("404 Client Error")

    def iter_content(self, chunk_size: int) -> list[bytes]:
        # honor the configured chunk size so the size contract is observable
        for c in self._chunks:
            assert len(c) <= chunk_size
        return self._chunks


def test_download_writes_chunks_under_raw_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_urls: list[str] = []

    def fake_get(url: str, stream: bool = False) -> _FakeResponse:
        assert stream is True
        seen_urls.append(url)
        return _FakeResponse([b"hello ", b"world"])

    monkeypatch.setattr("broadway.data.download.requests.get", fake_get)
    dest = download(
        "https://example.com/datasets/city%20file.csv", _env(tmp_path)
    )
    assert seen_urls == ["https://example.com/datasets/city%20file.csv"]
    # percent-encoded name is decoded for the on-disk filename
    assert dest.name == "city file.csv"
    assert dest == tmp_path / "data" / "raw" / "city file.csv"
    assert dest.read_bytes() == b"hello world"


def test_download_http_error_propagates_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "broadway.data.download.requests.get",
        lambda url, stream=False: _FakeResponse([b"x"], status_ok=False),
    )
    with pytest.raises(requests.HTTPError, match="404"):
        download("https://example.com/missing.csv", _env(tmp_path))
    assert not (tmp_path / "data" / "raw").exists() or (
        list((tmp_path / "data" / "raw").iterdir()) == []
    )


def test_download_url_without_path_uses_fallback_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "broadway.data.download.requests.get",
        lambda url, stream=False: _FakeResponse([b"d"]),
    )
    dest = download("https://example.com", _env(tmp_path))
    assert dest.name == "download"
    assert dest.read_bytes() == b"d"
