"""Serving endpoint tests — synthetic stores only, no server, no training.

Covers ``broadway.inference.api`` via starlette's ``TestClient`` (in-process
ASGI, no sockets) plus direct calls for the pure discovery helpers. The real
``mlruns/`` tree is never touched: every store is built under ``tmp_path``.
"""

from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from broadway.inference import api
from broadway.paths import repo_root


def _make_model(store: Path, exp: str, name: str, marker: str = "MLmodel") -> None:
    """Create one synthetic ``<exp>/models/<name>`` entry with metadata."""
    if marker == "MLmodel":
        target = store / exp / "models" / name / "artifacts" / "MLmodel"
    else:
        target = store / exp / "models" / name / marker
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("synthetic", encoding="utf-8")


def _tree_hashes(root: Path) -> dict[str, str]:
    """SHA-256 snapshot of every file under ``root`` (read-only check)."""
    return {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_health_returns_ok_and_version() -> None:
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": metadata.version("broadway")}


def test_models_empty_when_store_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "no-such-store"))
    client = TestClient(api.app)
    response = client.get("/models")
    assert response.status_code == 200
    assert response.json() == {"models": []}


def test_models_lists_synthetic_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "mlruns"
    _make_model(store, "1", "m-aaa")
    _make_model(store, "2", "m-bbb", marker="meta.yaml")
    _make_model(store, ".trash", "m-xxx")  # soft-deleted: must be skipped
    (store / "1" / "models" / "not-a-model").mkdir(parents=True)  # no marker
    (store / "3").mkdir(parents=True)  # experiment without models/
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(store))
    client = TestClient(api.app)
    response = client.get("/models")
    assert response.status_code == 200
    assert response.json() == {
        "models": [
            {"name": "m-aaa", "experiment_id": "1"},
            {"name": "m-bbb", "experiment_id": "2"},
        ]
    }


def test_models_never_writes_to_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "mlruns"
    _make_model(store, "1", "m-aaa")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(store))
    before = _tree_hashes(store)
    TestClient(api.app).get("/models")
    assert _tree_hashes(store) == before


def test_list_models_direct_missing_dir(tmp_path: Path) -> None:
    assert api.list_models(tmp_path / "no-such-store") == []


def test_tracking_dir_defaults_to_repo_mlruns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert api.tracking_dir() == repo_root() / "mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "mlruns")
    assert api.tracking_dir() == repo_root() / "mlruns"
