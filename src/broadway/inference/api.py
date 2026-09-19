"""Deferred serving endpoint — read-only model discovery over MLflow.

Exposes the real ASGI ``app`` invoked by ``k8s/api-deployment.yaml``
(``uvicorn broadway.inference.api:app``) with:

- ``GET /health`` — liveness plus installed package version.
- ``GET /models`` — models discoverable from the MLflow file store,
  scanned read-only (directory listing only — never trains, never writes).

Discovery source: ``MLFLOW_TRACKING_URI`` when it is a local path, else
``<repo_root>/mlruns``. A model is an ``<experiment>/models/<id>`` directory
carrying ``artifacts/MLmodel`` or ``meta.yaml``; dot-directories such as
``mlruns/.trash`` are skipped. ``mlruns/`` itself is git-ignored, so an
empty store simply yields an empty list.
"""

from __future__ import annotations

import os
from importlib import metadata
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from broadway.paths import repo_root


class HealthResponse(BaseModel):
    """Liveness payload: fixed status plus package version."""

    status: str
    version: str


class ModelEntry(BaseModel):
    """One discoverable model: registry dirname plus experiment id."""

    name: str
    experiment_id: str


class ModelsResponse(BaseModel):
    """Discovery payload: every model found by the read-only scan."""

    models: list[ModelEntry]


def package_version() -> str:
    """Return the installed ``broadway`` version, ``"unknown"`` if absent."""
    try:
        return metadata.version("broadway")
    except metadata.PackageNotFoundError:
        return "unknown"


def tracking_dir() -> Path:
    """Resolve the file-store dir used as the discovery source."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if uri and "://" not in uri:
        path = Path(uri)
        return path if path.is_absolute() else repo_root() / path
    return repo_root() / "mlruns"


def _is_model_dir(path: Path) -> bool:
    """True when a registry entry holds model metadata (read-only probe)."""
    return path.is_dir() and (
        (path / "artifacts" / "MLmodel").is_file() or (path / "meta.yaml").is_file()
    )


def list_models(store: Path | None = None) -> list[ModelEntry]:
    """List models in a file store; pure readdir, never trains or writes."""
    root = store if store is not None else tracking_dir()
    if not root.is_dir():
        return []
    found: list[ModelEntry] = []
    for exp in sorted(root.iterdir()):
        if not exp.is_dir() or exp.name.startswith("."):
            continue
        models = exp / "models"
        if not models.is_dir():
            continue
        for entry in sorted(models.iterdir()):
            if _is_model_dir(entry):
                found.append(ModelEntry(name=entry.name, experiment_id=exp.name))
    return found


app = FastAPI(title="broadway-inference")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe: fixed status plus installed package version."""
    return HealthResponse(status="ok", version=package_version())


@app.get("/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    """List discoverable models via a read-only store scan."""
    return ModelsResponse(models=list_models())
