"""Thin wrappers over MLflow for training tracking and model registry."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

logger = logging.getLogger(__name__)

_MLFLOW_FILE_STORE_ENV = "MLFLOW_ALLOW_FILE_STORE"

# Connection-refusal markers inside MLflow's wrapped error message. MLflow's
# HTTP store wraps the underlying requests/urllib3 connection failure into a
# bare MlflowException (no __cause__), so the refusal is only visible in text.
_CONNECTION_REFUSED_MARKERS = (
    "Connection refused",
    "Failed to establish a new connection",
    "Max retries exceeded",
)


def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    if "://" not in tracking_uri:
        os.environ[_MLFLOW_FILE_STORE_ENV] = "true"
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    except (ConnectionError, mlflow.exceptions.MlflowException) as exc:
        if not _is_unreachable_http_store(tracking_uri, exc):
            raise
        raise RuntimeError(_unreachable_server_hint(tracking_uri)) from exc


def _is_unreachable_http_store(tracking_uri: str, exc: Exception) -> bool:
    """True only for a connection refusal against an http(s) tracking store."""
    if not tracking_uri.startswith(("http://", "https://")):
        return False
    if isinstance(exc, ConnectionError):
        return True
    return any(marker in str(exc) for marker in _CONNECTION_REFUSED_MARKERS)


def _unreachable_server_hint(tracking_uri: str) -> str:
    return (
        f"MLflow server unreachable at {tracking_uri} — start it: "
        "uv run mlflow server --backend-store-uri sqlite:///$(pwd)/.mlflow.db "
        "--artifacts-destination file://$(pwd)/mlruns (README: dev setup)"
    )


def log_params(params: dict[str, float | int | str]) -> None:
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, float]) -> None:
    mlflow.log_metrics(metrics)


def log_metadata(metadata: dict[str, float]) -> None:
    """Log run metadata (e.g. train/predict time, model-size bytes) as metrics."""
    mlflow.log_metrics(metadata)


def log_dataset(dataset_id: str, source_path: str, context: str = "train") -> None:
    """Log a dataset id and, when the parquet source exists, its lineage.

    MLflow 3.x removed ``mlflow.data.from_parquet``; the parquet lineage is
    recorded via ``from_pandas`` with the file path as the dataset source.
    A missing source is recoverable: a warning is logged and logging continues.
    """
    mlflow.log_params({"dataset_id": dataset_id})
    path = Path(source_path)
    if not path.exists():
        logger.warning("dataset source not found, skipping lineage: %s", source_path)
        return
    dataset = mlflow.data.from_pandas(pd.read_parquet(path), source=str(path))  # type: ignore[attr-defined]
    mlflow.log_input(dataset, context=context)


def log_model(model: Any, artifact_path: str) -> str:
    info = mlflow.sklearn.log_model(model, artifact_path)
    return info.model_uri


def get_champion(model_name: str, alias: str = "champion") -> str | None:
    model_uri = f"models:/{model_name}@{alias}"
    try:
        mlflow.pyfunc.load_model(model_uri)
    except mlflow.exceptions.MlflowException as exc:
        logger.warning("champion model not found: %s (%s)", model_uri, exc)
        return None
    return model_uri


def promote_candidate(model_name: str, model_uri: str, alias: str = "champion") -> None:
    version = mlflow.register_model(model_uri, model_name)
    mlflow.tracking.MlflowClient().set_registered_model_alias(model_name, alias, version.version)
