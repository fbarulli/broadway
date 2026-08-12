"""Thin wrappers over MLflow for training tracking."""

from __future__ import annotations

import os
from typing import Any

import mlflow

_MLFLOW_FILE_STORE_ENV = "MLFLOW_ALLOW_FILE_STORE"


def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    if "://" not in tracking_uri:
        os.environ[_MLFLOW_FILE_STORE_ENV] = "true"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_metrics(metrics: dict[str, float]) -> None:
    mlflow.log_metrics(metrics)


def log_model(model: Any, artifact_path: str) -> str:
    info = mlflow.sklearn.log_model(model, artifact_path)
    return info.model_uri
