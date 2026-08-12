"""Thin wrappers over MLflow for training tracking and model registry."""

from __future__ import annotations

import logging
import os
from typing import Any

import mlflow

logger = logging.getLogger(__name__)

_MLFLOW_FILE_STORE_ENV = "MLFLOW_ALLOW_FILE_STORE"


def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    if "://" not in tracking_uri:
        os.environ[_MLFLOW_FILE_STORE_ENV] = "true"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_params(params: dict[str, float | int | str]) -> None:
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, float]) -> None:
    mlflow.log_metrics(metrics)


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
