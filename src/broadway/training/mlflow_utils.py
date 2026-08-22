"""Thin wrappers over MLflow for training tracking and model registry."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import ModelSignature

logger = logging.getLogger(__name__)

# Champion manifest buckets (SKLEARN_PIPELINES.md Slice 4, decision 3): a
# deployed champion's logging path. ModelPyFunc retirement is the CHECKED
# condition "bare_model is empty"; ambiguous always needs a human look.
BARE_MODEL = "bare_model"
PIPELINE_SIGNATURE = "pipeline_signature"
AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class ChampionArtifact:
    """One deployed champion: a registered model version holding the alias."""

    model_name: str
    version: str
    artifact_uri: str | None
    bucket: str
    reason: str = ""

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
    return f"MLflow server unreachable at {tracking_uri} — start commands: README · MLflow dev setup"


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


def log_model(model: Any, artifact_path: str, signature: ModelSignature | None = None) -> str:
    """Log a fitted model (bare or Pipeline) with cloudpickle serialization.

    MLflow 3.15's skops untrusted-type audit false-positives on tree models,
    so both log paths use the cloudpickle serialization format. An explicit
    ``signature`` (from ``infer_signature(X, y)``) makes MLflow own the
    fit/predict shape contract; without one MLflow's own inference applies.
    """
    info = mlflow.sklearn.log_model(
        model,
        artifact_path,
        signature=signature,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )
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


def classify_champion(artifact_uri: str) -> tuple[str, str]:
    """Classify a champion artifact by logging path from its MLmodel metadata.

    bare_model — python_function pythonmodel/code wrapper (the ModelPyFunc
    path) with no sklearn flavor. pipeline_signature — sklearn flavor with an
    explicit signature (the new-path Pipeline logging). Anything else is
    ambiguous with a reason string: never force a bucket without clean
    evidence (e.g. a signature-less sklearn artifact is the pre-Slice-3 bare
    logging path, which is neither of the two named paths).
    """
    try:
        info = mlflow.models.get_model_info(artifact_uri)
    except mlflow.exceptions.MlflowException as exc:
        return AMBIGUOUS, f"metadata unreadable: {exc}"
    flavors = info.flavors or {}
    pyfunc = flavors.get("python_function") or {}
    wrapped = bool(pyfunc) and (
        pyfunc.get("python_model") is not None or pyfunc.get("code") is not None
    )
    if "sklearn" in flavors:
        if info.signature is not None:
            return PIPELINE_SIGNATURE, ""
        return AMBIGUOUS, "sklearn flavor without explicit signature — pre-Slice-3 bare-model logging path"
    if wrapped:
        return BARE_MODEL, ""
    return AMBIGUOUS, "no sklearn flavor and no ModelPyFunc wrapper signal in metadata"


def list_champions(tracking_uri: str, alias: str = "champion") -> list[ChampionArtifact]:
    """List deployed champions, each classified by artifact logging path.

    A champion is a registered model version holding ``alias`` (set by
    promote_candidate). Registration and classification read the same file
    store the training path uses; see classify_champion for the bucket rules.
    """
    if "://" not in tracking_uri:
        os.environ[_MLFLOW_FILE_STORE_ENV] = "true"
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    champions: list[ChampionArtifact] = []
    for model in client.search_registered_models():
        try:
            version = client.get_model_version_by_alias(model.name, alias)
        except mlflow.exceptions.MlflowException:
            continue
        source = version.source
        if source is None:
            champions.append(
                ChampionArtifact(model.name, version.version, None, AMBIGUOUS, "registered version has no artifact URI")
            )
            continue
        bucket, reason = classify_champion(source)
        champions.append(
            ChampionArtifact(model.name, version.version, source, bucket, reason)
        )
    return champions
