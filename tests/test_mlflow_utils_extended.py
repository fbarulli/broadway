"""Extended tests for the MLflow thin wrappers: metadata and dataset lineage."""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.models import infer_signature
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from broadway.training.mlflow_utils import (
    log_dataset,
    log_metadata,
    log_model,
    setup_mlflow,
)

_LOGGER_NAME = "broadway.training.mlflow_utils"


def test_log_metadata_records_metrics(tmp_path: Path) -> None:
    setup_mlflow(str(tmp_path / "mlruns"), "test_experiment")
    with mlflow.start_run() as run:
        log_metadata({"train_time_seconds": 1.5, "model_size_bytes": 4096.0})
    run_data = mlflow.tracking.MlflowClient().get_run(run.info.run_id).data
    assert run_data.metrics["train_time_seconds"] == pytest.approx(1.5)
    assert run_data.metrics["model_size_bytes"] == pytest.approx(4096.0)


def test_log_dataset_records_param_and_lineage(tmp_path: Path) -> None:
    setup_mlflow(str(tmp_path / "mlruns"), "test_experiment")
    source = tmp_path / "train.parquet"
    pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}).to_parquet(source, index=False)
    with mlflow.start_run() as run:
        log_dataset("test-train", str(source), context="train")
    run_obj = mlflow.tracking.MlflowClient().get_run(run.info.run_id)
    assert run_obj.data.params["dataset_id"] == "test-train"
    inputs = run_obj.inputs.dataset_inputs
    assert len(inputs) == 1
    assert str(source) in inputs[0].dataset.source
    tags = {tag.key: tag.value for tag in inputs[0].tags}
    assert tags["mlflow.data.context"] == "train"


def test_log_dataset_missing_source_warns_and_continues(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    setup_mlflow(str(tmp_path / "mlruns"), "test_experiment")
    missing = tmp_path / "missing.parquet"
    with mlflow.start_run() as run, caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        log_dataset("test-train", str(missing), context="train")
    assert "missing.parquet" in caplog.text
    run_obj = mlflow.tracking.MlflowClient().get_run(run.info.run_id)
    assert run_obj.data.params["dataset_id"] == "test-train"
    assert run_obj.inputs.dataset_inputs == []


def test_logged_pipeline_reloads_and_predicts_raw_frame(tmp_path: Path) -> None:
    """The logged artifact is the Pipeline and carries the signature (Slice 3).

    Logs a fitted Pipeline with an explicit signature, reloads it via the
    sklearn flavor, and asserts it predicts on a raw feature frame — proving
    preprocessing (or passthrough) ships inside the artifact, not the bare
    model alone.
    """
    setup_mlflow(str(tmp_path / "mlruns"), "test_experiment")
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
    y = pd.Series([3.0, 6.0, 9.0])
    pipeline = Pipeline([("model", LinearRegression())]).fit(X, y)
    with mlflow.start_run():
        uri = log_model(pipeline, "model", signature=infer_signature(X, y))
    loaded = mlflow.sklearn.load_model(uri)
    preds = loaded.predict(X)
    assert preds.shape == (3,)
    assert np.issubdtype(preds.dtype, np.floating)
    assert mlflow.models.get_model_info(uri).signature is not None
