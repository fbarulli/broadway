"""Slice 4a — pyfunc inference surface for Pipeline+signature artifacts.

Pins the new-path inference contract: artifacts logged by the train path (an
sklearn Pipeline with an explicit signature, cloudpickle-serialized) load
through the evaluate seam's loading function (``mlflow.pyfunc.load_model``)
and predict on a RAW input frame — the pre-preprocessing feature frame the
signature was inferred from — with MLflow's pyfunc signature enforcement
making wrong inputs fail loud.

Empirical finding (MLflow 3.15.1, pinned here): pyfunc enforcement RAISES on
an unsafe dtype mismatch (int64 column where the signature declares double —
"Can not safely convert int64 to float64") and COERCES silently on safe
widening (float32 → double). Both the wrong-column-set and the unsafe
wrong-dtype cases raise ``MlflowException`` with
``error_class="SCHEMA_ENFORCEMENT_FAILED"`` — loud, never swallowed.

Hermetic: every test logs to a tmp file-store tracking URI and tmp artifact
location — no server, no network.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow.models import infer_signature

from broadway.config.schema import (
    ColumnRole,
    ColumnSchema,
    DatasetContract,
    DataSourceRef,
    EnvironmentConfig,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    PipelineConfig,
    PreprocessingStepConfig,
    SplitConfig,
    TaskType,
)
from broadway.evaluate.module import _load_champion
from broadway.training.mlflow_utils import log_model, setup_mlflow
from broadway.training.trainer import build_model_pipeline


def _pipeline_config() -> PipelineConfig:
    """Minimal synthetic PipelineConfig: one-hot preprocessing on the category column."""
    environment = EnvironmentConfig(
        log_level="INFO",
        data_dir="data",
        raw_subdir="raw",
        processed_subdir="processed",
        download_chunk_size=8192,
        mlflow_tracking_uri="mlruns",
        database_user="user",
        database_password="pass",
        database_name="db",
        database_host="localhost",
        database_port=5432,
        sample_size_ci=1000,
        sample_size_stats=10000,
        api_replicas_min=1,
        api_replicas_max=3,
        api_hpa_cpu_threshold=80,
        monitoring_schedule="0 * * * *",
    )
    dataset = DatasetContract(
        name="synthetic",
        path="synthetic.parquet",
        target="price",
        task=TaskType.REGRESSION,
        datetime_column=None,
        columns={
            "cat": ColumnSchema(dtype="object", null_count=0, role=ColumnRole.FEATURE),
            "num": ColumnSchema(dtype="float64", null_count=0, role=ColumnRole.FEATURE),
            "price": ColumnSchema(dtype="float64", null_count=0, role=ColumnRole.TARGET),
        },
        lookup_tables={},
    )
    experiment = ExperimentConfig(
        data_source=DataSourceRef(loader="canonical", schema_contract="raw"),
        features=FeatureConfig(include=["cat", "num"], exclude=[], derived=[], encodings=[]),
        model=ModelConfig(type="linear", params={}),
        split=SplitConfig(type="random", validation_size=0.2),
        random_state=42,
        target_metric="rmse",
        preprocessing=[PreprocessingStepConfig(type="one_hot", columns=["cat"], params={})],
    )
    return PipelineConfig(dataset=dataset, environment=environment, experiment=experiment)


def _log_pipeline(tmp_path: Path) -> str:
    """Fit + log the tiny Pipeline via the train path's helpers; return its model URI.

    y = num + 1 on one-hot-encoded cat: the fit is exact, so raw-frame
    predictions are deterministic ([6.0, 7.0] for num=[5.0, 6.0]).
    """
    X_train = pd.DataFrame({"cat": ["x", "y", "x", "y"], "num": [1.0, 2.0, 3.0, 4.0]})
    y_train = pd.Series([2.0, 3.0, 4.0, 5.0])
    pipeline = build_model_pipeline(_pipeline_config(), "linear", {})
    pipeline.fit(X_train, y_train)
    setup_mlflow(str(tmp_path / "mlruns"), "pyfunc_inference")
    with mlflow.start_run():
        return log_model(pipeline, "model", signature=infer_signature(X_train, y_train))


def test_champion_predicts_on_raw_frame(tmp_path: Path) -> None:
    uri = _log_pipeline(tmp_path)
    champion = _load_champion(uri)
    X_raw = pd.DataFrame({"cat": ["x", "y"], "num": [5.0, 6.0]})
    preds = champion.predict(X_raw)
    assert list(preds) == pytest.approx([6.0, 7.0])
    # The raw frame is the signature's input (pre-preprocessing): the artifact
    # one-hots `cat` inside, so an unseen category still predicts deterministically.
    unseen = pd.DataFrame({"cat": ["zz", "zz"], "num": [5.0, 6.0]})
    assert list(champion.predict(unseen)) == pytest.approx([6.0, 7.0])
    assert mlflow.models.get_model_info(uri).signature is not None


def test_wrong_column_set_fails_loud(tmp_path: Path) -> None:
    uri = _log_pipeline(tmp_path)
    champion = _load_champion(uri)
    with pytest.raises(mlflow.exceptions.MlflowException) as excinfo:
        champion.predict(pd.DataFrame({"num": [5.0, 6.0]}))
    exc = excinfo.value
    assert exc.error_class == "SCHEMA_ENFORCEMENT_FAILED"
    assert exc.error_code == "INVALID_PARAMETER_VALUE"
    assert "missing inputs ['cat']" in str(exc)


def test_wrong_dtype_fails_loud(tmp_path: Path) -> None:
    uri = _log_pipeline(tmp_path)
    champion = _load_champion(uri)
    with pytest.raises(mlflow.exceptions.MlflowException) as excinfo:
        champion.predict(pd.DataFrame({"cat": ["x"], "num": [5]}))  # int64 vs double
    exc = excinfo.value
    assert exc.error_class == "SCHEMA_ENFORCEMENT_FAILED"
    assert exc.error_code == "INVALID_PARAMETER_VALUE"
    assert "Can not safely convert int64 to float64" in str(exc)
