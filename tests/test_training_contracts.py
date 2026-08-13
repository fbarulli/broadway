from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from broadway.analysis.contracts import AnalysisContract, AnalysisMode
from broadway.config.schema import (
    ColumnRole,
    ColumnSchema,
    DatasetContract,
    EnvironmentConfig,
    EtlStep,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    PipelineConfig,
    SplitConfig,
    TaskType,
    TrainStep,
)
from broadway.lineage import records
from broadway.training import module
from broadway.training.contracts import TrainingResult
from broadway.training.mlflow_utils import log_metrics, log_model, setup_mlflow
from broadway.training.models.base import BaseModel
from broadway.training.optuna import run_study
from broadway.training.trainer import train


def test_training_result_json_round_trip() -> None:
    result = TrainingResult(
        model_type="linear",
        params={"fit_intercept": 1, "alpha": 0.5},
        train_time_seconds=1.5,
        artifact_path="runs:/abc123/linear",
    )
    loaded = TrainingResult.model_validate_json(result.model_dump_json())
    assert loaded == result


def test_training_result_artifact_path_defaults_to_none() -> None:
    result = TrainingResult(
        model_type="lgbm",
        params={"n_estimators": 100},
        train_time_seconds=2.0,
    )
    assert result.artifact_path is None


def test_run_study_finds_quadratic_optimum() -> None:
    def objective(params: dict) -> float:
        return (params["x"] - 3.0) ** 2

    best = run_study(
        objective,
        search_space={"x": [-10.0, 10.0]},
        n_trials=50,
        random_state=42,
    )
    assert best["x"] == pytest.approx(3.0, abs=1.0)


def test_setup_mlflow_logs_without_server(tmp_path) -> None:
    setup_mlflow(str(tmp_path), "test_experiment")
    model = LinearRegression().fit(
        np.array([[1.0], [2.0], [3.0]]), np.array([2.0, 4.0, 6.0])
    )
    with mlflow.start_run():
        log_metrics({"rmse": 0.1, "r2": 0.99})
        uri = log_model(model, "linear")
    assert isinstance(uri, str)
    assert uri


def test_base_model_is_importable_and_abstract() -> None:
    assert callable(BaseModel)
    with pytest.raises(TypeError):
        BaseModel()


def test_trainer_returns_training_result() -> None:
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
    y = pd.Series([3.0, 6.0, 9.0])
    model, result = train("linear", X, y, n_jobs=1)
    assert result.model_type == "linear"
    assert result.params == {"n_jobs": 1}
    assert result.train_time_seconds >= 0
    assert result.artifact_path is None
    assert hasattr(model, "predict")


def _make_config(tmp_path: Path) -> PipelineConfig:
    environment = EnvironmentConfig(
        log_level="INFO",
        data_dir=str(tmp_path / "data"),
        raw_subdir="raw",
        processed_subdir="processed",
        download_chunk_size=8192,
        mlflow_tracking_uri=str(tmp_path / "mlruns"),
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
            "rooms": ColumnSchema(dtype="int64", null_count=0, role=ColumnRole.FEATURE),
            "area": ColumnSchema(dtype="int64", null_count=0, role=ColumnRole.FEATURE),
            "price": ColumnSchema(dtype="float64", null_count=0, role=ColumnRole.TARGET),
        },
        lookup_tables={},
        row_count=40,
    )
    experiment = ExperimentConfig(
        features=FeatureConfig(include=["rooms", "area"], exclude=[], derived=[], encodings=[]),
        model=ModelConfig(type="linear", params={}),
        split=SplitConfig(type="random", validation_size=0.2),
        random_state=42,
        target_metric="rmse",
        hpo=None,
    )
    etl = EtlStep(
        ci_sample_size=0,
        random_state=42,
        train_file="train.parquet",
        val_file="val.parquet",
        training_data_file="training_data.parquet",
        train_features_file="train_features.parquet",
        val_features_file="val_features.parquet",
        raw_dir="data/raw",
        processed_dir="data/processed",
        processed_file="training_data.parquet",
        min_trip_distance=0,
        max_trip_distance=50,
        min_trip_duration_minutes=1,
        max_trip_duration_minutes=180,
        batch_size=200000,
        rename_map={},
        taxi_urls=[],
        lookup_url="",
        lookup_filename="lookup.csv",
        training_table="training_data",
        target="price",
        validation_cutoff="2024-01-01",
        encoding_smoothing=50,
        frequency_fill=0,
    )
    train_step = TrainStep(
        random_state=42,
        n_jobs=1,
        cv_folds=2,
        model_file="model.pkl",
        n_estimators=10,
        learning_rate=0.05,
        num_leaves=31,
        subsample=1.0,
        colsample_bytree=1.0,
        quantile_tail=0.9,
        output_dir=str(tmp_path / "artifacts" / "training"),
        output_file="training_result.json",
    )
    return PipelineConfig(
        dataset=dataset,
        environment=environment,
        experiment=experiment,
        etl=etl,
        train=train_step,
        analysis=AnalysisContract(
            name="taxi",
            mode=AnalysisMode.PREDICTION,
            goal="predict price",
            row_definition="one row",
            decision_moment="now",
            available_info=["rooms"],
            leakage_notes=[],
            success_criterion="beat baseline",
        ),
    )


def test_module_run_writes_training_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_config(tmp_path)
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")
    out_dir = Path(cfg.environment.data_dir) / cfg.environment.processed_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "rooms": np.arange(1, 41),
            "area": np.arange(40, 80),
            "price": np.arange(1, 41) * 100.0,
        }
    )
    df.to_parquet(out_dir / cfg.etl.train_features_file, index=False)
    df.tail(10).to_parquet(out_dir / cfg.etl.val_features_file, index=False)

    module.run(cfg)

    result_path = Path(cfg.train.output_dir) / cfg.train.output_file
    assert result_path.exists()
    loaded = TrainingResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    assert loaded.model_type == "linear"
    assert loaded.artifact_path
