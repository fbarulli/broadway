from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from broadway.analysis.contracts import AnalysisContract, AnalysisMode
from broadway.config.schema import (
    ColumnRole,
    ColumnSchema,
    DatasetContract,
    DataSourceRef,
    EnvironmentConfig,
    EtlStep,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    PipelineConfig,
    PreprocessingStepConfig,
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
from broadway.training.trainer import build_model_pipeline, train


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


def test_trainer_returns_training_result(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
    y = pd.Series([3.0, 6.0, 9.0])
    model, result = train(cfg, X, y, n_jobs=1)
    assert isinstance(model, Pipeline)
    assert result.model_type == "linear"
    assert result.params == {"n_jobs": 1}
    assert result.train_time_seconds >= 0
    assert result.artifact_path is None
    assert hasattr(model, "predict")


def test_build_model_pipeline_applies_pre_params(tmp_path: Path) -> None:
    """pre__<step>__<param> keys address the preprocessing segment of the
    Pipeline (the HPO search-space contract); model keys stay bare."""
    cfg = _make_config(tmp_path)
    experiment = cfg.experiment.model_copy(
        update={
            "preprocessing": [
                PreprocessingStepConfig(
                    type="target_encoding", columns=["rooms"], params={"smoothing": 20}
                )
            ]
        }
    )
    cfg = cfg.model_copy(update={"experiment": experiment})
    pipeline = build_model_pipeline(
        cfg, "lgbm", {"n_estimators": 10, "pre__target_encoding_0__smoothing": 35}
    )
    params = pipeline.get_params()
    assert params["pre__target_encoding_0__smoothing"] == 35
    assert params["model__n_estimators"] == 10


def test_target_encoding_recipe_end_to_end_fits_and_predicts(tmp_path: Path) -> None:
    """C1 end-to-end: a target_encoding recipe composes through
    build_model_pipeline, fits with the target passed as the ``y`` argument
    (never read out of X — eligible_feature_columns drops the target before
    fit), and the fitted pipeline predicts on a fresh frame."""
    cfg = _make_config(tmp_path)
    experiment = cfg.experiment.model_copy(
        update={
            "preprocessing": [
                PreprocessingStepConfig(
                    type="target_encoding", columns=["zone_id"], params={"smoothing": 20}
                )
            ]
        }
    )
    cfg = cfg.model_copy(update={"experiment": experiment})
    rng = np.random.default_rng(7)
    n = 60
    X_train = pd.DataFrame(
        {
            "zone_id": rng.integers(1, 6, size=n),
            "area": rng.normal(size=n),
        }
    )
    y_train = pd.Series(
        2.0 * X_train["zone_id"] + X_train["area"] + rng.normal(scale=0.1, size=n)
    )
    pipeline = build_model_pipeline(cfg, "linear", {})
    pipeline.fit(X_train, y_train)
    # Three coefficients prove the recipe step ran: zone_id, area, and the
    # y-derived zone_id_target_enc column were all present at model fit.
    assert pipeline.named_steps["model"].coef_.shape == (3,)
    X_test = pd.DataFrame({"zone_id": [1, 3, 5], "area": [0.0, 1.0, -1.0]})
    preds = pipeline.predict(X_test)
    assert preds.shape == (3,)
    assert np.all(np.isfinite(preds))


def test_build_model_pipeline_seeds_estimator_from_experiment(tmp_path: Path) -> None:
    """C2: models whose registry entry accepts random_state are seeded from
    cfg.experiment.random_state; an explicit param random_state wins; models
    without random_state stay unseeded."""
    cfg = _make_config(tmp_path)
    seeded = build_model_pipeline(cfg, "lgbm", {"n_estimators": 10})
    assert seeded.get_params()["model__random_state"] == cfg.experiment.random_state
    explicit = build_model_pipeline(cfg, "lgbm", {"n_estimators": 10, "random_state": 7})
    assert explicit.get_params()["model__random_state"] == 7
    unseeded = build_model_pipeline(cfg, "linear", {})
    assert "model__random_state" not in unseeded.get_params()


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
    )
    experiment = ExperimentConfig(
        data_source=DataSourceRef(loader="canonical", schema_contract="raw"),
        features=FeatureConfig(include=["rooms", "area"], exclude=[], derived=[], encodings=[]),
        model=ModelConfig(type="linear", params={}),
        split=SplitConfig(type="random", validation_size=0.2),
        random_state=42,
        target_metric="rmse",
        hpo=None,
    )
    etl = EtlStep(
        ci_sample_size=0,
        max_drop_fraction=0.5,
        random_state=42,
        train_file="train.parquet",
        val_file="val.parquet",
        training_data_file="training_data.parquet",
        train_features_file="train_features.parquet",
        val_features_file="val_features.parquet",
        missing_encodings=["", "NA", "null"],
    )
    train_step = TrainStep(
        random_state=42,
        n_jobs=1,
        cv_folds=2,
        cv_kind="kfold",
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
            name="test",
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
