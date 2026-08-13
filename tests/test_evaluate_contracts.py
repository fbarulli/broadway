from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from broadway.analysis.contracts import AnalysisContract, AnalysisMode
from broadway.baseline.contracts import BaselineResult, save_result
from broadway.config.schema import (
    BaselineStep,
    ColumnRole,
    ColumnSchema,
    DatasetContract,
    EnvironmentConfig,
    EtlStep,
    EvaluateStep,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    PipelineConfig,
    SplitConfig,
    TaskType,
    TrainStep,
)
from broadway.evaluate import module as evaluate_module
from broadway.evaluate.comparison import compare_models
from broadway.evaluate.contracts import EvaluationResult, ModelComparison
from broadway.evaluate.promotion import should_promote
from broadway.evaluate.validation import cross_validate, residual_summary
from broadway.lineage import records
from broadway.training import module as training_module


def test_evaluation_result_json_round_trip() -> None:
    comparison = ModelComparison(
        metrics={
            "mae": {
                "candidate": 0.5,
                "champion": 0.4,
                "delta": 0.1,
                "delta_pct": 0.25,
            }
        }
    )
    result = EvaluationResult(
        metrics={"mae": 0.5, "rmse": 0.8, "r2": 0.9},
        promote=True,
        reason="improvement over champion",
        cv_metrics={"mae": 0.6, "rmse": 0.9, "r2": 0.88},
        residuals={"mean_residual": 0.1, "std_residual": 0.2, "max_abs_residual": 0.5},
        comparison=comparison,
    )
    data = result.model_dump_json()
    assert json.loads(data) == {
        "metrics": {"mae": 0.5, "rmse": 0.8, "r2": 0.9},
        "promote": True,
        "reason": "improvement over champion",
        "cv_metrics": {"mae": 0.6, "rmse": 0.9, "r2": 0.88},
        "residuals": {"mean_residual": 0.1, "std_residual": 0.2, "max_abs_residual": 0.5},
        "comparison": {
            "metrics": {
                "mae": {
                    "candidate": 0.5,
                    "champion": 0.4,
                    "delta": 0.1,
                    "delta_pct": 0.25,
                }
            }
        },
        "baseline": None,
        "warnings": [],
    }
    assert EvaluationResult.model_validate_json(data) == result


def test_evaluation_result_optional_fields_default_to_none() -> None:
    result = EvaluationResult(
        metrics={"mae": 0.5, "rmse": 0.8, "r2": 0.9},
        promote=True,
        reason="no champion",
    )
    assert result.cv_metrics is None
    assert result.residuals is None
    assert result.comparison is None
    assert result.baseline is None
    assert result.warnings == []
    loaded = EvaluationResult.model_validate_json(result.model_dump_json())
    assert loaded.cv_metrics is None
    assert loaded.residuals is None
    assert loaded.comparison is None
    assert loaded.warnings == []


def test_should_promote_no_champion() -> None:
    promote, reason = should_promote(candidate_rmse=0.8, champion_rmse=None, threshold=0.05)
    assert promote is True
    assert reason


def test_should_promote_worse_candidate() -> None:
    promote, reason = should_promote(candidate_rmse=1.2, champion_rmse=0.8, threshold=0.05)
    assert promote is False
    assert "degradation" in reason


def test_compare_models_champion_none() -> None:
    candidate = {"mae": 0.5, "rmse": 0.8, "r2": 0.9}
    result = compare_models(candidate, None)
    assert isinstance(result, ModelComparison)
    assert set(result.metrics) == set(candidate)
    for metric, value in candidate.items():
        assert result.metrics[metric].candidate == value
        assert result.metrics[metric].champion is None
        assert result.metrics[metric].delta is None
        assert result.metrics[metric].delta_pct is None


def test_compare_models_with_champion() -> None:
    candidate = {"mae": 0.5, "rmse": 0.8, "r2": 0.9}
    champion = {"mae": 0.4, "rmse": 1.0}
    result = compare_models(candidate, champion)

    mae = result.metrics["mae"]
    assert mae.candidate == 0.5
    assert mae.champion == 0.4
    assert mae.delta == pytest.approx(0.1)
    assert mae.delta_pct == pytest.approx(0.25)

    rmse = result.metrics["rmse"]
    assert rmse.candidate == 0.8
    assert rmse.champion == 1.0
    assert rmse.delta == pytest.approx(-0.2)
    assert rmse.delta_pct == pytest.approx(-0.2)

    r2 = result.metrics["r2"]
    assert r2.candidate == 0.9
    assert r2.champion is None
    assert r2.delta is None
    assert r2.delta_pct is None


def test_cross_validate_returns_finite_metrics() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 3))
    coef = np.array([1.5, -2.0, 0.5])
    y = X @ coef + rng.normal(scale=0.1, size=60)
    metrics = cross_validate(LinearRegression(), X, y, cv_folds=5, random_state=0)
    assert set(metrics) == {"mae", "rmse", "r2"}
    assert all(np.isfinite(value) for value in metrics.values())


def test_residual_summary() -> None:
    y_true = np.zeros(4)
    y_pred = np.array([1.0, -1.0, 1.0, -1.0])
    summary = residual_summary(y_true, y_pred)
    assert summary == {
        "mean_residual": pytest.approx(1.0),
        "std_residual": pytest.approx(1.0),
        "max_abs_residual": pytest.approx(1.0),
    }


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
    evaluate_step = EvaluateStep(
        target_metric="rmse",
        promotion_threshold=0.05,
        output_dir=str(tmp_path / "artifacts" / "evaluation"),
        output_file="metrics.json",
    )
    baseline_step = BaselineStep(
        output_dir=str(tmp_path / "artifacts" / "baseline"),
        output_file="baseline.json",
    )
    return PipelineConfig(
        dataset=dataset,
        environment=environment,
        experiment=experiment,
        etl=etl,
        train=train_step,
        evaluate=evaluate_step,
        baseline=baseline_step,
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


def test_module_run_writes_evaluation_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    training_module.run(cfg)
    evaluate_module.run(cfg)

    result_path = Path(cfg.evaluate.output_dir) / cfg.evaluate.output_file
    assert result_path.exists()
    evaluation = EvaluationResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    assert set(evaluation.metrics) == {"mae", "rmse", "r2"}
    assert evaluation.cv_metrics is not None
    assert set(evaluation.cv_metrics) == {"mae", "rmse", "r2"}
    assert evaluation.residuals is not None
    assert isinstance(evaluation.promote, bool)


def test_module_run_writes_baseline_comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    baseline = BaselineResult(
        mode=AnalysisMode.PREDICTION,
        strategy="mean",
        metric="mae",
        value=10000.0,
        details={"mean": 2000.0},
        notes=["n"],
    )
    baseline_dir = Path(cfg.baseline.output_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    save_result(baseline, baseline_dir / cfg.baseline.output_file)

    training_module.run(cfg)
    evaluate_module.run(cfg)

    result_path = Path(cfg.evaluate.output_dir) / cfg.evaluate.output_file
    evaluation = EvaluationResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    assert evaluation.baseline is not None
    assert evaluation.baseline.metric == "mae"
    assert evaluation.baseline.improvement is not None
