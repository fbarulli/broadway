"""Public behavior of split-conformal interval calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from broadway.analysis.contracts import AnalysisContract, AnalysisMode
from broadway.calibration import CALIBRATION_FILE
from broadway.calibration.conformal import SplitConformal
from broadway.calibration.contracts import (
    CalibrationConfig,
    PeriodCoverage,
)
from broadway.calibration.metrics import coverage, coverage_by_period
from broadway.config.schema import (
    ColumnRole,
    ColumnSchema,
    DatasetContract,
    DataSourceRef,
    EnvironmentConfig,
    EtlStep,
    EvaluateStep,
    ExperimentConfig,
    HPOConfig,
    ModelConfig,
    ModelHPOSpec,
    PipelineConfig,
    SplitConfig,
    TaskType,
    TrainStep,
)
from broadway.evaluate import module as evaluate_module
from broadway.evaluate.contracts import EvaluationResult
from broadway.features.plan import resolve_plan
from broadway.lineage import records
from broadway.training import module as training_module
from broadway.training.contracts import TrainingResult


def test_split_conformal_holdout_coverage_matches_nominal() -> None:
    rng = np.random.default_rng(11)
    calibration_residuals = rng.normal(size=4000)
    q = SplitConformal().quantile(calibration_residuals, alpha=0.2)
    holdout_true = rng.normal(size=4000)
    lower, upper = SplitConformal().interval(np.zeros_like(holdout_true), q)
    assert coverage(holdout_true, lower, upper) == pytest.approx(0.8, abs=0.04)


def test_split_conformal_quantile_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    residuals = rng.normal(size=500)
    conformal = SplitConformal()
    first = conformal.quantile(residuals, alpha=0.1)
    assert conformal.quantile(residuals, alpha=0.1) == first
    assert conformal.quantile(rng.permutation(residuals), alpha=0.1) == first


def test_split_conformal_quantile_grows_as_alpha_decreases() -> None:
    residuals = np.random.default_rng(5).normal(size=1000)
    conformal = SplitConformal()
    quantiles = [conformal.quantile(residuals, alpha=alpha) for alpha in (0.3, 0.2, 0.1, 0.05)]
    assert quantiles == sorted(quantiles)
    assert len(set(quantiles)) == len(quantiles)


def test_coverage_by_period_labels_iso_weeks() -> None:
    index = pd.DatetimeIndex(["2024-03-04", "2024-03-05", "2024-03-11"])
    periods = coverage_by_period(
        index,
        np.array([0.0, 5.0, 0.0]),
        np.zeros(3),
        np.ones(3),
    )
    assert periods == [
        PeriodCoverage(period="2024-W10", n=2, coverage=0.5),
        PeriodCoverage(period="2024-W11", n=1, coverage=1.0),
    ]


def test_coverage_by_period_excludes_partial_periods() -> None:
    index = pd.DatetimeIndex(["2024-03-04", "2024-03-05", "2024-03-11"])
    periods = coverage_by_period(
        index,
        np.array([0.0, 5.0, 0.0]),
        np.zeros(3),
        np.ones(3),
        min_rows=2,
    )
    assert periods == [PeriodCoverage(period="2024-W10", n=2, coverage=0.5)]


def _make_config(tmp_path: Path) -> PipelineConfig:
    environment = EnvironmentConfig(
        log_level="INFO",
        data_dir=str(tmp_path / "data"),
        raw_subdir="raw",
        processed_subdir="processed",
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
        feature_plan="broadway.features.plans:contract_numeric_features",
        model=ModelConfig(type="linear", params={}),
        split=SplitConfig(type="random", validation_size=0.2),
        random_state=42,
        target_metric="rmse",
        calibration=CalibrationConfig(method="split_conformal", alpha=0.2, calibration_size=0.25),
    )
    experiment.bind_plan(resolve_plan(experiment.feature_plan, dataset))
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
    evaluate_step = EvaluateStep(
        target_metric="rmse",
        promotion_threshold=0.05,
        coverage_tolerance=0.3,
        output_dir=str(tmp_path / "artifacts" / "evaluation"),
        output_file="metrics.json",
    )
    return PipelineConfig(
        dataset=dataset,
        environment=environment,
        experiment=experiment,
        etl=etl,
        train=train_step,
        evaluate=evaluate_step,
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


def test_calibration_payload_reaches_holdout_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_config(tmp_path)
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")
    df = pd.DataFrame(
        {
            "rooms": np.arange(1, 61),
            "area": np.arange(40, 100),
            "price": np.arange(1, 61) * 100.0,
        }
    )
    # Disjoint fit/calibration (from the 48-row train) and holdout (last 12).
    _write_features(cfg, df.iloc[:48], df.iloc[48:])

    training_module.run(cfg)
    evaluate_module.run(cfg)

    training = TrainingResult.model_validate_json(
        (Path(cfg.train.output_dir) / cfg.train.output_file).read_text(encoding="utf-8")
    )
    assert training.calibration is not None
    # 25% of 48 = 12 calibration rows, never the 12 holdout rows: a leak that
    # fitted q on the holdout would report 12 from the wrong frame only if the
    # counts differed, so the disjoint construction above is load-bearing.
    assert training.calibration.n_calibration == 12

    evaluation = EvaluationResult.model_validate_json(
        (Path(cfg.evaluate.output_dir) / cfg.evaluate.output_file).read_text(encoding="utf-8")
    )
    assert evaluation.calibration is not None
    assert evaluation.calibration.q == training.calibration.q
    assert evaluation.calibration.mean_width == pytest.approx(2 * training.calibration.q)
    assert evaluation.calibration.calibrated is not None
    assert (Path(cfg.train.output_dir) / CALIBRATION_FILE).exists()
    assert (Path(cfg.evaluate.output_dir) / CALIBRATION_FILE).exists()


def test_calibration_rejects_hpo(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    assert cfg.experiment is not None
    cfg.experiment.hpo = HPOConfig(
        engine="optuna",
        direction="minimize",
        total_trials=1,
        initial_trials_per_model=1,
        top_k=1,
        target_metric="rmse",
        models=[ModelHPOSpec(name="linear")],
    )
    with pytest.raises(ValueError, match="calibration with hpo"):
        training_module.run(cfg)


def test_calibration_rejects_a_classification_task(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    assert cfg.dataset is not None
    cfg.dataset.task = TaskType.CLASSIFICATION
    with pytest.raises(ValueError, match="regression-only"):
        training_module.run(cfg)


def test_calibration_flags_a_tolerance_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_config(tmp_path)
    assert cfg.evaluate is not None
    cfg.evaluate.coverage_tolerance = 0.0001
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")
    df = pd.DataFrame(
        {
            "rooms": np.arange(1, 61),
            "area": np.arange(40, 100),
            "price": np.arange(1, 61) * 100.0,
        }
    )
    holdout = df.iloc[48:].copy()
    holdout["price"] += 1000.0  # shift the holdout outside the fitted intervals
    _write_features(cfg, df.iloc[:48], holdout)

    training_module.run(cfg)
    evaluate_module.run(cfg)

    evaluation = EvaluationResult.model_validate_json(
        (Path(cfg.evaluate.output_dir) / cfg.evaluate.output_file).read_text(encoding="utf-8")
    )
    assert evaluation.calibration is not None
    assert evaluation.calibration.calibrated is False
    assert any("calibration coverage" in warning for warning in evaluation.warnings)


def _write_features(cfg: PipelineConfig, train: pd.DataFrame, val: pd.DataFrame) -> None:
    assert cfg.etl is not None
    out_dir = Path(cfg.environment.data_dir) / cfg.environment.processed_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out_dir / cfg.etl.train_features_file, index=False)
    val.to_parquet(out_dir / cfg.etl.val_features_file, index=False)
