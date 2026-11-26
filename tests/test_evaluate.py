"""Evaluate step error-analysis artifacts: per-row ledger and threshold report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from broadway.analysis.contracts import AnalysisContract, AnalysisMode
from broadway.config.schema import (
    ColumnRole,
    ColumnSchema,
    DatasetContract,
    DataSourceRef,
    EnvironmentConfig,
    EtlStep,
    EvaluateStep,
    ExperimentConfig,
    ModelConfig,
    PipelineConfig,
    SplitConfig,
    TaskType,
    TrainStep,
)
from broadway.evaluate import module as evaluate_module
from broadway.evaluate.contracts import (
    DecisionConfig,
    DecisionReport,
    EvaluationResult,
    ThresholdGrid,
)
from broadway.features.plan import resolve_plan
from broadway.lineage import records
from broadway.reports.decision import DecisionRule
from broadway.training import module as training_module


def _make_config(tmp_path: Path, decision: DecisionConfig | None = None) -> PipelineConfig:
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
        hpo=None,
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
        output_dir=str(tmp_path / "artifacts" / "evaluation"),
        output_file="metrics.json",
        decision=decision,
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


def _run(cfg: PipelineConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(records, "LINEAGE_DIR", Path(cfg.environment.data_dir) / "lineage")
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


def test_evaluate_writes_error_ledger_without_inventing_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_config(tmp_path)
    _run(cfg, monkeypatch)

    eval_dir = Path(cfg.evaluate.output_dir)
    ledger = pd.read_parquet(eval_dir / cfg.evaluate.error_analysis.ledger_file)
    assert len(ledger) == 10
    assert set(ledger.columns) == {"y_true", "y_pred", "residual", "abs_error", "rooms", "area"}
    np.testing.assert_allclose(ledger["residual"], ledger["y_pred"] - ledger["y_true"])
    np.testing.assert_allclose(ledger["abs_error"], ledger["residual"].abs())
    assert not (eval_dir / cfg.evaluate.error_analysis.thresholds_file).exists()


def test_evaluate_writes_threshold_report_only_when_decision_declared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision = DecisionConfig(
        threshold=3500.0,
        objective="f1",
        grid=ThresholdGrid(min=2500.0, max=4500.0, steps=5),
    )
    cfg = _make_config(tmp_path, decision)
    _run(cfg, monkeypatch)

    report_path = Path(cfg.evaluate.output_dir) / cfg.evaluate.error_analysis.thresholds_file
    report = DecisionReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    assert report.column == cfg.dataset.target
    assert report.label_threshold == decision.threshold
    assert len(report.rows) == decision.grid.steps
    assert report.rows[0].threshold == pytest.approx(decision.grid.min)
    assert report.rows[-1].threshold == pytest.approx(decision.grid.max)
    assert any(report.chosen_threshold == pytest.approx(row.threshold) for row in report.rows)


def test_evaluate_uses_the_configured_quantile_interval_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_config(tmp_path)
    assert cfg.evaluate is not None and cfg.experiment is not None
    cfg.evaluate.interval_source = "quantile"
    monkeypatch.setattr(records, "LINEAGE_DIR", Path(cfg.environment.data_dir) / "lineage")
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
    # The interval level is the model's own alpha (95%), read from the config.
    cfg.experiment.model = ModelConfig(type="lgbm_quantile", params={"alpha": 0.05})

    class _QuantileCandidate:
        def predict(self, features: pd.DataFrame) -> np.ndarray:
            return np.full(len(features), 2000.0)

        def predict_interval(
            self, features: pd.DataFrame
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            size = len(features)
            return (
                np.full(size, 1500.0),
                np.full(size, 2000.0),
                np.full(size, 2500.0),
            )

    monkeypatch.setattr(
        evaluate_module, "_load_candidate", lambda result: _QuantileCandidate()
    )
    evaluate_module.run(cfg)

    evaluation = EvaluationResult.model_validate_json(
        (Path(cfg.evaluate.output_dir) / cfg.evaluate.output_file).read_text(encoding="utf-8")
    )
    calibration = evaluation.calibration
    assert calibration is not None
    assert calibration.method == "quantile"
    assert calibration.q is None
    assert calibration.alpha == pytest.approx(0.05)
    assert calibration.nominal_coverage == pytest.approx(0.95)
    assert calibration.mean_width == pytest.approx(1000.0)


def test_evaluate_quantile_source_requires_a_quantile_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_config(tmp_path)
    assert cfg.evaluate is not None and cfg.experiment is not None
    cfg.evaluate.interval_source = "quantile"
    monkeypatch.setattr(records, "LINEAGE_DIR", Path(cfg.environment.data_dir) / "lineage")
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
    cfg.experiment.model = ModelConfig(type="lgbm_quantile", params={"alpha": 0.05})

    with pytest.raises(ValueError, match="predict_interval"):
        evaluate_module.run(cfg)


def test_evaluate_publishes_time_series_cv_and_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_config(tmp_path)
    assert cfg.train is not None and cfg.evaluate is not None
    cfg.train.cv_kind = "expanding"
    cfg.train.cv_gap = 1
    cfg.train.cv_folds = 2
    cfg.evaluate.cv_decision = DecisionRule(
        metric="rmse",
        direction="max",
        min_effect=0.01,
        min_sample_size=5,
        action="ship the candidate",
        owner="test-owner",
    )
    _run(cfg, monkeypatch)

    evaluation = EvaluationResult.model_validate_json(
        (Path(cfg.evaluate.output_dir) / cfg.evaluate.output_file).read_text(encoding="utf-8")
    )
    assert evaluation.cv_detail is not None
    assert evaluation.cv_detail.kind == "expanding"
    assert evaluation.cv_detail.time_indexed is False
    assert evaluation.cv_metrics == evaluation.cv_detail.metrics
    assert evaluation.cv_decision is not None
    assert evaluation.cv_decision.owner == "test-owner"
