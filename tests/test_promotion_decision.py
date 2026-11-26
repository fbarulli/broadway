"""Promotion + HPO decisions: the interval decides, and the record is published."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from broadway.analysis.contracts import AnalysisContract, AnalysisMode
from broadway.config.schema import (
    ColumnRole,
    ColumnSchema,
    DatasetContract,
    DataSourceRef,
    EnvironmentConfig,
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
from broadway.evaluate.module import PromotionDecision
from broadway.features.plan import resolve_plan
from broadway.training.module import TrainingDecision


def _environment(tmp_path: Path) -> EnvironmentConfig:
    return EnvironmentConfig(
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


def _dataset() -> DatasetContract:
    return DatasetContract(
        name="synthetic",
        path="synthetic.parquet",
        target="price",
        task=TaskType.REGRESSION,
        datetime_column=None,
        columns={
            "num": ColumnSchema(dtype="float64", null_count=0, role=ColumnRole.FEATURE),
            "price": ColumnSchema(dtype="float64", null_count=0, role=ColumnRole.TARGET),
        },
        lookup_tables={},
    )


def _config(tmp_path: Path, hpo: HPOConfig | None = None) -> PipelineConfig:
    dataset = _dataset()
    experiment = ExperimentConfig(
        data_source=DataSourceRef(loader="canonical", schema_contract="raw"),
        feature_plan="broadway.features.plans:contract_numeric_features",
        model=ModelConfig(type="linear", params={}),
        split=SplitConfig(type="random", validation_size=0.2),
        random_state=42,
        target_metric="mae",
        hpo=hpo,
    )
    experiment.bind_plan(resolve_plan(experiment.feature_plan, dataset))
    return PipelineConfig(
        dataset=dataset,
        environment=_environment(tmp_path),
        experiment=experiment,
        evaluate=EvaluateStep(
            target_metric="mae",
            promotion_threshold=0.05,
            output_dir=str(tmp_path / "artifacts" / "evaluation"),
            output_file="metrics.json",
        ),
        train=TrainStep(
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
        ),
        analysis=AnalysisContract(
            name="test",
            mode=AnalysisMode.PREDICTION,
            goal="predict price",
            row_definition="one row",
            decision_moment="now",
            available_info=["num"],
            leakage_notes=[],
            success_criterion="beat the baseline",
        ),
    )


def test_promotion_needs_the_interval_to_clear_the_threshold(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    truth = np.linspace(1.0, 10.0, 400)
    candidate = truth + 1.0
    champion = truth + 5.0

    decision = PromotionDecision.measure(
        cfg, "mae", 1.0, 5.0, "models:/champion/1", truth, candidate, champion
    )
    published = decision.publish(Path(cfg.evaluate.output_dir))

    assert decision.promote is True
    assert published.verdict == "act"
    assert published.value == 0.8
    assert (Path(cfg.evaluate.output_dir) / "promotion_decision.json").exists()


def test_a_tiny_holdout_cannot_act_on_promotion(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    truth = np.arange(1.0, 6.0)
    decision = PromotionDecision.measure(
        cfg, "mae", 1.0, 5.0, "models:/champion/1", truth, truth + 1.0, truth + 3.0
    )

    assert decision.promote is False
    assert decision.decision().verdict == "insufficient"


def test_training_decision_publishes_hpo_vs_defaults(tmp_path: Path) -> None:
    hpo = HPOConfig(
        engine="optuna",
        direction="minimize",
        total_trials=2,
        initial_trials_per_model=1,
        top_k=1,
        target_metric="mae",
        models=[ModelHPOSpec(name="linear", search_space={})],
    )
    cfg = _config(tmp_path, hpo=hpo)
    generator = np.random.default_rng(0)
    X = pd.DataFrame({"num": generator.normal(size=60)})
    y = pd.Series(2.0 * X["num"] + generator.normal(scale=0.1, size=60))

    decision = TrainingDecision.measure(cfg, "linear", {}, {}, X, y, X, y)

    assert decision is not None
    published = decision.publish()
    assert published.sample_size == len(y)
    assert (Path(cfg.evaluate.output_dir).parent / "training" / "training_decision.json").exists()
