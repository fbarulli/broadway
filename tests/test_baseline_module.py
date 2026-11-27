from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from broadway.analysis.contracts import AnalysisContract, AnalysisMode
from broadway.baseline import causal, hypothesis, module, prediction
from broadway.baseline.contracts import BaselineResult, load_result, save_result
from broadway.baseline.improvement import improvement_vs_baseline
from broadway.config.loader import load_config
from broadway.config.schema import CausalStep, PipelineConfig, TaskType
from broadway.lineage import records
from broadway.reports.decision import Decision


def test_contract_round_trip(tmp_path: Path) -> None:
    r = BaselineResult(
        mode=AnalysisMode.PREDICTION,
        strategy="mean",
        metric="mae",
        value=1.5,
        details={"mean": 10.0},
        notes=["n"],
    )
    save_result(r, tmp_path / "b.json")
    assert load_result(tmp_path / "b.json") == r


def test_prediction_regression_baseline() -> None:
    df = pd.DataFrame({"t": [10.0, 20.0, 30.0]})
    r = prediction.run(df, "t", TaskType.REGRESSION)
    assert r.mode == AnalysisMode.PREDICTION
    assert r.strategy == "mean"
    assert r.metric == "mae"
    assert r.value == pytest.approx(6.6666666667, rel=1e-3)
    assert r.details["mean"] == 20.0


def test_prediction_classification_baseline() -> None:
    df = pd.DataFrame({"t": [0.0, 0.0, 1.0]})
    r = prediction.run(df, "t", TaskType.CLASSIFICATION)
    assert r.strategy == "naive_prevalence"
    assert r.metric == "pr_auc"
    assert r.value == pytest.approx(1 / 3)
    assert r.details["prevalence"] == pytest.approx(1 / 3)
    assert r.details["bad"] == 1


def test_hypothesis_baseline() -> None:
    df = pd.DataFrame(
        {"g": ["a", "a", "a", "b", "b", "b"], "t": [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]}
    )
    r = hypothesis.run(df, "t", "g", ["a", "b"])
    assert r.metric == "mean_difference"
    assert r.value == pytest.approx(10.0)


def test_causal_baseline() -> None:
    c = CausalStep(
        treatment_column="trt",
        outcome_column="out",
        power=0.8,
        alpha=0.05,
        effect_size=0.5,
        output_dir="artifacts/causal",
        output_file="design.json",
    )
    r = causal.run(c)
    assert r.mode == AnalysisMode.CAUSAL
    assert r.metric == "sample_size"
    assert r.value > 0
    assert r.details["mde"] > 0


def test_module_requires_analysis() -> None:
    cfg = load_config("baseline")
    with pytest.raises(ValueError):
        module.run(cfg)


def test_module_dispatch_prediction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config("baseline", dataset="test", analysis="test")
    assert cfg.analysis is not None
    cfg = cfg.model_copy(
        update={"baseline": cfg.baseline.model_copy(update={"output_dir": str(tmp_path)})}
    )
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")
    monkeypatch.setattr(
        module, "load", lambda dataset: pd.DataFrame({"target": [10.0, 20.0, 30.0]})
    )
    module.run(cfg)
    p = tmp_path / cfg.baseline.output_file
    assert p.exists()
    r = load_result(p)
    assert r.mode == AnalysisMode.PREDICTION
    assert r.trace is not None
    assert r.trace.dataset == "test"
    assert r.trace.analysis_goal == "predict target"
    assert r.trace.commit


# --- baseline step guard rails ----------------------------------------------


def test_git_commit_falls_back_to_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outside a git work tree the trace records 'unknown' instead of lying."""
    import subprocess

    def boom(*a: object, **k: object) -> None:
        raise subprocess.CalledProcessError(128, ["git"])

    monkeypatch.setattr(module.subprocess, "run", boom)
    assert module._git_commit() == "unknown"

    def missing(*a: object, **k: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(module.subprocess, "run", missing)
    assert module._git_commit() == "unknown"


def test_baseline_requires_baseline_config() -> None:
    cfg = load_config("baseline", dataset="test", experiment="baseline", analysis="test_hypothesis")
    no_baseline = cfg.model_copy(update={"baseline": None})
    with pytest.raises(ValueError, match="baseline step requires baseline config"):
        module.run(no_baseline)


def test_prediction_baseline_requires_dataset_config() -> None:
    cfg = load_config("baseline", dataset="test", analysis="test")
    no_dataset = cfg.model_copy(update={"dataset": None})
    with pytest.raises(ValueError, match="prediction baseline requires a dataset config"):
        module._compute_baseline(no_dataset)


def test_hypothesis_baseline_requires_dataset_and_hypothesis_group() -> None:
    cfg = load_config("baseline", dataset="test", experiment="baseline", analysis="test_hypothesis")
    no_dataset = cfg.model_copy(update={"dataset": None})
    with pytest.raises(ValueError, match="hypothesis baseline requires a dataset config"):
        module._compute_baseline(no_dataset)

    no_group = cfg.model_copy(
        update={"analysis": cfg.analysis.model_copy(update={"hypothesis": None})}
    )
    with pytest.raises(
        ValueError, match="requires an analysis contract with a hypothesis group"
    ):
        module._compute_baseline(no_group)


def test_module_dispatch_causal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The causal arm dispatches through causal.run with the merged CausalStep
    config and persists a sample-size baseline result."""
    cfg = load_config("baseline", dataset="test", experiment="baseline", analysis="test_causal")
    assert cfg.analysis is not None
    cfg = cfg.model_copy(
        update={
            "causal": CausalStep(
                treatment_column="trt",
                outcome_column="out",
                power=0.8,
                alpha=0.05,
                effect_size=0.5,
                output_dir=str(tmp_path),
                output_file="design.json",
            ),
            "baseline": cfg.baseline.model_copy(update={"output_dir": str(tmp_path)}),
        }
    )
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")

    module.run(cfg)

    saved = load_result(tmp_path / cfg.baseline.output_file)
    assert saved.mode == AnalysisMode.CAUSAL
    assert saved.metric == "sample_size"


def test_hypothesis_baseline_single_observation_group_std_is_nan() -> None:
    """n=1 groups report std=NaN in the persisted details (pandas ddof=1
    semantics), never a fake 0.0."""
    import math

    df = pd.DataFrame({"g": ["a", "a", "b"], "t": [1.0, 2.0, 7.0]})
    r = hypothesis.run(df, "t", "g", ["a", "b"])
    means = r.details["group_means"]
    assert means["b"]["count"] == 1
    assert math.isnan(means["b"]["std"])
    assert means["a"]["std"] == pytest.approx(0.5 ** 0.5)
    assert r.value == pytest.approx(5.5)  # mean(b)=7.0 - mean(a)=1.5 range


def test_improvement_regression_lower_is_better() -> None:
    b = BaselineResult(
        mode=AnalysisMode.PREDICTION,
        strategy="mean",
        metric="mae",
        value=10.0,
        details={},
        notes=[],
    )
    assert improvement_vs_baseline(8.0, b, TaskType.REGRESSION) == pytest.approx(0.2)
    assert improvement_vs_baseline(12.0, b, TaskType.REGRESSION) == pytest.approx(-0.2)


def test_improvement_classification_higher_is_better() -> None:
    b = BaselineResult(
        mode=AnalysisMode.PREDICTION,
        strategy="majority_class",
        metric="accuracy",
        value=0.5,
        details={},
        notes=[],
    )
    assert improvement_vs_baseline(0.6, b, TaskType.CLASSIFICATION) == pytest.approx(0.2)


def test_improvement_zero_baseline_returns_none() -> None:
    b = BaselineResult(
        mode=AnalysisMode.PREDICTION,
        strategy="mean",
        metric="mae",
        value=0.0,
        details={},
        notes=[],
    )
    assert improvement_vs_baseline(1.0, b, TaskType.REGRESSION) is None


def _classification_baseline_cfg(tmp_path: Path) -> PipelineConfig:
    from broadway.config.schema import (
        BaselineStep,
        BootstrapConfig,
        ColumnRole,
        ColumnSchema,
        DatasetContract,
        DataSourceRef,
        EnvironmentConfig,
        EtlStep,
        ExperimentConfig,
        ModelConfig,
        SplitConfig,
    )
    from broadway.features.plan import resolve_plan

    dataset = DatasetContract(
        name="synthetic_cls",
        path="synthetic.parquet",
        target="bad",
        task=TaskType.CLASSIFICATION,
        datetime_column=None,
        columns={
            "rooms": ColumnSchema(dtype="int64", null_count=0, role=ColumnRole.FEATURE),
            "area": ColumnSchema(dtype="int64", null_count=0, role=ColumnRole.FEATURE),
            "bad": ColumnSchema(dtype="float64", null_count=0, role=ColumnRole.TARGET),
        },
        lookup_tables={},
    )
    experiment = ExperimentConfig(
        data_source=DataSourceRef(loader="canonical", schema_contract="engineered"),
        feature_plan="broadway.features.plans:contract_numeric_features",
        model=ModelConfig(type="linear", params={}),
        split=SplitConfig(type="stratified", validation_size=0.2),
        random_state=42,
        target_metric="pr_auc",
    ).bind_plan(resolve_plan("broadway.features.plans:contract_numeric_features", dataset))
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
    baseline = BaselineStep(
        output_dir=str(tmp_path / "baseline"),
        output_file="baseline.json",
        linear_reference_model="linear",
        bootstrap=BootstrapConfig(iterations=30, seed=1, level=0.95),
        null_iterations=3,
        min_effect=0.01,
        min_sample_size=10,
        action="ship only a model that clears the prevalence floor",
        owner="synthetic",
    )
    return PipelineConfig(
        analysis=AnalysisContract(
            name="synthetic_cls",
            mode=AnalysisMode.PREDICTION,
            goal="predict a binary outcome",
            row_definition="one row per entity",
            decision_moment="before the outcome",
            available_info=["rooms", "area"],
            leakage_notes=["post-outcome columns are excluded"],
            success_criterion="beat the prevalence on pr_auc",
        ),
        dataset=dataset,
        environment=environment,
        experiment=experiment,
        etl=EtlStep(
            ci_sample_size=0,
            max_drop_fraction=0.5,
            random_state=42,
            train_file="train.parquet",
            val_file="val.parquet",
            training_data_file="training_data.parquet",
            train_features_file="train_features.parquet",
            val_features_file="val_features.parquet",
            missing_encodings=[""],
        ),
        baseline=baseline,
    )


def test_classification_baseline_publishes_prevalence_and_pr_auc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    cfg = _classification_baseline_cfg(tmp_path)
    directory = tmp_path / "data" / "processed"
    directory.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rooms = rng.integers(1, 6, 300)
    area = rng.integers(40, 120, 300)
    bad = (rooms >= 4).astype(float)
    frame = pd.DataFrame({"rooms": rooms, "area": area, "bad": bad})
    frame.iloc[:240].to_parquet(directory / "train_features.parquet", index=False)
    frame.iloc[240:].to_parquet(directory / "val_features.parquet", index=False)
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")

    module.run(cfg)

    assert cfg.baseline is not None
    result = load_result(tmp_path / "baseline" / "baseline.json")
    assert result.metric == "pr_auc"
    assert result.details["holdout_rows"] == 60
    assert 0.0 < result.value < 1.0
    record = Decision.model_validate_json(
        (tmp_path / "baseline" / "baseline_decision.json").read_text(encoding="utf-8")
    )
    assert record.metric == "improvement_over_naive_pr_auc"
    assert record.direction == "max"
    assert record.sample_size == 60
    assert record.null_threshold is not None
    assert {segment.segment for segment in record.evidence.segments} == {
        "first_half",
        "second_half",
    }
    assert record.evidence.provenance["prevalence"]
