from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from broadway.analysis.contracts import AnalysisMode
from broadway.baseline import causal, hypothesis, module, prediction
from broadway.baseline.contracts import BaselineResult, load_result, save_result
from broadway.baseline.improvement import improvement_vs_baseline
from broadway.config.loader import load_config
from broadway.config.schema import CausalStep, TaskType
from broadway.lineage import records


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
    df = pd.DataFrame({"t": ["a", "a", "b"]})
    r = prediction.run(df, "t", TaskType.CLASSIFICATION)
    assert r.strategy == "majority_class"
    assert r.metric == "accuracy"
    assert r.value == pytest.approx(2 / 3)


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
