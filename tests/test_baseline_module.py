from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from broadway.analysis.contracts import AnalysisMode
from broadway.baseline import causal, hypothesis, prediction, module
from broadway.baseline.contracts import BaselineResult, load_result, save_result
from broadway.config.loader import load_config
from broadway.config.schema import CausalStep, TaskType


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
    cfg = load_config("baseline", dataset="taxi", analysis="taxi")
    assert cfg.analysis is not None
    cfg = cfg.model_copy(
        update={"baseline": cfg.baseline.model_copy(update={"output_dir": str(tmp_path)})}
    )
    monkeypatch.setattr(
        module, "load", lambda dataset: pd.DataFrame({"trip_duration_minutes": [10.0, 20.0, 30.0]})
    )
    module.run(cfg)
    p = tmp_path / cfg.baseline.output_file
    assert p.exists()
    r = load_result(p)
    assert r.mode == AnalysisMode.PREDICTION
