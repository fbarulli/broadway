from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from broadway.evaluate.comparison import compare_models
from broadway.evaluate.contracts import EvaluationResult
from broadway.evaluate.validation import cross_validate, residual_summary


def test_evaluation_result_json_round_trip() -> None:
    result = EvaluationResult(
        metrics={"mae": 0.5, "rmse": 0.8, "r2": 0.9},
        promote=True,
        reason="improvement over champion",
    )
    data = result.model_dump_json()
    assert json.loads(data) == {
        "metrics": {"mae": 0.5, "rmse": 0.8, "r2": 0.9},
        "promote": True,
        "reason": "improvement over champion",
    }
    assert EvaluationResult.model_validate_json(data) == result


def test_compare_models_champion_none() -> None:
    candidate = {"mae": 0.5, "rmse": 0.8, "r2": 0.9}
    result = compare_models(candidate, None)
    assert set(result) == set(candidate)
    for metric, value in candidate.items():
        assert result[metric]["candidate"] == value
        assert result[metric]["champion"] is None
        assert result[metric]["delta"] is None
        assert result[metric]["delta_pct"] is None


def test_compare_models_with_champion() -> None:
    candidate = {"mae": 0.5, "rmse": 0.8, "r2": 0.9}
    champion = {"mae": 0.4, "rmse": 1.0}
    result = compare_models(candidate, champion)

    mae = result["mae"]
    assert mae["candidate"] == 0.5
    assert mae["champion"] == 0.4
    assert mae["delta"] == pytest.approx(0.1)
    assert mae["delta_pct"] == pytest.approx(0.25)

    rmse = result["rmse"]
    assert rmse["candidate"] == 0.8
    assert rmse["champion"] == 1.0
    assert rmse["delta"] == pytest.approx(-0.2)
    assert rmse["delta_pct"] == pytest.approx(-0.2)

    r2 = result["r2"]
    assert r2["candidate"] == 0.9
    assert r2["champion"] is None
    assert r2["delta"] is None
    assert r2["delta_pct"] is None


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
