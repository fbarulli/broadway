"""cross_validate tests — sklearn.model_selection.cross_validate swap.

The golden dict below was captured from the pre-swap hand-written KFold
loop on the fixture (sklearn 1.7.2). Hardcoded goldens are intentional
brittleness: a sklearn-version shift must fail loud.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from broadway.evaluate.validation import _mean_metrics, cross_validate


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    return make_regression(n_samples=300, n_features=5, noise=10.0, random_state=42)


def test_cross_validate_matches_hand_loop_golden() -> None:
    X, y = _fixture()
    result = cross_validate(LinearRegression(), X, y, cv_folds=5, random_state=42)
    assert result == {
        "mae": 7.7527,
        "rmse": 9.8213,
        "r2": 0.9912,
        "mape": 0.2867,
        "max_error": 25.1997,
        "median_ae": 6.3965,
        "explained_var": 0.9913,
    }


def test_mean_metrics_sign_and_rounding() -> None:
    scores = {
        "test_mae": np.array([-1.111, -2.222]),
        "test_rmse": np.array([-3.0, -4.0]),
        "test_r2": np.array([0.5, 0.7]),
    }
    assert _mean_metrics(scores, decimals=2) == {"mae": 1.67, "rmse": 3.5, "r2": 0.6}


def test_cross_validate_reproducible() -> None:
    X, y = _fixture()
    first = cross_validate(LinearRegression(), X, y, cv_folds=5, random_state=42)
    second = cross_validate(LinearRegression(), X, y, cv_folds=5, random_state=42)
    assert first == second


def test_cross_validate_different_seed_differs() -> None:
    X, y = _fixture()
    seed_1 = cross_validate(LinearRegression(), X, y, cv_folds=5, random_state=1)
    seed_2 = cross_validate(LinearRegression(), X, y, cv_folds=5, random_state=2)
    assert seed_1 != seed_2


def test_cross_validate_insufficient_samples_fails_loud() -> None:
    X, y = make_regression(n_samples=3, n_features=5, noise=10.0, random_state=42)
    with pytest.raises(ValueError):
        cross_validate(LinearRegression(), X, y, cv_folds=5, random_state=42)


def test_cross_validate_output_types_are_json_safe() -> None:
    X, y = _fixture()
    result = cross_validate(LinearRegression(), X, y, cv_folds=5, random_state=42)
    for value in result.values():
        assert isinstance(value, float)
        assert not isinstance(value, np.floating)
    json.dumps(result)


def test_cross_validate_accepts_pipeline_not_just_bare_estimator() -> None:
    X, y = _fixture()
    pipeline = Pipeline([("passthrough", "passthrough"), ("model", LinearRegression())])
    result = cross_validate(pipeline, X, y, cv_folds=5, random_state=42)
    assert set(result) == {
        "mae",
        "rmse",
        "r2",
        "mape",
        "max_error",
        "median_ae",
        "explained_var",
    }
