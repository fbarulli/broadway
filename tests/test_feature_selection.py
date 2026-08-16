"""Tests for the RFE curve helper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from broadway.evaluate.feature_selection import rfe_curve


def _make_synthetic() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(42)
    n = 60
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = 2.0 * x1 - 1.0 * x2 + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "noise": noise}), y


def test_rfe_curve_returns_aligned_curves() -> None:
    X, y = _make_synthetic()
    result = rfe_curve(LinearRegression(), X, y)

    n_features = result["n_features"]
    scores = result["score"]
    assert len(n_features) == len(scores) >= 3
    assert n_features == sorted(n_features)
    assert n_features[0] == 1
    assert n_features[-1] == X.shape[1]


def test_rfe_curve_full_model_beats_noise_only() -> None:
    X, y = _make_synthetic()
    result = rfe_curve(LinearRegression(), X, y)

    # Score is a positive error metric (lower = better): the full-feature
    # model must not be worse than the single-feature subset.
    full_score = result["score"][-1]
    smallest_subset_score = result["score"][0]
    assert full_score <= smallest_subset_score

    # Explicit noise-only baseline: a model trained on the pure-noise column
    # alone must be strictly worse than the full-feature model.
    noise_only = LinearRegression().fit(X[["noise"]], y)
    noise_only_score = mean_absolute_error(y, noise_only.predict(X[["noise"]]))
    assert full_score < noise_only_score


def test_rfe_curve_rejects_non_numeric_X() -> None:
    X, y = _make_synthetic()
    X["category"] = "a"
    with pytest.raises(ValueError, match="numeric"):
        rfe_curve(LinearRegression(), X, y)
