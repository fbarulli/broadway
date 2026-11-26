"""Public behavior of the quantile interval adapter and the quantile bundle."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from broadway.calibration.metrics import coverage
from broadway.calibration.quantile import QuantileInterval
from broadway.training.models.lightgbm import QuantileLGBMRegressor


def test_quantile_interval_returns_the_ordered_pair_and_covers_nominal() -> None:
    truth = np.random.default_rng(7).normal(0.0, 1.0, 4000)
    lower = np.full_like(truth, -1.96)
    mid = np.zeros_like(truth)
    upper = np.full_like(truth, 1.96)

    low, high = QuantileInterval().interval(lower, mid, upper)

    np.testing.assert_array_equal(low, lower)
    np.testing.assert_array_equal(high, upper)
    assert coverage(truth, low, high) == pytest.approx(0.95, abs=0.02)


def test_quantile_interval_rejects_crossed_or_invalid_quantiles() -> None:
    adapter = QuantileInterval()
    with pytest.raises(ValueError, match="lower <= mid <= upper"):
        adapter.interval(np.array([2.0]), np.array([1.0]), np.array([3.0]))
    with pytest.raises(ValueError, match="same shape"):
        adapter.interval(np.zeros(2), np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError, match="one-dimensional"):
        adapter.interval(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="at least one"):
        adapter.interval(np.array([]), np.array([]), np.array([]))
    with pytest.raises(ValueError, match="non-finite"):
        adapter.interval(np.array([np.nan]), np.zeros(1), np.ones(1))


def test_quantile_lgbm_regressor_orders_its_three_quantiles() -> None:
    rng = np.random.default_rng(3)
    features = pd.DataFrame({"x": rng.normal(size=300)})
    target = 3.0 * features["x"].to_numpy() + rng.normal(0.0, 0.5, 300)
    model = QuantileLGBMRegressor(alpha=0.05, n_estimators=40, verbosity=-1)
    model.fit(features, target)

    lower, mid, upper = model.predict_interval(features)

    assert model.quantiles == (0.025, 0.5, 0.975)
    assert np.all(lower <= mid) and np.all(mid <= upper)
    np.testing.assert_allclose(model.predict(features), mid)
    assert model.get_params()["alpha"] == pytest.approx(0.05)
    assert model.set_params(n_estimators=10) is model


def test_quantile_lgbm_regressor_requires_a_valid_level_and_a_fit() -> None:
    with pytest.raises(ValueError, match="miscoverage"):
        QuantileLGBMRegressor(alpha=1.0)
    with pytest.raises(ValueError, match="not fitted"):
        QuantileLGBMRegressor(alpha=0.05).predict(pd.DataFrame({"x": [1.0]}))
