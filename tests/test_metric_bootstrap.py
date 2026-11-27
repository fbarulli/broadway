"""Percentile-bootstrap holdout metric intervals."""

from __future__ import annotations

import numpy as np
import pytest

from broadway.evaluate.contracts import BootstrapConfig
from broadway.evaluate.metrics import MetricBootstrap, compute_metrics


def test_bootstrap_intervals_bracket_the_point_estimates() -> None:
    rng = np.random.default_rng(3)
    y_true = rng.uniform(10.0, 60.0, size=500)
    y_pred = y_true + rng.normal(scale=5.0, size=500)
    config = BootstrapConfig(iterations=400, seed=7, level=0.95)
    intervals = MetricBootstrap(config).intervals(y_true, y_pred)
    point = compute_metrics(y_true, y_pred)

    assert set(intervals) == {"rmse", "mae", "mape"}
    for metric in ("rmse", "mae", "mape"):
        assert intervals[metric].lower < intervals[metric].upper
        assert intervals[metric].lower <= point[metric] <= intervals[metric].upper
    assert MetricBootstrap(config).intervals(y_true, y_pred) == intervals

    with pytest.raises(ValueError, match="one-dimensional"):
        MetricBootstrap(config).intervals(y_true.reshape(-1, 1), y_pred)
    with pytest.raises(ValueError, match="finite"):
        MetricBootstrap(config).intervals(np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="zero"):
        MetricBootstrap(config).intervals(np.zeros(10), np.ones(10))


def test_paired_improvement_interval_brackets_the_point_improvement() -> None:
    rng = np.random.default_rng(11)
    truth = rng.uniform(10.0, 60.0, size=400)
    baseline = truth + rng.normal(scale=8.0, size=400)
    candidate = truth + rng.normal(scale=3.0, size=400)
    config = BootstrapConfig(iterations=300, seed=5, level=0.95)

    interval = MetricBootstrap(config).improvement_interval(truth, candidate, baseline, "rmse")

    point = compute_metrics(truth, baseline)["rmse"] - compute_metrics(truth, candidate)["rmse"]
    assert interval.lower <= point <= interval.upper
    with pytest.raises(ValueError, match="improvement metric"):
        MetricBootstrap(config).improvement_interval(truth, candidate, baseline, "bogus")
    with pytest.raises(ValueError, match="same shape"):
        MetricBootstrap(config).improvement_interval(truth, candidate[:-1], baseline, "rmse")
    with pytest.raises(ValueError, match="must be finite"):
        MetricBootstrap(config).improvement_interval(
            truth, candidate, np.full_like(baseline, np.nan), "rmse"
        )
