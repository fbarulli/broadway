from __future__ import annotations

import numpy as np
import pytest

from broadway.stats.decompose import DecomposeModel, decompose, lowess_trend


def test_stl_decompose_returns_full_length_components_and_a_constant_residual() -> None:
    period = 24
    n = period * 10
    index = np.arange(n, dtype=float)
    series = 10.0 + index / 50.0 + 3.0 * np.sin(2.0 * np.pi * index / period) + 5.0

    result = decompose(series, period=period, model=DecomposeModel.ADDITIVE)

    assert len(result.trend) == len(result.seasonal) == len(result.resid) == n
    assert result.period == period
    assert np.std(result.resid) < 1e-6
    assert 0.0 <= result.seasonal_strength <= 1.0
    assert 0.0 <= result.trend_strength <= 1.0


def test_multiplicative_decompose_returns_ratio_components() -> None:
    period = 24
    index = np.arange(period * 10, dtype=float)
    series = (100.0 + index / 10.0) * (1.0 + 0.2 * np.sin(2.0 * np.pi * index / period))

    result = decompose(series, period=period, model="multiplicative")

    assert len(result.trend) == len(result.seasonal) == len(result.resid) == index.size
    assert np.mean(result.resid) == pytest.approx(1.0, abs=0.05)
    assert np.all(np.array(result.seasonal) > 0.0)


def test_lowess_trend_returns_one_smoothed_value_per_observation() -> None:
    series = np.linspace(0.0, 1.0, 100) ** 2

    trend = lowess_trend(series, frac=0.5)

    assert len(trend.trend) == series.size
    assert trend.trend[50] < trend.trend[-1]
