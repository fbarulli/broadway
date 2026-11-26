from __future__ import annotations

import numpy as np
import pytest

from broadway.stats.time_series import (
    acf,
    adf,
    cross_correlation,
    difference,
    durbin_watson,
    first_values,
    inverse_difference,
    ljung_box,
    pacf,
    plot_acf,
)


def _ar1(n: int, phi: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, n)
    series = np.zeros(n)
    series[0] = noise[0]
    for i in range(1, n):
        series[i] = phi * series[i - 1] + noise[i]
    return series


def test_acf_and_pacf_return_one_value_per_lag() -> None:
    series = _ar1(500, 0.7, seed=3)

    autocorrelation = acf(series, lags=24)
    partial = pacf(series, lags=24)

    assert autocorrelation.shape == (25,)
    assert partial.shape == (25,)
    assert autocorrelation[0] == pytest.approx(1.0)
    assert abs(autocorrelation[1]) > 0.5


def test_cross_correlation_peaks_at_the_lead_lag() -> None:
    x = np.random.default_rng(8).normal(0.0, 1.0, 500)
    y = np.concatenate([x[3:], np.zeros(3)])  # y leads x by three observations

    values = cross_correlation(x, y, lags=6)

    assert values.shape == (7,)
    assert int(np.argmax(values)) == 3
    assert values[3] > 0.95


def test_ljung_box_returns_pvalue_for_autocorrelated_series() -> None:
    result = ljung_box(_ar1(500, 0.8, seed=4), lags=10)

    assert result.lags == 10
    assert result.statistic > 0.0
    assert result.p_value < 1e-3


def test_adf_reports_used_lag_and_observations() -> None:
    result = adf(np.random.default_rng(5).normal(0.0, 1.0, 500))

    assert result.n_obs > 0
    assert result.used_lag >= 0
    assert 0.0 <= result.p_value <= 1.0
    assert "5%" in result.critical_values


def test_difference_inverse_round_trip_is_exact() -> None:
    rng = np.random.default_rng(6)
    series = np.cumsum(np.cumsum(rng.normal(0.0, 1.0, 200)))

    diffed = difference(series, order=2)
    restored = inverse_difference(diffed, first_values(series, order=2), order=2)

    assert diffed.shape == (198,)
    np.testing.assert_allclose(restored, series)


def test_durbin_watson_near_two_for_white_noise() -> None:
    resid = np.random.default_rng(42).normal(0.0, 1.0, 500)

    assert durbin_watson(resid) == pytest.approx(2.0, abs=0.3)


def test_durbin_watson_detects_positive_autocorrelation() -> None:
    assert durbin_watson(_ar1(500, 0.9, seed=1)) < 1.0


def test_plot_acf_saves_png(tmp_path) -> None:
    out_path = tmp_path / "acf.png"
    plot_acf(np.random.default_rng(0).normal(0.0, 1.0, 200), lags=20, out_path=str(out_path))
    assert out_path.exists()
    assert out_path.stat().st_size > 0
