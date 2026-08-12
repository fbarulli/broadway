from __future__ import annotations

import numpy as np
import statsmodels.api as sm

from broadway.stats.diagnostics import bp_test, durbin_watson, jb_test, plot_residuals


def test_bp_test_returns_two_floats_in_sane_range() -> None:
    rng = np.random.default_rng(42)
    n = 200
    exog = sm.add_constant(rng.normal(size=(n, 2)))
    resid = rng.normal(size=n)

    statistic, p_value = bp_test(resid, exog)

    assert isinstance(statistic, float)
    assert isinstance(p_value, float)
    assert statistic >= 0.0
    assert 0.0 <= p_value <= 1.0


def test_jb_test_returns_four_floats_for_normal_residuals() -> None:
    rng = np.random.default_rng(42)
    resid = rng.normal(size=1000)

    statistic, p_value, skew, kurtosis = jb_test(resid)

    assert isinstance(statistic, float)
    assert isinstance(p_value, float)
    assert isinstance(skew, float)
    assert isinstance(kurtosis, float)
    assert statistic >= 0.0
    assert 0.0 <= p_value <= 1.0
    assert abs(skew) < 1.0


def test_jb_test_rejects_skewed_residuals() -> None:
    rng = np.random.default_rng(42)
    resid = rng.exponential(size=1000)

    _, p_value_normal, _, _ = jb_test(rng.normal(size=1000))
    _, p_value_skewed, _, _ = jb_test(resid)

    assert p_value_skewed < p_value_normal


def test_durbin_watson_near_two_for_iid_residuals() -> None:
    rng = np.random.default_rng(42)
    resid = rng.normal(size=500)

    dw = durbin_watson(resid)

    assert isinstance(dw, float)
    assert 1.5 <= dw <= 2.5


def test_plot_residuals_saves_png(tmp_path) -> None:
    rng = np.random.default_rng(42)
    n = 200
    X = sm.add_constant(rng.normal(size=(n, 2)))
    y = X @ np.array([1.0, 2.0, 3.0]) + rng.normal(scale=0.5, size=n)
    model = sm.OLS(y, X).fit()

    out_path = str(tmp_path / "residuals.png")
    plot_residuals(model, out_path)

    assert tmp_path.joinpath("residuals.png").exists()
    assert tmp_path.joinpath("residuals.png").stat().st_size > 0
