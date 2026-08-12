from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from broadway.stats.regression import bp_jb, fit_ols, fit_robust


def _make_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 2.0 * x + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"x": x, "y": y})


def test_fit_ols_recovers_coefficients_and_r2() -> None:
    df = _make_df()
    model = fit_ols(df, "y ~ x")

    assert model.rsquared > 0.9
    assert model.params["x"] == pytest.approx(2.0, abs=0.15)


def test_fit_robust_returns_hc3_covariance() -> None:
    df = _make_df()
    model = fit_ols(df, "y ~ x")
    robust = fit_robust(model, cov_type="HC3")

    assert robust.cov_type == "HC3"
    assert robust.params[1] == pytest.approx(2.0, abs=0.15)


def test_bp_jb_returns_six_keys() -> None:
    df = _make_df()
    model = fit_ols(df, "y ~ x")

    result = bp_jb(model)

    assert set(result) == {
        "bp_stat",
        "bp_pval",
        "jb_stat",
        "jb_pval",
        "skew",
        "kurtosis",
    }
    assert result["bp_stat"] >= 0.0
    assert 0.0 <= result["bp_pval"] <= 1.0
    assert result["jb_stat"] >= 0.0
    assert 0.0 <= result["jb_pval"] <= 1.0
