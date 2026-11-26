"""Residual diagnostics over the persisted holdout ledger."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from broadway.evaluate.contracts import ResidualDiagnosticsConfig
from broadway.evaluate.validation import ResidualDiagnosticsPass
from broadway.stats.diagnostics import qq_correlation


def _ledger(path: Path, residuals: np.ndarray) -> Path:
    pd.DataFrame({"residual": residuals}).to_parquet(path, index=False)
    return path


def _ar1(rng: np.random.Generator, size: int, coefficient: float) -> np.ndarray:
    noise = rng.normal(size=size)
    values = np.empty(size)
    values[0] = noise[0]
    for index in range(1, size):
        values[index] = coefficient * values[index - 1] + noise[index]
    return values


def test_residual_diagnostics_separate_ar1_from_white_noise(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    white = rng.normal(size=2000)
    autocorrelated = _ar1(rng, 2000, 0.8)
    diagnostics = ResidualDiagnosticsPass(ResidualDiagnosticsConfig(acf_lags=3, ljung_box_lags=3))

    noisy = diagnostics.compute(_ledger(tmp_path / "white.parquet", white))
    correlated = diagnostics.compute(_ledger(tmp_path / "ar1.parquet", autocorrelated))

    assert noisy.n_obs == 2000 == correlated.n_obs
    assert set(correlated.acf_lags) == {1, 2, 3}
    assert abs(noisy.acf_lags[1]) < 0.1
    assert correlated.acf_lags[1] > 0.6
    assert noisy.ljung_box_lags == 3
    assert noisy.ljung_box_p_value > 0.05
    assert correlated.ljung_box_p_value < 1e-6
    assert qq_correlation(white) > 0.99
    with pytest.raises(ValueError, match="constant residual vector"):
        qq_correlation(np.ones(10))
    with pytest.raises(ValueError, match="at least 3"):
        qq_correlation(np.array([1.0, 2.0]))
