from __future__ import annotations

import numpy as np
import pytest

from broadway.stats.time_series import durbin_watson_test


def test_durbin_watson_near_two_for_white_noise() -> None:
    rng = np.random.default_rng(42)
    resid = rng.normal(0.0, 1.0, 500)
    dw = durbin_watson_test(resid)
    assert isinstance(dw, float)
    assert dw == pytest.approx(2.0, abs=0.3)


def test_durbin_watson_detects_positive_autocorrelation() -> None:
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, 1.0, 500)
    resid = np.zeros(500)
    resid[0] = noise[0]
    for i in range(1, 500):
        resid[i] = 0.9 * resid[i - 1] + noise[i]
    dw = durbin_watson_test(resid)
    assert dw < 1.0


def test_plot_acf_saves_png(tmp_path) -> None:
    from broadway.stats.time_series import plot_acf

    rng = np.random.default_rng(0)
    out_path = tmp_path / "acf.png"
    plot_acf(rng.normal(0.0, 1.0, 200), lags=20, out_path=str(out_path))
    assert out_path.exists()
    assert out_path.stat().st_size > 0
