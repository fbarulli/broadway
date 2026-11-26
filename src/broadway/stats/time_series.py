"""Canonical time-series numerics: ACF/PACF, CCF, Ljung-Box, ADF, differencing, DW.

This module is the single numeric owner for the series statistics the platform
and the timeseries experiments use — callers never import
``statsmodels.tsa.stattools`` directly. Series arrive as a 1-D sequence; every
function fails loudly on non-finite observations, because interpolating a gap
is a caller decision, never a silent side effect here.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.stats import chi2
from statsmodels.graphics.tsaplots import plot_acf as _plot_acf
from statsmodels.stats.stattools import durbin_watson as _durbin_watson
from statsmodels.tsa.stattools import acf as _acf
from statsmodels.tsa.stattools import adfuller as _adfuller
from statsmodels.tsa.stattools import ccf as _ccf
from statsmodels.tsa.stattools import pacf as _pacf

from broadway import viz

# One definition for every caller: partial autocorrelation uses the unbiased
# Yule-Walker estimator, which is stable on the long (thousands of points)
# hourly demand series the timeseries experiments pass in.
PACF_METHOD = "ywm"

ArrayLike = np.ndarray | Sequence[float]


class ADFResult(BaseModel):
    """Augmented Dickey-Fuller outcome; the caller owns the alpha comparison."""

    model_config = ConfigDict(extra="forbid")
    statistic: float
    p_value: float
    used_lag: int
    n_obs: int
    critical_values: dict[str, float]


class LjungBoxResult(BaseModel):
    """Ljung-Box portmanteau test of residual autocorrelation up to ``lags``."""

    model_config = ConfigDict(extra="forbid")
    statistic: float
    p_value: float
    lags: int


def as_series(series: ArrayLike) -> np.ndarray:
    """Coerce a 1-D sequence to float, rejecting empty/non-finite input."""
    values = np.asarray(series, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"series must be one-dimensional, got shape {values.shape}")
    if values.size == 0:
        raise ValueError("series is empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("series contains non-finite values (NaN/inf); handle gaps before calling")
    return values


def acf(series: ArrayLike, lags: int) -> np.ndarray:
    """Autocorrelation for lags ``0..lags`` (``lags + 1`` values, lag 0 == 1)."""
    values = as_series(series)
    _require_lags(lags, values.size)
    return np.asarray(_acf(values, nlags=lags, fft=True), dtype=float)


def pacf(series: ArrayLike, lags: int) -> np.ndarray:
    """Partial autocorrelation for lags ``0..lags`` (``lags + 1`` values)."""
    values = as_series(series)
    _require_lags(lags, values.size)
    return np.asarray(_pacf(values, nlags=lags, method=PACF_METHOD), dtype=float)


def cross_correlation(x: ArrayLike, y: ArrayLike, lags: int) -> np.ndarray:
    """Cross-correlation of ``x`` and ``y`` for lags ``0..lags`` (``lags + 1`` values).

    Value ``k`` correlates ``x[t]`` with ``y[t + k]``: a positive lag means ``y``
    leads ``x``. statsmodels' ``nlags`` is exclusive, so one is added to return
    the same ``lags + 1`` shape as :func:`acf`.
    """
    first, second = as_series(x), as_series(y)
    if first.size != second.size:
        raise ValueError(f"series must be the same length, got {first.size} and {second.size}")
    _require_lags(lags, first.size)
    return np.asarray(_ccf(first, second, adjusted=False, nlags=lags + 1), dtype=float)


def ljung_box(series: ArrayLike, lags: int) -> LjungBoxResult:
    """Ljung-Box statistic and p-value at one lag cut-off.

    Computed from this module's FFT autocorrelation rather than
    ``statsmodels.stats.diagnostic.acorr_ljungbox``: that helper calls
    ``acf(..., fft=False)``, whose direct correlation is O(n^2) and minutes
    slow on the multi-million-row residual ledgers the evaluate step reads.
    The statistic is the same Ljung-Box form
    ``n (n + 2) sum(r_k^2 / (n - k))`` with a chi-square tail.
    """
    values = as_series(series)
    _require_lags(lags, values.size)
    autocorrelation = acf(values, lags)
    n = values.size
    lag_index = np.arange(1, lags + 1)
    statistic = float(
        n * (n + 2) * np.sum(autocorrelation[1:] ** 2 / (n - lag_index))
    )
    return LjungBoxResult(
        statistic=statistic,
        p_value=float(chi2.sf(statistic, lags)),
        lags=lags,
    )


def adf(series: ArrayLike) -> ADFResult:
    """Augmented Dickey-Fuller unit-root test (constant + no trend term)."""
    statistic, p_value, used_lag, n_obs, critical, _icbest = _adfuller(as_series(series))
    return ADFResult(
        statistic=float(statistic),
        p_value=float(p_value),
        used_lag=int(used_lag),
        n_obs=int(n_obs),
        critical_values={name: float(value) for name, value in critical.items()},
    )


def difference(series: ArrayLike, order: int) -> np.ndarray:
    """Apply ``order`` first differences (order 0 returns a copy).

    The values lost at each level are returned by :func:`first_values`; keep
    both to invert with :func:`inverse_difference`.
    """
    values = as_series(series)
    _require_order(order, values.size)
    return np.diff(values, n=order) if order else values.copy()


def first_values(series: ArrayLike, order: int) -> list[float]:
    """First observation of every differencing level, in application order."""
    values = as_series(series)
    _require_order(order, values.size)
    seeds: list[float] = []
    for _ in range(order):
        seeds.append(float(values[0]))
        values = np.diff(values)
    return seeds


def inverse_difference(diffed: ArrayLike, firsts: Sequence[float], order: int) -> np.ndarray:
    """Undo :func:`difference`: integrate ``diffed`` back using ``firsts``."""
    values = as_series(diffed)
    if order < 0:
        raise ValueError(f"differencing order must be >= 0, got {order}")
    if len(firsts) != order:
        raise ValueError(f"order {order} needs {order} boundary values, got {len(firsts)}")
    for seed in reversed(firsts):
        values = np.concatenate(([seed], seed + np.cumsum(values)))
    return values


def durbin_watson(resid: ArrayLike) -> float:
    """Durbin-Watson statistic of residuals (2.0 == no serial correlation)."""
    return float(_durbin_watson(np.asarray(resid, dtype=float)))


# Legacy name kept for ``project/scripts/10_durbin_watson_time.py``; the
# canonical entry point is :func:`durbin_watson`.
durbin_watson_test = durbin_watson


def plot_acf(resid: ArrayLike, lags: int, out_path: str) -> None:
    fig = _plot_acf(np.asarray(resid, dtype=float), lags=lags)
    for ax in fig.axes:
        viz.despine(ax)
    fig.savefig(out_path)
    plt.close(fig)


def _require_lags(lags: int, n: int) -> None:
    if not 0 <= lags < n:
        raise ValueError(f"lags must be in [0, {n - 1}] for {n} observations, got {lags}")


def _require_order(order: int, n: int) -> None:
    if order < 0:
        raise ValueError(f"differencing order must be >= 0, got {order}")
    if n <= order:
        raise ValueError(f"series has {n} observations; order {order} leaves nothing to difference")
