"""Seasonal-trend decomposition: STL trend + seasonal + residual.

statsmodels owns the algorithm (``STL``). The multiplicative form is the same
fit on ``log(series)`` raised back — STL has no multiplicative mode, and the
classical ``seasonal_decompose`` alternative leaves edge artefacts (its trend is
only a centred moving average) that corrupt the strength scores. Callers pass the
seasonality period and the model from config and get arrays plus Hyndman-style
strength scores back, never a statsmodels object.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import STL

from broadway.stats.time_series import ArrayLike, as_series

MIN_PERIOD = 2


class DecomposeModel(str, Enum):
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"


class DecomposeResult(BaseModel):
    """Decomposition arrays (same length as the input) and variance strengths."""

    model_config = ConfigDict(extra="forbid")
    period: int
    model: DecomposeModel
    trend: list[float]
    seasonal: list[float]
    resid: list[float]
    trend_strength: float
    seasonal_strength: float


class TrendResult(BaseModel):
    """LOESS-smoothed trend at the observed positions (same length as input)."""

    model_config = ConfigDict(extra="forbid")
    frac: float
    trend: list[float]


def decompose(
    series: ArrayLike, period: int, model: DecomposeModel | str
) -> DecomposeResult:
    """Decompose ``series`` at ``period``, additive or multiplicative.

    ``trend_strength``/``seasonal_strength`` are ``1 - var(resid)/var(resid +
    component)`` for additive and ``1 - var(resid)/var(resid * component)`` for
    multiplicative, floored at 0 (strength of 0 means the component explains no
    variance). ``model`` accepts the config string and is parsed here, so a
    caller never has to cast.
    """
    model = DecomposeModel(model)
    values = as_series(series)
    if period < MIN_PERIOD:
        raise ValueError(f"period must be >= {MIN_PERIOD}, got {period}")
    if values.size < 2 * period:
        raise ValueError(
            f"decomposition needs at least two full periods ({2 * period} observations), "
            f"got {values.size}"
        )
    trend, seasonal, resid = _components(values, period, model)
    return DecomposeResult(
        period=period,
        model=model,
        trend=trend.tolist(),
        seasonal=seasonal.tolist(),
        resid=resid.tolist(),
        trend_strength=_strength(resid, trend, model),
        seasonal_strength=_strength(resid, seasonal, model),
    )


def lowess_trend(series: ArrayLike, frac: float) -> TrendResult:
    """Non-linear trend via local regression (``frac`` is the window fraction)."""
    values = as_series(series)
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"lowess frac must be in (0, 1], got {frac}")
    trend = lowess(values, np.arange(values.size, dtype=float), frac=frac, return_sorted=False)
    return TrendResult(frac=frac, trend=[float(value) for value in trend])


def _components(
    values: np.ndarray, period: int, model: DecomposeModel
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit STL for ``model`` and return trend/seasonal/resid on the data scale."""
    if model is DecomposeModel.MULTIPLICATIVE:
        if np.any(values <= 0.0):
            raise ValueError("multiplicative decomposition needs strictly positive observations")
        fitted = STL(np.log(values), period=period).fit()
        return np.exp(fitted.trend), np.exp(fitted.seasonal), np.exp(fitted.resid)
    fitted = STL(values, period=period).fit()
    return fitted.trend, fitted.seasonal, fitted.resid


def _strength(resid: np.ndarray, component: np.ndarray, model: DecomposeModel) -> float:
    combined = resid * component if model is DecomposeModel.MULTIPLICATIVE else resid + component
    combined_variance = float(np.var(combined))
    if combined_variance == 0.0:
        return 0.0
    return max(0.0, 1.0 - float(np.var(resid)) / combined_variance)
