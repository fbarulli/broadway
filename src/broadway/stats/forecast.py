"""ARIMA-family forecasts with their confidence interval.

The single owner for fitted ARIMA forecasts. Callers pass the order, the
seasonal order, and the miscoverage level from config and get a typed result
back — point forecast plus lower/upper bounds at that level — never a
statsmodels or statsforecast object, and never a hardcoded order or level.

Two backends share one contract:

- :class:`SarimaxForecaster` — statsmodels SARIMAX (always available).
- :class:`AutoArimaForecaster` — ``statsforecast.models.AutoARIMA`` behind an
  optional import that fails loud with the forecast-extra install hint.
"""

from __future__ import annotations

import importlib
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from statsmodels.tsa.statespace.sarimax import SARIMAX

from broadway.stats.time_series import ArrayLike, as_series

AUTO_ARIMA_INSTALL_HINT = "bash scripts/uv.sh sync --extra forecast"

# The statsmodels trend vocabulary; "c" (constant) is the usual level term.
Trend = Literal["n", "c", "t", "ct"]


class ArimaOrder(BaseModel):
    """Non-seasonal ``(p, d, q)`` order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    p: int = Field(ge=0)
    d: int = Field(ge=0)
    q: int = Field(ge=0)

    def as_tuple(self) -> tuple[int, int, int]:
        """The order as the tuple statsmodels expects."""
        return (self.p, self.d, self.q)


class SeasonalOrder(BaseModel):
    """Seasonal ``(P, D, Q, s)`` order; ``s == 0`` disables seasonality."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    p: int = Field(ge=0)
    d: int = Field(ge=0)
    q: int = Field(ge=0)
    s: int = Field(ge=0)

    def as_tuple(self) -> tuple[int, int, int, int]:
        """The seasonal order as the tuple statsmodels expects."""
        return (self.p, self.d, self.q, self.s)


class ForecastResult(BaseModel):
    """Point forecast with its lower/upper bound at one miscoverage level.

    ``alpha`` is the miscoverage (0.05 -> a 95% interval). The bounds are
    ordered by contract: a backend that returns a crossed interval raises here
    instead of shipping an invalid range.
    """

    model_config = ConfigDict(extra="forbid")

    alpha: float = Field(gt=0.0, lt=1.0)
    point: list[float]
    lower: list[float]
    upper: list[float]

    @model_validator(mode="after")
    def _ordered_bounds(self) -> ForecastResult:
        if not self.point:
            raise ValueError("forecast requires at least one forecast point")
        if not (len(self.point) == len(self.lower) == len(self.upper)):
            raise ValueError(
                "forecast point/lower/upper must have the same length, got "
                f"{len(self.point)}/{len(self.lower)}/{len(self.upper)}"
            )
        if any(
            low > point or point > high
            for low, point, high in zip(self.lower, self.point, self.upper, strict=True)
        ):
            raise ValueError("forecast bounds must satisfy lower <= point <= upper")
        return self


class SarimaxForecaster:
    """SARIMAX fitted from explicit orders, with a confidence interval."""

    def __init__(
        self,
        order: ArimaOrder,
        seasonal_order: SeasonalOrder,
        alpha: float,
        trend: Trend = "c",
    ) -> None:
        self._order = order.as_tuple()
        self._seasonal_order = seasonal_order.as_tuple()
        self._alpha = _require_alpha(alpha)
        self._trend = trend

    def forecast(self, series: ArrayLike, horizon: int) -> ForecastResult:
        """Fit on ``series`` and forecast ``horizon`` steps ahead."""
        values = as_series(series)
        _require_horizon(horizon)
        fitted = SARIMAX(
            values,
            order=self._order,
            seasonal_order=self._seasonal_order,
            trend=self._trend,
        ).fit(disp=False)
        forecast = fitted.get_forecast(steps=horizon)
        bounds = np.asarray(forecast.conf_int(alpha=self._alpha), dtype=float)
        return ForecastResult(
            alpha=self._alpha,
            point=np.asarray(forecast.predicted_mean, dtype=float).tolist(),
            lower=bounds[:, 0].tolist(),
            upper=bounds[:, 1].tolist(),
        )


class AutoArimaForecaster:
    """statsforecast AutoARIMA (optional dependency), same forecast contract."""

    def __init__(self, alpha: float, season_length: int) -> None:
        self._alpha = _require_alpha(alpha)
        if season_length < 1:
            raise ValueError(f"season_length must be >= 1, got {season_length}")
        self._season_length = season_length

    def forecast(self, series: ArrayLike, horizon: int) -> ForecastResult:
        """Fit AutoARIMA on ``series`` and forecast ``horizon`` steps ahead."""
        values = as_series(series)
        _require_horizon(horizon)
        level = 100.0 * (1.0 - self._alpha)
        model = _auto_arima_class()(season_length=self._season_length)
        prediction = model.forecast(y=values, h=horizon, level=[level])
        return ForecastResult(
            alpha=self._alpha,
            point=np.asarray(prediction["mean"], dtype=float).tolist(),
            lower=np.asarray(prediction[f"lo-{level}"], dtype=float).tolist(),
            upper=np.asarray(prediction[f"hi-{level}"], dtype=float).tolist(),
        )


def _auto_arima_class() -> Any:
    """Load ``statsforecast.models.AutoARIMA`` or fail loud with the install hint."""
    try:
        models = importlib.import_module("statsforecast.models")
    except ImportError as exc:
        raise ImportError(
            "AutoARIMA needs the statsforecast package; install the forecast extra:\n"
            f"  {AUTO_ARIMA_INSTALL_HINT}"
        ) from exc
    return models.AutoARIMA


def _require_alpha(alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha (miscoverage) must be in (0, 1), got {alpha}")
    return alpha


def _require_horizon(horizon: int) -> None:
    if horizon < 1:
        raise ValueError(f"forecast horizon must be >= 1, got {horizon}")
