"""Choose the weekly-seasonal forecast owner for the lane (one factory).

AutoARIMA (statsforecast, optional forecast extra) is preferred when installed;
the always-available SARIMAX owner is the documented deterministic fallback
with the same weekly seasonal order. The backend name is recorded in the
decision provenance so a reader knows which model produced the numbers.
"""

from __future__ import annotations

import importlib.util
import logging

from broadway.config.schema import TimeseriesStep
from broadway.stats.forecast import (
    ArimaOrder,
    AutoArimaForecaster,
    ForecastResult,
    SarimaxForecaster,
    SeasonalOrder,
)
from broadway.stats.time_series import ArrayLike

logger = logging.getLogger(__name__)


class ForecastBackend:
    """A configured forecaster plus the name recorded in provenance."""

    def __init__(
        self,
        name: str,
        forecaster: SarimaxForecaster | AutoArimaForecaster,
    ) -> None:
        self.name = name
        self._forecaster = forecaster

    @classmethod
    def from_config(cls, step: TimeseriesStep) -> ForecastBackend:
        """Build AutoARIMA when available, else the config-declared SARIMAX."""
        if step.model == "auto_arima" and _statsforecast_available():
            return cls(
                "auto_arima",
                AutoArimaForecaster(alpha=step.alpha, season_length=step.season_length),
            )
        if step.model == "auto_arima":
            logger.warning(
                "timeseries: statsforecast is absent; using the deterministic SARIMAX "
                "fallback with seasonal order %s (install the forecast extra to use AutoARIMA)",
                step.seasonal_order,
            )
        name = "sarima_fallback" if step.model == "auto_arima" else "sarima"
        return cls(
            name,
            SarimaxForecaster(
                order=ArimaOrder(p=step.order[0], d=step.order[1], q=step.order[2]),
                seasonal_order=SeasonalOrder(
                    p=step.seasonal_order[0],
                    d=step.seasonal_order[1],
                    q=step.seasonal_order[2],
                    s=step.seasonal_order[3],
                ),
                alpha=step.alpha,
            ),
        )

    def forecast(self, series: ArrayLike, horizon: int) -> ForecastResult:
        """Fit on ``series`` and forecast ``horizon`` days."""
        return self._forecaster.forecast(series, horizon)


def _statsforecast_available() -> bool:
    """Whether the optional AutoARIMA backend can be imported."""
    return importlib.util.find_spec("statsforecast") is not None
