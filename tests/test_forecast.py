"""Public behavior of the ARIMA forecast owners."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import broadway.stats.forecast as forecast_module
from broadway.stats.forecast import (
    ArimaOrder,
    AutoArimaForecaster,
    ForecastResult,
    SarimaxForecaster,
    SeasonalOrder,
)

PERIOD = 12


def _seasonal_series(size: int = 240) -> np.ndarray:
    rng = np.random.default_rng(0)
    steps = np.arange(size)
    return (
        50.0
        + 10.0 * steps / size
        + 8.0 * np.sin(2.0 * np.pi * steps / PERIOD)
        + rng.normal(0.0, 1.0, size)
    )


def test_sarimax_forecast_returns_an_ordered_conf_int() -> None:
    result = SarimaxForecaster(
        order=ArimaOrder(p=1, d=1, q=1),
        seasonal_order=SeasonalOrder(p=0, d=0, q=1, s=PERIOD),
        alpha=0.05,
    ).forecast(_seasonal_series(), horizon=6)

    assert result.alpha == 0.05
    assert len(result.point) == len(result.lower) == len(result.upper) == 6
    assert all(
        low <= point <= high
        for low, point, high in zip(result.lower, result.point, result.upper, strict=True)
    )
    assert all(high > low for low, high in zip(result.lower, result.upper, strict=True))


def test_auto_arima_guard_names_the_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_: str) -> None:
        raise ImportError("No module named 'statsforecast'")

    monkeypatch.setattr(forecast_module.importlib, "import_module", missing)
    with pytest.raises(ImportError, match="uv.sh sync --extra forecast"):
        AutoArimaForecaster(alpha=0.05, season_length=24).forecast(
            np.arange(48.0), horizon=4
        )


def test_auto_arima_maps_the_native_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    class _AutoArima:
        def __init__(self, season_length: int) -> None:
            self.season_length = season_length

        def forecast(
            self, y: np.ndarray, h: int, level: list[float]
        ) -> dict[str, np.ndarray]:
            assert level == [95.0]
            mean = np.linspace(10.0, 20.0, h)
            return {"mean": mean, "lo-95.0": mean - 2.0, "hi-95.0": mean + 2.0}

    real_import = forecast_module.importlib.import_module

    def fake_import(name: str) -> object:
        if name == "statsforecast.models":
            return SimpleNamespace(AutoARIMA=_AutoArima)
        return real_import(name)

    monkeypatch.setattr(forecast_module.importlib, "import_module", fake_import)
    result = AutoArimaForecaster(alpha=0.05, season_length=24).forecast(
        np.arange(48.0), horizon=3
    )

    assert result.point == [10.0, 15.0, 20.0]
    assert result.lower == [8.0, 13.0, 18.0]
    assert result.upper == [12.0, 17.0, 22.0]


def test_forecast_rejects_an_invalid_level_or_horizon() -> None:
    forecaster = SarimaxForecaster(
        order=ArimaOrder(p=1, d=0, q=1),
        seasonal_order=SeasonalOrder(p=0, d=0, q=0, s=0),
        alpha=0.05,
    )
    with pytest.raises(ValueError, match="alpha"):
        SarimaxForecaster(
            order=ArimaOrder(p=1, d=0, q=1),
            seasonal_order=SeasonalOrder(p=0, d=0, q=0, s=0),
            alpha=1.0,
        )
    with pytest.raises(ValueError, match="horizon"):
        forecaster.forecast(_seasonal_series(), horizon=0)
    with pytest.raises(ValueError, match="season_length"):
        AutoArimaForecaster(alpha=0.05, season_length=0)


def test_forecast_result_rejects_unordered_or_ragged_bounds() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ForecastResult(alpha=0.05, point=[], lower=[], upper=[])
    with pytest.raises(ValueError, match="same length"):
        ForecastResult(alpha=0.05, point=[1.0], lower=[], upper=[2.0])
    with pytest.raises(ValueError, match="lower <= point <= upper"):
        ForecastResult(alpha=0.05, point=[1.0], lower=[2.0], upper=[3.0])
