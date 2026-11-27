"""Chronological backtest: same-weekday seasonal naive first, then the model.

Folds come from the canonical forward-chaining splitter
(:class:`broadway.evaluate.validation.TimeSeriesCvSplitter`) so a test block
never trains on a later day and a fold boundary never splits a week. Both
forecasters are scored on the identical rows; the pooled per-day absolute
errors are what the decision interval and null are computed from.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from broadway.config.schema import TimeseriesCv
from broadway.evaluate.validation import TimeAxis, TimeSeriesCvSplitter
from broadway.stats.forecast import ForecastResult
from broadway.stats.time_series import ArrayLike, as_series
from broadway.timeseries.series import DailySeries

logger = logging.getLogger(__name__)


class PointForecaster(Protocol):
    """One multi-step point forecast from a fitted model."""

    def forecast(self, series: ArrayLike, horizon: int) -> ForecastResult:
        """Fit on ``series`` and return at least ``horizon`` point forecasts."""
        ...


class SeasonalNaive:
    """Same weekday last week: repeat the last ``period`` observations.

    With daily data and ``period=7`` this is the standard seasonal-naive
    benchmark: every forecast day copies the value one week earlier, recursively
    over the horizon.
    """

    def __init__(self, period: int = 7) -> None:
        if period < 1:
            raise ValueError(f"seasonal-naive period must be >= 1, got {period}")
        self._period = period

    def forecast(self, history: ArrayLike, horizon: int) -> np.ndarray:
        """Forecast ``horizon`` days by repeating the last full week."""
        values = as_series(history)
        if values.size < self._period:
            raise ValueError(
                f"seasonal-naive needs at least {self._period} observations, got {values.size}"
            )
        if horizon < 1:
            raise ValueError(f"forecast horizon must be >= 1, got {horizon}")
        return np.resize(values[-self._period :], horizon)


@dataclass(frozen=True)
class FoldResult:
    """One forward-chaining fold's scores on its test block."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_size: int
    test_size: int
    baseline_mae: float
    model_mae: float
    baseline_rmse: float
    model_rmse: float
    relative_improvement: float


@dataclass(frozen=True)
class BacktestResult:
    """Pooled out-of-sample errors plus the per-fold scores behind them."""

    folds: tuple[FoldResult, ...]
    dates: pd.DatetimeIndex
    horizon_positions: np.ndarray
    fold_ids: np.ndarray
    baseline_errors: np.ndarray
    model_errors: np.ndarray

    @property
    def n(self) -> int:
        """Out-of-sample days scored."""
        return int(self.baseline_errors.size)

    @property
    def baseline_mae(self) -> float:
        return float(np.mean(self.baseline_errors))

    @property
    def model_mae(self) -> float:
        return float(np.mean(self.model_errors))

    @property
    def baseline_rmse(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.baseline_errors))))

    @property
    def model_rmse(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.model_errors))))

    @property
    def pooled_relative_improvement(self) -> float:
        """Relative improvement over all out-of-sample days pooled together."""
        return relative_improvement(self.baseline_mae, self.model_mae)

    @property
    def fold_improvements(self) -> np.ndarray:
        """Each fold's relative improvement (the CV unit of evidence)."""
        return np.asarray(
            [fold.relative_improvement for fold in self.folds], dtype=float
        )

    @property
    def mean_fold_improvement(self) -> float:
        """Mean relative improvement across folds (the headline metric)."""
        return float(np.mean(self.fold_improvements))

    @property
    def fold_improvement_spread(self) -> float:
        """Standard deviation of the fold improvements (the CV spread)."""
        values = self.fold_improvements
        return float(np.std(values, ddof=1)) if values.size > 1 else 0.0

    def fold_improvement(self, mask: np.ndarray | None = None) -> float:
        """Mean within-fold relative improvement, optionally over a day slice."""
        values: list[float] = []
        for fold in np.unique(self.fold_ids):
            selected = self.fold_ids == fold
            if mask is not None:
                selected &= mask
            if not bool(np.any(selected)):
                continue
            values.append(
                relative_improvement(
                    float(np.mean(self.baseline_errors[selected])),
                    float(np.mean(self.model_errors[selected])),
                )
            )
        return float(np.mean(values)) if values else 0.0


def relative_improvement(baseline_error: float, model_error: float) -> float:
    """Relative MAE reduction; a perfect baseline cannot be improved, so it is 0."""
    if baseline_error <= 0.0:
        return 0.0
    return float((baseline_error - model_error) / baseline_error)


class ForecastBacktest:
    """Score the seasonal-naive baseline and the configured model on one series."""

    def __init__(
        self,
        baseline: SeasonalNaive,
        model: PointForecaster,
        cv: TimeseriesCv,
    ) -> None:
        self._baseline = baseline
        self._model = model
        self._cv = cv

    def run(self, series: DailySeries) -> BacktestResult:
        """Run every chronological fold and pool the per-day absolute errors."""
        axis = TimeAxis.build(series.dates)
        splitter = TimeSeriesCvSplitter(
            n_splits=self._cv.folds,
            kind=self._cv.kind,
            gap=self._cv.gap,
            seasonal_period=self._cv.seasonal_period,
            train_size=self._cv.train_size,
            axis=axis,
        )
        folds: list[FoldResult] = []
        baseline_parts: list[np.ndarray] = []
        model_parts: list[np.ndarray] = []
        dates: list[pd.Timestamp] = []
        positions: list[int] = []
        fold_ids: list[int] = []
        for index, (train_idx, test_idx) in enumerate(
            splitter.split(np.arange(series.n)), start=1
        ):
            started = time.perf_counter()
            train = series.values[train_idx]
            actual = series.values[test_idx]
            baseline_prediction = self._baseline.forecast(train, len(test_idx))
            model_prediction = np.asarray(
                self._model.forecast(train, len(test_idx)).point, dtype=float
            )
            if model_prediction.shape != actual.shape:
                raise ValueError(
                    f"fold {index}: model returned {model_prediction.shape} forecasts "
                    f"for a {actual.shape} test block"
                )
            baseline_error = np.abs(actual - baseline_prediction)
            model_error = np.abs(actual - model_prediction)
            baseline_parts.append(baseline_error)
            model_parts.append(model_error)
            dates.extend(series.dates[test_idx].tolist())
            positions.extend(range(len(test_idx)))
            fold_ids.extend([index] * len(test_idx))
            folds.append(
                FoldResult(
                    index=index,
                    train_start=series.dates[train_idx[0]],
                    train_end=series.dates[train_idx[-1]],
                    test_start=series.dates[test_idx[0]],
                    test_end=series.dates[test_idx[-1]],
                    train_size=int(train_idx.size),
                    test_size=int(test_idx.size),
                    baseline_mae=float(np.mean(baseline_error)),
                    model_mae=float(np.mean(model_error)),
                    baseline_rmse=float(np.sqrt(np.mean(np.square(baseline_error)))),
                    model_rmse=float(np.sqrt(np.mean(np.square(model_error)))),
                    relative_improvement=relative_improvement(
                        float(np.mean(baseline_error)), float(np.mean(model_error))
                    ),
                )
            )
            logger.info(
                "timeseries: %s fold %d/%d train=%d test=%d baseline_mae=%.4f "
                "model_mae=%.4f elapsed=%.2fs",
                series.name,
                index,
                self._cv.folds,
                train_idx.size,
                test_idx.size,
                folds[-1].baseline_mae,
                folds[-1].model_mae,
                time.perf_counter() - started,
            )
        return BacktestResult(
            folds=tuple(folds),
            dates=pd.DatetimeIndex(dates),
            horizon_positions=np.asarray(positions, dtype=int),
            fold_ids=np.asarray(fold_ids, dtype=int),
            baseline_errors=np.concatenate(baseline_parts),
            model_errors=np.concatenate(model_parts),
        )
