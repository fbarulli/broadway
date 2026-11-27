"""One figure per series: the daily path with CV windows, and fold MAE bars."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from broadway.timeseries.backtest import BacktestResult
from broadway.timeseries.series import DailySeries


class TimeseriesFigures:
    """Render the baseline-vs-model evidence for one or more daily series."""

    def __init__(
        self,
        series: dict[str, DailySeries],
        results: dict[str, BacktestResult],
    ) -> None:
        self._series = series
        self._results = results

    def render(self, names: Sequence[str]) -> Figure:
        """A two-row panel per series: the daily path, then MAE by CV fold."""
        if not names:
            raise ValueError("timeseries figure needs at least one series")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(
            2, len(names), figsize=(6.0 * len(names), 6.4), squeeze=False
        )
        for column, name in enumerate(names):
            self._plot_series(axes[0][column], self._series[name], self._results[name])
            self._plot_folds(axes[1][column], self._series[name], self._results[name])
        figure.suptitle("Weekly-seasonal forecast vs same-weekday seasonal naive")
        figure.tight_layout()
        return figure

    @staticmethod
    def _plot_series(axis: Axes, series: DailySeries, result: BacktestResult) -> None:
        axis.plot(series.dates, series.values, linewidth=0.8, color="#1f4e79")
        for fold in result.folds:
            axis.axvspan(fold.test_start, fold.test_end, color="#d08c00", alpha=0.12)
        axis.set_title(f"{series.label}: daily series (shaded = CV test windows)")
        axis.set_ylabel(series.unit or "value")
        axis.tick_params(axis="x", labelrotation=30)

    @staticmethod
    def _plot_folds(axis: Axes, series: DailySeries, result: BacktestResult) -> None:
        indices = np.arange(len(result.folds))
        baseline = np.asarray([fold.baseline_mae for fold in result.folds])
        model = np.asarray([fold.model_mae for fold in result.folds])
        width = 0.38
        axis.bar(
            indices - width / 2, baseline, width, label="seasonal naive", color="#9ecae1"
        )
        axis.bar(indices + width / 2, model, width, label="model", color="#08519c")
        axis.set_title(f"{series.label}: MAE by fold")
        axis.set_xlabel("fold")
        axis.set_ylabel("MAE")
        axis.legend(frameon=False)
