"""Plotly-free figures: matplotlib rendered to embedded base64 images.

The repository already ships matplotlib/seaborn (no plotly), so the report
stays a single self-contained HTML file with no new dependency and no
external asset requests.
"""

from __future__ import annotations

import base64
import math
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from broadway import viz
from broadway.config.viz import VizConfig, load_viz_config
from broadway.eda.models import (
    EmbeddedFigure,
    MissingResult,
    QualityResult,
    SummaryResult,
)

_PLOT_COLUMNS = 2
_BAR_WIDTH = 0.4
_PERCENT = 100.0


class FigureRenderer:
    """Builds the report's figures and encodes each one as a data URI payload."""

    def __init__(self, plot_format: str, max_categories: int, max_plots: int) -> None:
        self._format = plot_format
        self._max_categories = max_categories
        self._max_plots = max_plots
        self._viz: VizConfig = load_viz_config()

    def render(
        self,
        frame: pd.DataFrame,
        summary: SummaryResult,
        missing: MissingResult,
        quality: QualityResult,
    ) -> list[EmbeddedFigure]:
        figures: list[EmbeddedFigure] = []
        candidates = (
            ("Missing value rates", self._missing_figure(missing)),
            ("Numeric distributions", self._numeric_figure(frame, summary)),
            ("Correlation heatmap", self._correlation_figure(frame, summary)),
            ("Categorical top values", self._categorical_figure(summary)),
            ("Outlier rates", self._outlier_figure(quality)),
        )
        for title, figure in candidates:
            if figure is not None:
                figures.append(self._embed(figure, title))
        return figures

    def _missing_figure(self, missing: MissingResult) -> Figure | None:
        if not missing.columns:
            return None
        columns = missing.columns[: self._max_plots]
        rates = [column.missing_rate * _PERCENT for column in columns]
        figure, ax = plt.subplots(figsize=(8, max(2.5, 0.4 * len(columns) + 1.5)))
        ax.barh(
            [column.column for column in columns],
            rates,
            color=viz.palette_colors(len(columns)),
        )
        ax.invert_yaxis()
        ax.set_xlabel("missing %")
        ax.set_title("Missing value rate by column")
        viz.despine(ax)
        return figure

    def _numeric_figure(
        self, frame: pd.DataFrame, summary: SummaryResult
    ) -> Figure | None:
        names = [item.name for item in summary.detail if item.kind == "numeric"][
            : self._max_plots
        ]
        if not names:
            return None
        figure = self._grid(len(names))
        axes = figure.axes
        for ax, name in zip(axes, names):
            values = frame[name].dropna()
            ax.hist(values, bins="auto", color=viz.palette_colors(1)[0])
            ax.set_title(name)
            ax.set_ylabel("rows")
            viz.despine(ax)
        self._hide_unused(axes, len(names))
        figure.suptitle("Numeric distributions")
        return figure

    def _correlation_figure(
        self, frame: pd.DataFrame, summary: SummaryResult
    ) -> Figure | None:
        names = [item.name for item in summary.detail if item.kind == "numeric"][
            : self._max_plots
        ]
        if len(names) < 2:
            return None
        correlation = frame[names].corr(numeric_only=True)
        figure, ax = plt.subplots(figsize=(7.5, max(4.0, 0.45 * len(names) + 2.0)))
        image = ax.imshow(
            correlation.to_numpy(),
            vmin=-1.0,
            vmax=1.0,
            cmap=self._viz.diagnostics.colormap,
        )
        figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)
        ax.set_title("Numeric correlation")
        return figure

    def _categorical_figure(self, summary: SummaryResult) -> Figure | None:
        items = [
            item
            for item in summary.detail
            if item.kind == "categorical"
            and item.categorical is not None
            and 1 < item.categorical.unique <= self._max_categories
        ][: self._max_plots]
        if not items:
            return None
        figure = self._grid(len(items))
        axes = figure.axes
        for ax, item in zip(axes, items):
            assert item.categorical is not None
            labels = [value.value for value in item.categorical.top_values]
            counts = [value.count for value in item.categorical.top_values]
            ax.barh(labels, counts, color=viz.palette_colors(len(labels)))
            ax.invert_yaxis()
            ax.set_title(item.name)
            ax.set_xlabel("rows")
            viz.despine(ax)
        self._hide_unused(axes, len(items))
        figure.suptitle("Categorical top values")
        return figure

    def _outlier_figure(self, quality: QualityResult) -> Figure | None:
        items = quality.outliers[: self._max_plots]
        if not items:
            return None
        labels = [item.column for item in items]
        positions = np.arange(len(items))
        figure, ax = plt.subplots(figsize=(8, max(3.0, 0.45 * len(items) + 1.5)))
        ax.bar(
            positions - _BAR_WIDTH / 2,
            [item.iqr_rate * _PERCENT for item in items],
            width=_BAR_WIDTH,
            label="IQR",
        )
        ax.bar(
            positions + _BAR_WIDTH / 2,
            [item.z_rate * _PERCENT for item in items],
            width=_BAR_WIDTH,
            label="z-score",
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("outlier %")
        ax.set_title("Outlier rate by numeric column")
        ax.legend()
        viz.despine(ax)
        return figure

    def _grid(self, count: int) -> Figure:
        rows = math.ceil(count / _PLOT_COLUMNS)
        size = self._viz.fig_size_per_subplot
        figure, _ = plt.subplots(
            rows,
            _PLOT_COLUMNS,
            figsize=(8.0, max(3.0, rows * size)),
            layout="constrained",
        )
        return figure

    @staticmethod
    def _hide_unused(axes: list[Axes], used: int) -> None:
        for ax in axes[used:]:
            ax.set_visible(False)

    def _embed(self, figure: Figure, title: str) -> EmbeddedFigure:
        buffer = BytesIO()
        figure.savefig(
            buffer, format=self._format, dpi=self._viz.dpi, bbox_inches="tight"
        )
        plt.close(figure)
        mime = "image/png" if self._format == "png" else "image/svg+xml"
        return EmbeddedFigure(
            title=title,
            mime_type=mime,
            data_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
        )
