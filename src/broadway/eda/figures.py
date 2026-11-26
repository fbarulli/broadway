"""The decision figures: one per metric, each showing the line the verdict is read against.

A bar chart without its threshold is decoration. The effect figure reuses the
existing ``evaluate.leaderboard.MetricLeaderboard`` (ranked bars, top three
highlighted) and overlays the config's minimum effect; the correlation figure
keeps Pearson and Spearman side by side because they disagree exactly on the
skewed columns; the outlier figure bars the IQR share against the maximum share.
Agg-only, no pyplot registry (deterministic reruns, no figure leaks).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from pandas import DataFrame

from broadway.eda.models import (
    CorrelationResult,
    FeatureTestBundle,
    OutlierResult,
)
from broadway.evaluate.leaderboard import MetricDirection, MetricLeaderboard

_CORRELATION_COLORMAP = "RdBu_r"  # signed coefficients need a diverging scale
_THRESHOLD_COLOR = "#D55E00"
_ACT_COLOR = "#D55E00"
_INVESTIGATE_COLOR = "#E69F00"
_QUIET_COLOR = "#BDBDBD"
_FIGURE_WIDTH = 8.0
_ROW_HEIGHT = 0.42
_MIN_HEIGHT = 2.8
_PANEL_HEIGHT = 4.2
_PERCENT = 100.0
_LABEL_FONTSIZE = 8
_TITLE_FONTSIZE = 10
_GRID_ALPHA = 0.3
_VALUE_DECIMALS = 2


class DecisionFigures:
    """Effect bars, the correlation heatmap and the outlier-share bars."""

    def effects(self, bundle: FeatureTestBundle) -> Figure | None:
        """Per-feature effect, ranked, top three highlighted, thresholds drawn.

        Two lines matter and they are not the same: the declared minimum effect
        and each feature's *effective* threshold ``max(min_effect, null band)``
        the verdict actually compares against. Drawing only the minimum would
        show a bar "past the line" next to an `investigate` verdict.
        """
        rows = [row for row in bundle.features if row.verdict is not None] or list(
            bundle.features
        )
        if not rows:
            return None
        frame = DataFrame(
            {"feature": [row.feature for row in rows], "effect": [row.effect for row in rows]}
        )
        leaderboard = MetricLeaderboard.from_frame(
            frame, id_column="feature", metrics={"effect": MetricDirection.MAXIMIZE}
        )
        figure = leaderboard.figures()["effect"]
        axis = figure.axes[0]
        thresholds = self._effective_thresholds(bundle, [row.feature for row in rows])
        axis.scatter(
            thresholds,
            np.arange(len(rows)),
            marker="|",
            s=180,
            linewidths=1.6,
            color=_THRESHOLD_COLOR,
            zorder=3,
            label="effective threshold (max(min_effect, null 95th))",
        )
        axis.axvline(
            bundle.min_effect,
            color=_THRESHOLD_COLOR,
            linestyle="--",
            linewidth=1.0,
            label=f"declared minimum effect {bundle.min_effect:g}",
        )
        axis.legend(loc="lower right", fontsize=_LABEL_FONTSIZE)
        axis.set_title(
            f"association effect vs threshold — {bundle.target}", fontsize=_TITLE_FONTSIZE
        )
        return figure

    @staticmethod
    def _effective_thresholds(bundle: FeatureTestBundle, features: list[str]) -> list[float]:
        """Each feature's own effective threshold; the declared minimum when undecided."""
        thresholds = {decision.metric: decision.threshold for decision in bundle.decisions}
        return [
            thresholds.get(f"association_{feature}", bundle.min_effect) for feature in features
        ]

    def correlation(self, result: CorrelationResult) -> Figure | None:
        """Pearson and Spearman coefficients side by side, values annotated."""
        if len(result.columns) < 2:
            return None
        figure = Figure(
            figsize=(_FIGURE_WIDTH * 1.4, max(_PANEL_HEIGHT, 0.42 * len(result.columns) + 2.0)),
            layout="constrained",
        )
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, 2)
        for axis, label, matrix in (
            (axes[0], "Pearson", result.pearson),
            (axes[1], "Spearman", result.spearman),
        ):
            values = np.array(
                [[np.nan if cell is None else cell for cell in row] for row in matrix],
                dtype=float,
            )
            image: AxesImage = axis.imshow(
                values, vmin=-1.0, vmax=1.0, cmap=_CORRELATION_COLORMAP
            )
            self._annotate(axis, values)
            axis.set_xticks(range(len(result.columns)))
            axis.set_yticks(range(len(result.columns)))
            axis.set_xticklabels(result.columns, rotation=45, ha="right", fontsize=7)
            axis.set_yticklabels(result.columns, fontsize=7)
            axis.set_title(f"{label} — {result.target}", fontsize=_TITLE_FONTSIZE)
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        return figure

    def outliers(self, result: OutlierResult) -> Figure | None:
        """IQR outlier share per column, coloured by verdict, maximum share drawn."""
        if not result.columns:
            return None
        ordered = sorted(result.columns, key=lambda item: (-item.iqr_share, item.column))
        figure = Figure(
            figsize=(_FIGURE_WIDTH, max(_MIN_HEIGHT, _ROW_HEIGHT * len(ordered) + 1.6)),
            layout="constrained",
        )
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        positions = np.arange(len(ordered))
        axis.barh(
            positions,
            [item.iqr_share * _PERCENT for item in ordered],
            color=[self._verdict_color(item.verdict) for item in ordered],
        )
        limit = result.max_share * _PERCENT
        axis.axvline(
            limit,
            color=_THRESHOLD_COLOR,
            linestyle="--",
            linewidth=1.0,
            label=f"maximum share {limit:.2f}%",
        )
        axis.set_yticks(positions)
        axis.set_yticklabels([item.column for item in ordered], fontsize=7)
        axis.invert_yaxis()
        for position, item in enumerate(ordered):
            axis.annotate(
                f"{item.iqr_share * _PERCENT:.2f}% (z {item.z_share * _PERCENT:.2f}%, n={item.n:,})",
                xy=(item.iqr_share * _PERCENT, position),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=_LABEL_FONTSIZE,
            )
        axis.margins(x=0.22)
        axis.set_xlabel("rows outside the IQR fence (%)")
        axis.set_title("outlier share vs maximum share", fontsize=_TITLE_FONTSIZE)
        axis.grid(axis="x", alpha=_GRID_ALPHA)
        axis.set_axisbelow(True)
        axis.legend(loc="lower right", fontsize=_LABEL_FONTSIZE)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        return figure

    @staticmethod
    def _annotate(axis, values: np.ndarray) -> None:
        size = values.shape[0]
        for row in range(size):
            for column in range(size):
                value = values[row, column]
                if np.isnan(value):
                    continue
                axis.text(
                    column,
                    row,
                    f"{value:.{_VALUE_DECIMALS}f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="#ffffff" if abs(value) >= 0.5 else "#1f2933",
                )

    @staticmethod
    def _verdict_color(verdict: str) -> str:
        if verdict == "act":
            return _ACT_COLOR
        if verdict == "investigate":
            return _INVESTIGATE_COLOR
        return _QUIET_COLOR
