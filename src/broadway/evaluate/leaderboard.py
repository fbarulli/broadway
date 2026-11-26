"""Per-metric leaderboard figures: one ranked bar chart per metric, top 3 highlighted.

The comparison is model-agnostic: the caller hands a frame with one row per
model run (an id column plus metric columns) and declares, per metric, which end
is better. Each metric becomes its own figure so a metric is never read off a
second axis; the three best runs keep a colour and a ``#1``/``#2``/``#3`` label,
the rest are muted. Rendering is Agg-only and deterministic — no pyplot
registry, no new dependency.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Okabe-Ito primaries (colourblind-safe): one colour per rank, the field muted.
_RANK_COLORS: tuple[str, ...] = ("#0072B2", "#E69F00", "#009E73")
_MUTED_COLOR: str = "#BDBDBD"

_TOP_N: int = 3
_FIGURE_WIDTH: float = 8.0
_ROW_HEIGHT: float = 0.42
_MIN_FIGURE_HEIGHT: float = 2.6
_BAR_HEIGHT: float = 0.7
_LABEL_PADDING: float = 0.08  # x-margin fraction leaving room for the value labels
_LABEL_FONTSIZE: int = 8
_TITLE_FONTSIZE: int = 10
_GRID_ALPHA: float = 0.3
_DPI: int = 160

_UNSAFE_FILENAME = re.compile(r"[^0-9A-Za-z._-]+")


class MetricDirection(StrEnum):
    """Which end of a metric is better — declared by the caller, never inferred."""

    MINIMIZE = "min"
    MAXIMIZE = "max"

    @property
    def ascending(self) -> bool:
        return self is MetricDirection.MINIMIZE

    @property
    def label(self) -> str:
        return "lower is better" if self is MetricDirection.MINIMIZE else "higher is better"


class ChampionDecision(BaseModel):
    """The single best run for one declared metric, with its interval.

    "Best" is only meaningful together with the declared direction and the
    uncertainty around it: the champion is the top-ranked run, ``value`` its
    point estimate, ``interval_*`` the measured 95% interval, and ``margin``
    the lead over the runner-up (positive = the champion is better).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str = Field(min_length=1)
    direction: MetricDirection
    champion: str = Field(min_length=1)
    value: float
    interval_low: float
    interval_high: float
    runner_up: str | None = None
    margin: float | None = None
    n_runs: int = Field(gt=0)

    @model_validator(mode="after")
    def _ordered_interval(self) -> ChampionDecision:
        if self.interval_low > self.interval_high:
            raise ValueError("interval_low must not exceed interval_high")
        if not self.interval_low <= self.value <= self.interval_high:
            raise ValueError("value must lie inside its interval")
        return self


class MetricLeaderboard:
    """One ranked bar chart per metric, the top three runs highlighted."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        id_column: str,
        metrics: Mapping[str, MetricDirection],
    ) -> None:
        if not metrics:
            raise ValueError("a leaderboard needs at least one metric")
        self._id_column = id_column
        self._metrics = {
            metric: MetricDirection(direction) for metric, direction in metrics.items()
        }
        self._rankings = {metric: self._rank(frame, metric) for metric in self._metrics}

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        id_column: str,
        metrics: Mapping[str, MetricDirection],
    ) -> MetricLeaderboard:
        """Build from a one-row-per-run frame and the declared metric directions."""
        return cls(frame, id_column=id_column, metrics=metrics)

    def figures(self) -> dict[str, Figure]:
        """One figure per declared metric, in declaration order."""
        return {metric: self._figure(metric) for metric in self._metrics}

    def write(self, results_dir: Path, stem: str) -> dict[str, Path]:
        """Save each figure as ``<stem>_<metric>.png``; returns the written paths."""
        directory = Path(results_dir)
        directory.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for metric, figure in self.figures().items():
            path = directory / f"{stem}_{self._slug(metric)}.png"
            figure.savefig(path, dpi=_DPI)
            written[metric] = path
        return written

    def top(self, metric: str, limit: int = _TOP_N) -> list[str]:
        """Ids of the best ``limit`` runs for ``metric``, best first."""
        ranked = self._rankings[metric][self._id_column].head(limit)
        return [str(run) for run in ranked]

    def champion(
        self,
        metric: str,
        intervals: Mapping[str, tuple[float, float]],
    ) -> ChampionDecision:
        """The top run for ``metric`` plus its measured 95% interval.

        ``intervals`` is the caller's measured interval per run id — the
        leaderboard ranks point estimates, so a champion without its interval
        would hide how close the field is. A missing or non-contained interval
        fails loud instead of defaulting to a zero-width one.
        """
        ranked = self._rankings[metric]
        champion = str(ranked[self._id_column].iloc[0])
        value = float(ranked[metric].iloc[0])
        if champion not in intervals:
            raise ValueError(f"no interval measured for champion candidate {champion!r}")
        low, high = intervals[champion]
        runner_up = str(ranked[self._id_column].iloc[1]) if len(ranked) > 1 else None
        margin = None
        if runner_up is not None:
            runner_value = float(ranked[metric].iloc[1])
            margin = (
                runner_value - value
                if self._metrics[metric] is MetricDirection.MINIMIZE
                else value - runner_value
            )
        return ChampionDecision(
            metric=metric,
            direction=self._metrics[metric],
            champion=champion,
            value=value,
            interval_low=min(float(low), value),
            interval_high=max(float(high), value),
            runner_up=runner_up,
            margin=margin,
            n_runs=len(ranked),
        )

    def _rank(self, frame: pd.DataFrame, metric: str) -> pd.DataFrame:
        direction = self._metrics[metric]
        ranked = frame[[self._id_column, metric]].copy()
        unmeasured = ranked[metric].isna()
        if unmeasured.any():
            names = ", ".join(sorted(ranked.loc[unmeasured, self._id_column].astype(str)))
            raise ValueError(f"{metric} is unmeasured for: {names} — cannot rank them")
        ordered = ranked.sort_values(metric, ascending=direction.ascending, kind="stable")
        return ordered.reset_index(drop=True)

    def _figure(self, metric: str) -> Figure:
        ranked = self._rankings[metric]
        names = ranked[self._id_column].astype(str).tolist()
        values = ranked[metric].to_numpy(dtype=float)
        count = len(ranked)
        figure = Figure(
            figsize=(_FIGURE_WIDTH, max(_MIN_FIGURE_HEIGHT, _ROW_HEIGHT * count + 1.4)),
            layout="constrained",
        )
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        positions = np.arange(count)
        colors = [
            _RANK_COLORS[index] if index < len(_RANK_COLORS) else _MUTED_COLOR
            for index in range(count)
        ]
        axis.barh(positions, values, height=_BAR_HEIGHT, color=colors)
        axis.set_yticks(positions)
        axis.set_yticklabels(names)
        axis.invert_yaxis()
        for index, value in enumerate(values):
            rank = index + 1
            label = self._value_label(value)
            if rank <= _TOP_N:
                label = f"#{rank} · {label}"
            axis.annotate(
                label,
                xy=(value, index),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=_LABEL_FONTSIZE,
            )
        axis.margins(x=_LABEL_PADDING)
        axis.set_title(f"{metric} — {self._metrics[metric].label}", fontsize=_TITLE_FONTSIZE)
        axis.set_xlabel(metric)
        axis.grid(axis="x", alpha=_GRID_ALPHA)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        return figure

    @staticmethod
    def _value_label(value: float) -> str:
        magnitude = abs(value)
        if magnitude >= 1000.0:
            return f"{value:,.0f}"
        if magnitude >= 100.0:
            return f"{value:,.1f}"
        if magnitude >= 1.0:
            return f"{value:,.2f}"
        return f"{value:.4f}"

    @staticmethod
    def _slug(metric: str) -> str:
        return _UNSAFE_FILENAME.sub("_", metric).strip("_") or "metric"
