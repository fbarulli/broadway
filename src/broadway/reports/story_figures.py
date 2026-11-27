"""Story figures: in-memory matplotlib views of the published evidence.

The story page owns the narrative; this module owns the drawing. Every function
takes already-extracted evidence (records, funnels, effect rows) and returns a
``Figure`` — no file is written, nothing is read, and the Agg backend is pinned
for deterministic reruns. ``to_png`` serialises a figure for the HTTP route.
"""

from __future__ import annotations

from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from broadway import viz

_THRESHOLD_COLOR = "#D55E00"
_ROLE_COLORS = {
    "naive_reference": "#999999",
    "linear_reference": "#E69F00",
    "candidate": "#0072B2",
}
_DEFAULT_COLOR = "#0072B2"


def question_figure(decision_moment: str, target_label: str) -> Figure:
    """The decision boundary: features are known at decision time, the outcome after."""
    figure = Figure(figsize=(8.0, 1.9), dpi=150)
    axis = figure.add_subplot(111)
    axis.plot([0.1, 0.9], [0.0, 0.0], color="#666666", linewidth=2, zorder=1)
    axis.scatter([0.1], [0.0], s=130, color=_DEFAULT_COLOR, zorder=2)
    axis.scatter([0.9], [0.0], s=130, color=_THRESHOLD_COLOR, zorder=2)
    axis.text(
        0.1,
        0.2,
        f"decide: {decision_moment}",
        ha="left",
        fontsize=viz.TICK_FONTSIZE,
    )
    axis.text(
        0.9,
        0.2,
        f"observe: {target_label}",
        ha="right",
        fontsize=viz.TICK_FONTSIZE,
    )
    axis.text(
        0.5,
        -0.28,
        "features must be known here",
        ha="center",
        fontsize=viz.TICK_FONTSIZE,
        color="#666666",
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.45, 0.45)
    axis.set_axis_off()
    figure.suptitle("The decision boundary", fontsize=viz.TITLE_FONTSIZE)
    return figure


def join_figure(
    stages: list[tuple[str, int]],
    title: str,
    balance: list[tuple[str, int]] | None,
) -> Figure:
    """Rows through the join, with an optional target-balance panel."""
    columns = 2 if balance else 1
    figure = Figure(figsize=(8.0, 3.6), dpi=150)
    axis = figure.add_subplot(1, columns, 1)
    labels = [label for label, _ in stages]
    counts = [count for _, count in stages]
    axis.barh(labels, counts, color=viz.palette_colors(len(labels)))
    axis.set_title(title, fontsize=viz.TITLE_FONTSIZE)
    axis.set_xlabel("rows", fontsize=viz.LABEL_FONTSIZE)
    axis.tick_params(labelsize=viz.TICK_FONTSIZE)
    for index, (_, count) in enumerate(stages):
        axis.text(count, index, f" {count:,}", va="center", fontsize=viz.TICK_FONTSIZE)
    viz.despine(axis)
    if balance:
        balance_axis = figure.add_subplot(1, 2, 2)
        balance_labels = [label for label, _ in balance]
        balance_counts = [count for _, count in balance]
        balance_axis.bar(
            balance_labels, balance_counts, color=viz.palette_colors(len(balance_labels))
        )
        balance_axis.set_title("Target balance", fontsize=viz.TITLE_FONTSIZE)
        balance_axis.set_ylabel("rows", fontsize=viz.LABEL_FONTSIZE)
        balance_axis.tick_params(labelsize=viz.TICK_FONTSIZE)
        viz.despine(balance_axis)
    return figure


def eda_figure(
    threshold: float,
    unit: str | None,
    rows: list[tuple[str, float, float | None, float | None]],
) -> Figure | None:
    """Top effects with their 95% intervals, against the effect floor."""
    if not rows:
        return None
    top = rows[:10][::-1]
    figure = Figure(figsize=(8.0, max(2.8, 0.34 * len(top) + 1.5)), dpi=150)
    axis = figure.add_subplot(111)
    labels = [label for label, _, _, _ in top]
    if all(low is not None and high is not None for _, _, low, high in top):
        frame = pd.DataFrame(
            {
                "value": [value for _, value, _, _ in top],
                "low": [low for _, _, low, _ in top],
                "high": [high for _, _, _, high in top],
            },
            index=labels,
        )
        viz.draw_coef_forest(axis, frame["value"], frame["low"], frame["high"], annotate=False)
    else:
        axis.barh(labels, [value for _, value, _, _ in top], color=viz.palette_colors(len(top)))
    axis.axvline(
        threshold,
        color=_THRESHOLD_COLOR,
        linestyle="--",
        linewidth=1,
        label="effect floor",
    )
    axis.legend(fontsize=viz.TICK_FONTSIZE, frameon=False, loc="lower right")
    axis.set_title(
        "Association with the target (effect size, 95% interval)",
        fontsize=viz.TITLE_FONTSIZE,
    )
    axis.set_xlabel(unit or "effect size", fontsize=viz.LABEL_FONTSIZE)
    axis.tick_params(labelsize=viz.TICK_FONTSIZE)
    viz.despine(axis)
    return figure


def features_figure(stages: list[tuple[str, int]], title: str) -> Figure:
    """Feature-surface counts: plan columns, added columns, removed columns."""
    figure = Figure(figsize=(8.0, 2.8), dpi=150)
    axis = figure.add_subplot(111)
    labels = [label for label, _ in stages]
    counts = [count for _, count in stages]
    axis.barh(labels, counts, color=viz.palette_colors(len(labels)))
    axis.set_title(title, fontsize=viz.TITLE_FONTSIZE)
    axis.set_xlabel("columns", fontsize=viz.LABEL_FONTSIZE)
    axis.tick_params(labelsize=viz.TICK_FONTSIZE)
    for index, (_, count) in enumerate(stages):
        axis.text(count, index, f" {count:,}", va="center", fontsize=viz.TICK_FONTSIZE)
    viz.despine(axis)
    return figure


def model_figure(
    candidates: list[str],
    rmses: list[float],
    lows: list[float],
    highs: list[float],
    roles: list[str],
) -> Figure:
    """Repeated-CV RMSE per candidate with its 95% interval."""
    figure = Figure(figsize=(8.0, max(2.8, 0.5 * len(candidates) + 1.5)), dpi=150)
    axis = figure.add_subplot(111)
    for index, (rmse, low, high, role) in enumerate(zip(rmses, lows, highs, roles)):
        axis.errorbar(
            rmse,
            index,
            xerr=[[rmse - low], [high - rmse]],
            fmt="o",
            capsize=3,
            color=_ROLE_COLORS.get(role, _DEFAULT_COLOR),
        )
        axis.annotate(
            f"{rmse:.3f}",
            (rmse, index),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=viz.TICK_FONTSIZE,
        )
    axis.set_yticks(range(len(candidates)))
    axis.set_yticklabels(candidates)
    axis.invert_yaxis()
    axis.set_xlabel("RMSE (lower is better)", fontsize=viz.LABEL_FONTSIZE)
    axis.set_title("Repeated-CV RMSE by candidate (95% interval)", fontsize=viz.TITLE_FONTSIZE)
    axis.grid(axis="x", alpha=viz.GRID_ALPHA)
    viz.despine(axis)
    return figure


def to_png(figure: Figure) -> bytes:
    """Serialise one figure to PNG bytes (Agg canvas, no pyplot registry)."""
    FigureCanvasAgg(figure)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    return buffer.getvalue()
