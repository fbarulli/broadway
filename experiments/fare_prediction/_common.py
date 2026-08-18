"""Shared paths, constants, and the box-with-marks plot helper for this experiment.

The sample is declared once in ``configs/sample/fare_prediction_1m.yaml``
(seed/size/columns/filters/schema). Steps only consume the name — the sample
registry owns paths, filtering, and sampling.
"""

import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; set before pyplot import

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter, LogLocator

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "experiments" / "results" / HERE.name
SAMPLE_NAME = "fare_prediction_1m"

UNIT_FMT = {
    "fare_amount": "${:g}",
    "trip_distance": "{:g} mi",
    "trip_duration_minutes": "{:.0f} min",
}

MARK_LABELS: tuple[tuple[str, float], ...] = (
    ("min", 0.0),
    ("1%", 0.01),
    ("5%", 0.05),
    ("50%", 0.50),
    ("95%", 0.95),
    ("99%", 0.99),
    ("99.9%", 0.999),
    ("max", 1.0),
)


def box_with_marks(
    ax: plt.Axes,
    values: pd.Series,
    unit_fmt: str,
    title: str,
    tail: bool = True,
    counts: bool = True,
) -> None:
    """Box-whisker on log-y with labeled percentile marks and band counts.

    Whiskers span min→max; the mean is marked; min/1/5/50/95/99/99.9%/max are
    drawn as labeled dashed lines (labels left); each band's value count is
    shown on the right; the region above 99% is shaded as the tail.
    """
    sns.boxplot(
        y=values, log_scale=True, color="#4c72b0", whis=[0, 100],
        width=0.35, ax=ax, showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "#d62728",
                   "markeredgecolor": "#d62728", "markersize": 5},
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _p, fmt=unit_fmt: fmt.format(v))
    )
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2.0, 5.0)))
    ax.yaxis.set_minor_formatter(
        FuncFormatter(lambda v, _p, fmt=unit_fmt: fmt.format(v))
    )
    ax.tick_params(axis="y", which="minor", labelsize=6)
    thresholds: list[tuple[float, str]] = []
    for label, p in MARK_LABELS:
        y = float(values.min() if p == 0.0 else
                  values.max() if p == 1.0 else values.quantile(p))
        thresholds.append((y, label))
        ax.axhline(y, color="#d62728", linestyle="--", linewidth=0.8)
        ax.text(0.03, y, f"{label} = {unit_fmt.format(y)}",
                transform=ax.get_yaxis_transform(), fontsize=7,
                color="#d62728", va="bottom")
    if tail:
        ax.axhspan(float(values.quantile(0.99)), float(values.max()),
                   alpha=0.08, color="#d62728")
    if counts:
        for (lo, _), (hi, _) in itertools.pairwise(thresholds):
            n = int(((values > lo) & (values <= hi)).sum())
            ax.text(0.97, (lo * hi) ** 0.5, f"n = {n:,}",
                    transform=ax.get_yaxis_transform(), fontsize=6,
                    color="#555555", va="center", ha="right")
    ax.set_title(title)
    ax.set_ylabel("")
    ax.grid(True, alpha=0.3, axis="y")
