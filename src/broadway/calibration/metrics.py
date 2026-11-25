"""Interval coverage and width metrics, overall and per ISO week."""

from __future__ import annotations

import numpy as np
import pandas as pd

from broadway.calibration.contracts import PeriodCoverage


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of observations inside the inclusive ``[lower, upper]`` interval."""
    truth = np.asarray(y_true, dtype=float)
    if truth.size == 0:
        raise ValueError("coverage requires at least one observation")
    inside = (truth >= np.asarray(lower, dtype=float)) & (
        truth <= np.asarray(upper, dtype=float)
    )
    return float(np.mean(inside))


def mean_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean interval width ``upper - lower``."""
    widths = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
    if widths.size == 0:
        raise ValueError("mean_width requires at least one interval")
    return float(np.mean(widths))


def coverage_by_period(
    index: pd.DatetimeIndex,
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    min_rows: int = 1,
) -> list[PeriodCoverage]:
    """Coverage grouped by ISO week (``YYYY-Www``), ordered chronologically.

    Periods with fewer than ``min_rows`` observations are excluded: a partial
    terminal week must never be reported as a coverage collapse (same
    complete-bucket rule the series work uses).
    """
    if min_rows < 1:
        raise ValueError("min_rows must be at least 1")
    if len(index) == 0:
        return []
    truth = np.asarray(y_true, dtype=float)
    inside = (truth >= np.asarray(lower, dtype=float)) & (
        truth <= np.asarray(upper, dtype=float)
    )
    calendar = index.isocalendar()
    labels = (
        calendar["year"].astype(str)
        + "-W"
        + calendar["week"].astype(str).str.zfill(2)
    )
    grouped = pd.DataFrame(
        {"period": labels.to_numpy(), "covered": inside}
    ).groupby("period", sort=True)["covered"].agg(["size", "mean"])
    grouped = grouped[grouped["size"] >= min_rows]
    return [
        PeriodCoverage(period=str(period), n=int(count), coverage=float(mean))
        for period, count, mean in zip(
            grouped.index, grouped["size"], grouped["mean"], strict=True
        )
    ]
