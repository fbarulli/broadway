"""Levene's test (equal variance), Shapiro-Wilk + skew/kurtosis (normality)."""

from __future__ import annotations

import numpy as np
from scipy import stats

_SHAPIRO_MAX_N = 5000


def run_levene(groups: dict[str, np.ndarray]) -> dict[str, float]:
    statistic, p_value = stats.levene(*groups.values())
    return {"statistic": float(statistic), "p_value": float(p_value)}


def check_normality(groups: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, vals in groups.items():
        arr = np.asarray(vals, dtype=float)
        if arr.size > _SHAPIRO_MAX_N:
            rng = np.random.default_rng(0)
            arr = rng.choice(arr, size=_SHAPIRO_MAX_N, replace=False)
        result[name] = {
            "skew": float(stats.skew(arr)),
            "kurtosis": float(stats.kurtosis(arr)),
            "shapiro_p": float(stats.shapiro(arr).pvalue),
        }
    return result
