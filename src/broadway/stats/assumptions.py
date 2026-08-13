"""Levene's test (equal variance), Shapiro-Wilk + skew/kurtosis (normality)."""

from __future__ import annotations

import numpy as np
from scipy import stats

from broadway.stats.guards import validate_groups

_SHAPIRO_MAX_N = 5000


def run_levene(groups: dict[str, np.ndarray]) -> dict[str, float]:
    warnings = validate_groups(groups)
    if any("zero variance" in w for w in warnings):
        raise ValueError("Levene's test requires non-zero variance in every group")
    statistic, p_value = stats.levene(*groups.values())
    return {"statistic": float(statistic), "p_value": float(p_value)}


def check_normality(groups: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    warnings = validate_groups(groups)
    if any("zero variance" in w for w in warnings):
        raise ValueError("normality checks require non-constant groups")
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
