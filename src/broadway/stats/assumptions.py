"""Levene's test (equal variance), Shapiro-Wilk + skew/kurtosis (normality)."""

from __future__ import annotations

import numpy as np
from scipy import stats

from broadway.stats.guards import validate_groups


def run_levene(groups: dict[str, np.ndarray]) -> dict[str, float]:
    warnings = validate_groups(groups)
    if any("zero variance" in w for w in warnings):
        raise ValueError("Levene's test requires non-zero variance in every group")
    statistic, p_value = stats.levene(*groups.values())
    return {"statistic": float(statistic), "p_value": float(p_value)}


def check_normality(
    groups: dict[str, np.ndarray],
    shapiro_seed: int,
    shapiro_max_n: int = 5000,
) -> dict[str, dict[str, float]]:
    """Per-group skew/kurtosis/Shapiro; groups above ``shapiro_max_n`` are subsampled.

    The subsample seed is a required argument carried from config
    (``configs/step/walkthrough.yaml: shapiro_seed``) — no code-side default,
    so every call site names its provenance (YAML single-source-of-truth law).
    """
    warnings = validate_groups(groups)
    if any("zero variance" in w for w in warnings):
        raise ValueError("normality checks require non-constant groups")
    result: dict[str, dict[str, float]] = {}
    for name, vals in groups.items():
        arr = np.asarray(vals, dtype=float)
        if arr.size > shapiro_max_n:
            rng = np.random.default_rng(shapiro_seed)
            arr = rng.choice(arr, size=shapiro_max_n, replace=False)
        result[name] = {
            "skew": float(stats.skew(arr)),
            "kurtosis": float(stats.kurtosis(arr)),
            "shapiro_p": float(stats.shapiro(arr).pvalue),
        }
    return result
