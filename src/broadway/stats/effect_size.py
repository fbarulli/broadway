"""Effect-size computations (numpy only)."""

from __future__ import annotations

import numpy as np


def eta_squared(f_stat: float, df1: int, df2: int) -> float:
    denominator = df1 * f_stat + df2
    if denominator == 0:
        return 0.0
    return (df1 * f_stat) / denominator


def omega_squared(f_stat: float, df1: int, df2: int, n_total: int) -> float:
    numerator = df1 * (f_stat - 1)
    denominator = numerator + n_total
    if denominator == 0:
        return 0.0
    return numerator / denominator


def epsilon_squared(h_stat: float, k: int, n: int) -> float:
    if n <= k:
        return 0.0
    value = (h_stat - k + 1) / (n - k)
    return float(max(0.0, min(1.0, value)))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return 0.0
    n_a = a.size
    n_b = b.size
    pooled_var = ((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / (n_a + n_b - 2)
    if pooled_var == 0:
        return 0.0
    return float((a.mean() - b.mean()) / np.sqrt(pooled_var))


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    d = cohens_d(a, b)
    n_total = np.asarray(a).size + np.asarray(b).size
    denominator = 4 * n_total - 9
    if denominator <= 0:
        return d
    return d * (1 - 3 / denominator)


def group_imbalance(group_sizes: dict[str, int]) -> float:
    if len(group_sizes) < 2:
        return 1.0
    sizes = list(group_sizes.values())
    smallest = min(sizes)
    if smallest == 0:
        return 1.0
    return max(sizes) / smallest
