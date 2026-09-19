"""Randomization, stratification, blocking schemes."""

from __future__ import annotations

import numpy as np


def assign_randomly(n: int, n_treatment: int, random_state: int) -> np.ndarray:
    if n_treatment < 0 or n_treatment > n:
        raise ValueError(f"n_treatment must be in [0, {n}], got {n_treatment}")
    rng = np.random.default_rng(random_state)
    flags = np.zeros(n, dtype=int)
    chosen = rng.choice(n, size=n_treatment, replace=False)
    flags[chosen] = 1
    return flags


def _allocate(counts: np.ndarray, total: int) -> list[int]:
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    raw = counts * total / n
    base = np.floor(raw).astype(int)
    alloc = base.tolist()
    leftover = total - int(base.sum())
    order = np.argsort(-(raw - base), kind="stable")
    for i in range(leftover):
        alloc[order[i]] += 1
    return alloc


def assign_stratified(strata: np.ndarray, n_treatment: int, random_state: int) -> np.ndarray:
    strata = np.asarray(strata)
    n = strata.size
    if n_treatment < 0 or n_treatment > n:
        raise ValueError(f"n_treatment must be in [0, {n}], got {n_treatment}")
    rng = np.random.default_rng(random_state)
    unique, counts = np.unique(strata, return_counts=True)
    alloc = _allocate(counts, n_treatment)
    flags = np.zeros(n, dtype=int)
    for label, k in zip(unique, alloc):
        idx = np.flatnonzero(strata == label)
        chosen = rng.choice(idx, size=k, replace=False)
        flags[chosen] = 1
    return flags
