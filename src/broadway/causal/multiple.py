"""Multiple testing correction — Bonferroni, Benjamini-Hochberg, FWER."""

from __future__ import annotations

from statsmodels.stats.multitest import multipletests

_SUPPORTED_METHODS = {"bonferroni", "fdr_bh"}


def correct_pvalues(pvalues: list[float], method: str) -> list[float]:
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"unsupported method '{method}'. valid: {sorted(_SUPPORTED_METHODS)}"
        )
    _, corrected, _, _ = multipletests(pvalues, method=method)
    return [float(p) for p in corrected]
