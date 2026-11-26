"""Multiple-testing correction over one declared family of p-values.

An EDA pass tests every feature against the target, so a raw p-value at
``alpha=0.05`` is not evidence: with 20 features, one false positive is the
expectation. The family is corrected once, with Benjamini-Hochberg q-values
(false-discovery rate), so the rank order survives but the threshold does not
drift with the number of columns.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np


class MultipleTestingCorrection:
    """Benjamini-Hochberg q-values for one family of p-values (or raw p).

    ``bh``: ``q_(i) = min_{j >= i} p_(j) * m / j`` with the p-values sorted
    ascending — monotone in the p-value order and clipped to ``[0, 1]``.
    ``none``: the identity, for a caller that deliberately declares each test
    its own family (the config names which, never this class).
    """

    def __init__(self, method: Literal["bh", "none"]) -> None:
        self._method = method

    def adjust(self, p_values: Sequence[float]) -> list[float]:
        """Return one q-value per input p-value, in the input order."""
        p = np.asarray(p_values, dtype=float)
        if p.size == 0:
            return []
        if self._method == "none":
            return [float(value) for value in p]
        family = p.size
        order = np.argsort(p, kind="stable")
        ranked = p[order] * family / np.arange(1, family + 1)
        monotone = np.minimum.accumulate(ranked[::-1])[::-1]
        q = np.empty(family, dtype=float)
        q[order] = np.clip(monotone, 0.0, 1.0)
        return [float(value) for value in q]
