"""Split-conformal intervals from absolute calibration residuals."""

from __future__ import annotations

import math

import numpy as np


class SplitConformal:
    """Symmetric split-conformal intervals.

    Finite-sample quantile: with ``n`` absolute calibration residuals sorted
    ascending, the ``(1 - alpha)`` empirical quantile is the value at 1-based
    rank ``ceil((n + 1) * (1 - alpha))``, clamped to ``[1, n]``. Clamping at
    ``n`` is the standard "cannot exceed the calibration maximum" fallback
    (finite samples near ``alpha -> 0``); clamping at 1 covers ``alpha -> 1``.
    Under exchangeability the resulting interval covers at least
    ``1 - alpha`` of future points.
    """

    def quantile(self, residuals: np.ndarray, alpha: float) -> float:
        """Return the finite-sample ``(1 - alpha)`` quantile of ``|residual|``."""
        sorted_abs = np.sort(np.abs(np.asarray(residuals, dtype=float)))
        n = sorted_abs.size
        if n == 0:
            raise ValueError(
                "split-conformal calibration requires at least one residual"
            )
        rank = math.ceil((n + 1) * (1.0 - alpha))
        return float(sorted_abs[min(max(rank, 1), n) - 1])

    def interval(self, pred: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
        """Return symmetric ``(pred - q, pred + q)`` intervals."""
        predictions = np.asarray(pred, dtype=float)
        return predictions - q, predictions + q
