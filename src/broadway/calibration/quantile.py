"""Quantile-regression intervals from fitted lower/mid/upper predictions.

The adapter returns the same ``(lower, upper)`` pair shape
:meth:`broadway.calibration.conformal.SplitConformal.interval` returns, so an
interval source can be swapped without changing the calibration evaluation.

Ordering is a contract, not a convenience: a crossed quantile forecast
(``lower > mid`` or ``mid > upper``) is a mis-fit and raises. Sorting it away
would report an interval the model never predicted.
"""

from __future__ import annotations

import numpy as np


class QuantileInterval:
    """Turn three fitted quantile predictions into one interval pair."""

    def interval(
        self,
        lower: np.ndarray,
        mid: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)`` after checking the three quantiles are ordered."""
        low = _as_vector(lower, "lower")
        middle = _as_vector(mid, "mid")
        high = _as_vector(upper, "upper")
        if not (low.shape == middle.shape == high.shape):
            raise ValueError(
                "quantile predictions must have the same shape, got "
                f"{low.shape}/{middle.shape}/{high.shape}"
            )
        if np.any(low > middle) or np.any(middle > high):
            raise ValueError("quantile predictions must satisfy lower <= mid <= upper")
        return low, high


def _as_vector(values: np.ndarray, name: str) -> np.ndarray:
    """Coerce to a finite 1-D float vector, rejecting empty/non-finite input."""
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} quantile predictions must be one-dimensional, got {vector.shape}")
    if vector.size == 0:
        raise ValueError("quantile interval requires at least one prediction")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} quantile predictions contain non-finite values")
    return vector
