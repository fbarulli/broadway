"""Regression and classification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, *, decimals: int = 4
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if not np.all(np.isfinite(y_true)):
        raise ValueError("y_true contains NaN or infinite values")
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("y_pred contains NaN or infinite values")
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), decimals),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), decimals),
        "r2": round(float(r2_score(y_true, y_pred)), decimals),
    }
