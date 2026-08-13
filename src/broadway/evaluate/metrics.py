"""Regression and classification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

METRIC_DECIMALS = 4


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if not np.all(np.isfinite(y_true)):
        raise ValueError("y_true contains NaN or infinite values")
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("y_pred contains NaN or infinite values")
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), METRIC_DECIMALS),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), METRIC_DECIMALS),
        "r2": round(float(r2_score(y_true, y_pred)), METRIC_DECIMALS),
    }
