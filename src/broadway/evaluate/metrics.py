"""Regression and classification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

METRIC_DECIMALS = 4


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), METRIC_DECIMALS),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), METRIC_DECIMALS),
        "r2": round(float(r2_score(y_true, y_pred)), METRIC_DECIMALS),
    }
