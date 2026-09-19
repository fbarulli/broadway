"""Regression and classification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)

# Single source of the rounding precision for every metric the platform
# reports; the CV path (validation.py) imports it from here as well.
METRIC_DECIMALS: int = 4


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, *, decimals: int = METRIC_DECIMALS
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
        "mape": round(float(mean_absolute_percentage_error(y_true, y_pred)), decimals),
        "max_error": round(float(max_error(y_true, y_pred)), decimals),
        "median_ae": round(float(median_absolute_error(y_true, y_pred)), decimals),
        "explained_var": round(
            float(explained_variance_score(y_true, y_pred)), decimals
        ),
    }


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    *,
    decimals: int = METRIC_DECIMALS,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if not np.all(np.isfinite(y_true)):
        raise ValueError("y_true contains NaN or infinite values")
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("y_pred contains NaN or infinite values")
    y_bin = (y_true >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_bin, y_pred)), decimals),
        "pr_auc": round(float(average_precision_score(y_bin, y_pred)), decimals),
    }
