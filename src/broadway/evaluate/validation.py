"""Cross-validation and residual analysis."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate as sklearn_cross_validate

_SCORING: dict[str, str] = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
    "mape": "neg_mean_absolute_percentage_error",
    "max_error": "max_error",
    "median_ae": "neg_median_absolute_error",
    "explained_var": "explained_variance",
}

# Metrics whose sklearn scorers return negated values (lower is better).
# max_error is included even though its scorer is not neg_-prefixed: sklearn
# registers it with greater_is_better=False, so its values are also negated.
_NEGATED_METRICS: frozenset[str] = frozenset(
    {"mae", "rmse", "mape", "median_ae", "max_error"}
)


def cross_validate(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    random_state: int,
    decimals: int = 4,
) -> dict[str, float]:
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or infinite values")
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = sklearn_cross_validate(model, X, y, cv=cv, scoring=_SCORING)
    return _mean_metrics(scores, decimals)


def residual_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = y_pred - y_true
    return {
        "mean_residual": float(np.mean(np.abs(residuals))),
        "std_residual": float(np.std(residuals)),
        "max_abs_residual": float(np.max(np.abs(residuals))),
    }


def _mean_metrics(scores: dict[str, np.ndarray], decimals: int = 4) -> dict[str, float]:
    means: dict[str, float] = {}
    for name, fold_values in scores.items():
        if not name.startswith("test_"):
            continue
        metric = name.removeprefix("test_")
        values = -fold_values if metric in _NEGATED_METRICS else fold_values
        rounded_folds = [round(float(value), decimals) for value in values]
        means[metric] = round(float(np.mean(rounded_folds)), decimals)
    return means
