"""Cross-validation and residual analysis."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold

from broadway.evaluate.metrics import METRIC_DECIMALS, compute_metrics


def cross_validate(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    random_state: int,
) -> dict[str, float]:
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_scores: list[dict[str, float]] = []
    for train_idx, val_idx in kf.split(X):
        fold_model = clone(model)
        fold_model.fit(X[train_idx], y[train_idx])
        fold_scores.append(
            compute_metrics(y[val_idx], fold_model.predict(X[val_idx]))
        )
    return _mean_metrics(fold_scores)


def residual_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = y_pred - y_true
    return {
        "mean_residual": float(np.mean(np.abs(residuals))),
        "std_residual": float(np.std(residuals)),
        "max_abs_residual": float(np.max(np.abs(residuals))),
    }


def _mean_metrics(fold_scores: list[dict[str, float]]) -> dict[str, float]:
    return {
        metric: round(float(np.mean([scores[metric] for scores in fold_scores])), METRIC_DECIMALS)
        for metric in fold_scores[0]
    }
