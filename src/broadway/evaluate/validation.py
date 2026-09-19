"""Cross-validation and residual analysis."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator, KFold, TimeSeriesSplit
from sklearn.model_selection import cross_validate as sklearn_cross_validate

from broadway.evaluate.metrics import METRIC_DECIMALS

_SCORING: dict[str, str] = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
    "mape": "neg_mean_absolute_percentage_error",
    "max_error": "max_error",
    "median_ae": "neg_median_absolute_error",
    "explained_var": "explained_variance",
}

# The only negated metric whose scorer name lacks the neg_ prefix: sklearn
# registers max_error with greater_is_better=False
# (sklearn.metrics.get_scorer("max_error")._sign < 0), so its values are
# negated despite the plain name. Everything else is derivable from _SCORING.
_NEGATED_WITHOUT_PREFIX: frozenset[str] = frozenset({"max_error"})
_NEGATED_METRICS: frozenset[str] = frozenset(
    metric for metric, scorer in _SCORING.items() if scorer.startswith("neg_")
) | _NEGATED_WITHOUT_PREFIX


def _make_cv(cv_kind: str, cv_folds: int, random_state: int) -> BaseCrossValidator:
    if cv_kind == "time_series_split":
        # TimeSeriesSplit is deterministic by construction; random_state
        # is intentionally unused here.
        return TimeSeriesSplit(n_splits=cv_folds)
    if cv_kind == "kfold":
        return KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    raise ValueError(f"unknown cv_kind: {cv_kind!r}")


def cross_validate(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    random_state: int,
    cv_kind: str,
    decimals: int = METRIC_DECIMALS,
) -> dict[str, float]:
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or infinite values")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values")
    cv = _make_cv(cv_kind, cv_folds, random_state)
    scores = sklearn_cross_validate(model, X, y, cv=cv, scoring=_SCORING)
    return _mean_metrics(scores, decimals)


def residual_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = y_pred - y_true
    return {
        "mean_residual": float(np.mean(np.abs(residuals))),
        "std_residual": float(np.std(residuals)),
        "max_abs_residual": float(np.max(np.abs(residuals))),
    }


def _mean_metrics(
    scores: dict[str, np.ndarray], decimals: int = METRIC_DECIMALS
) -> dict[str, float]:
    means: dict[str, float] = {}
    for name, fold_values in scores.items():
        if not name.startswith("test_"):
            continue
        metric = name.removeprefix("test_")
        values = -fold_values if metric in _NEGATED_METRICS else fold_values
        rounded_folds = [round(float(value), decimals) for value in values]
        means[metric] = round(float(np.mean(rounded_folds)), decimals)
    return means
