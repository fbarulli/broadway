"""Regression and classification metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import get_args

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

from broadway.evaluate.contracts import ThresholdRow, TuningObjective

# Single source of the rounding precision for every metric the platform
# reports; the CV path (validation.py) imports it from here as well.
METRIC_DECIMALS: int = 4

_TUNING_OBJECTIVES: frozenset[str] = frozenset(get_args(TuningObjective))


def _validated_binary_pair(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate binary truth labels and finite scores once, for all cutoffs.

    ``y_true`` is the decision's TRUTH labels (0/1 or bool); ``y_pred`` is the
    continuous score each cutoff is applied to. Anything else is a misuse and
    fails loud rather than silently binarizing a regression target.
    """
    labels = np.asarray(y_true)
    scores = np.asarray(y_pred, dtype=float)
    if labels.shape != scores.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {labels.shape} and {scores.shape}"
        )
    if labels.ndim != 1:
        raise ValueError(f"y_true and y_pred must be one-dimensional, got {labels.ndim}D")
    if labels.dtype != bool:
        labels = labels.astype(float)
        if not np.all(np.isfinite(labels)):
            raise ValueError("y_true contains NaN or infinite values")
        if not np.all((labels == 0.0) | (labels == 1.0)):
            raise ValueError(
                "y_true must hold binary truth labels (0/1 or bool) — binarize "
                "the regression target at the declared decision threshold first"
            )
    if not np.all(np.isfinite(scores)):
        raise ValueError("y_pred contains NaN or infinite values")
    return labels, scores


def _confusion(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    decimals: int,
) -> ThresholdRow:
    """Confusion counts and rates for one cutoff, on pre-validated inputs."""
    predicted = scores >= threshold
    actual = labels.astype(bool)
    tp = int(np.count_nonzero(actual & predicted))
    fp = int(np.count_nonzero(~actual & predicted))
    tn = int(np.count_nonzero(~actual & ~predicted))
    fn = int(np.count_nonzero(actual & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / labels.size if labels.size else 0.0
    return ThresholdRow(
        threshold=threshold,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=round(precision, decimals),
        recall=round(recall, decimals),
        f1=round(f1, decimals),
        accuracy=round(accuracy, decimals),
    )


def confusion_at(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    *,
    decimals: int = METRIC_DECIMALS,
) -> dict[str, int | float]:
    """TP/FP/TN/FN counts and precision/recall/f1/accuracy at a score cutoff."""
    labels, scores = _validated_binary_pair(y_true, y_pred)
    return _confusion(labels, scores, threshold, decimals).model_dump(exclude={"threshold"})


def tune_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: Sequence[float] | np.ndarray,
    objective: TuningObjective,
    *,
    decimals: int = METRIC_DECIMALS,
) -> tuple[float, list[ThresholdRow]]:
    """Pick the cutoff maximizing ``objective``; return it plus per-cutoff rows.

    Ties keep the earliest threshold in the supplied (config-sampled) order so
    the choice is deterministic.
    """
    if objective not in _TUNING_OBJECTIVES:
        raise ValueError(
            f"unknown tuning objective {objective!r}; expected one of {sorted(_TUNING_OBJECTIVES)}"
        )
    grid = np.asarray(thresholds, dtype=float)
    if grid.size == 0:
        raise ValueError("thresholds must contain at least one candidate cutoff")
    if not np.all(np.isfinite(grid)):
        raise ValueError("thresholds contains NaN or infinite values")
    labels, scores = _validated_binary_pair(y_true, y_pred)
    rows = [_confusion(labels, scores, float(threshold), decimals) for threshold in grid]
    best_index = max(range(len(rows)), key=lambda index: getattr(rows[index], objective))
    return float(grid[best_index]), rows


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
