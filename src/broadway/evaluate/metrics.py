"""Regression and classification metrics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import get_args

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    explained_variance_score,
    f1_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)

from broadway.evaluate.contracts import (
    BootstrapConfig,
    MetricInterval,
    ThresholdRow,
    TuningObjective,
)

# Single source of the rounding precision for every metric the platform
# reports; the CV path (validation.py) imports it from here as well.
METRIC_DECIMALS: int = 4

# MAPE unit SSOT: this platform reports MAPE as a FRACTION (0.5143 == 51.43%).
# A consumer that needs percent multiplies by this constant, never a literal.
MAPE_PERCENT_SCALE: float = 100.0

# The holdout metrics the bootstrap pass reports intervals for. The default
# vocabulary stays regression-only (see ``_DEFAULT_METRICS``); a classification
# caller names its metric explicitly.
BOOTSTRAP_METRICS: tuple[str, ...] = ("rmse", "mae", "mape")

# The binary score cutoff the scalar ``f1`` vocabulary member applies to a
# continuous score. It is the metric's own definition, not an analysis knob.
BINARY_DECISION_THRESHOLD: float = 0.5

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


def percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-row absolute percentage error as a fraction (|error| / |truth|).

    MAPE has no value where truth is zero, so that case fails loud instead of
    letting an epsilon clip turn it into a plausible-looking absurd number;
    zero-inflated targets should be scored with MAE/RMSE.
    """
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if np.any(truth == 0.0):
        raise ValueError(
            "mape is undefined when any y_true value is zero; use mae or rmse "
            "for zero-inflated targets"
        )
    return np.abs(prediction - truth) / np.abs(truth)


def require_mape_defined(y_true: np.ndarray) -> None:
    """Fail loud when a metric-vocabulary member cannot be computed."""
    if np.any(np.asarray(y_true, dtype=float) == 0.0):
        raise ValueError(
            "the metric vocabulary includes mape, which is undefined when any "
            "y_true value is zero; use a target without exact zeros or drop mape "
            "from the scored vocabulary"
        )


# One definition per metric name: the holdout scorer, the CV blocks, and the
# bootstrap all execute these functions, so a metric can never drift between
# paths (the sklearn CV subset lives in validation._SCORING and is parity-pinned).
_REGRESSION_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "mae": lambda truth, prediction: float(mean_absolute_error(truth, prediction)),
    "rmse": lambda truth, prediction: float(
        np.sqrt(mean_squared_error(truth, prediction))
    ),
    "r2": lambda truth, prediction: float(r2_score(truth, prediction)),
    "mape": lambda truth, prediction: float(np.mean(percentage_error(truth, prediction))),
    "max_error": lambda truth, prediction: float(max_error(truth, prediction)),
    "median_ae": lambda truth, prediction: float(median_absolute_error(truth, prediction)),
    "explained_var": lambda truth, prediction: float(
        explained_variance_score(truth, prediction)
    ),
}


def _pr_auc(truth: np.ndarray, score: np.ndarray) -> float:
    """Average precision under the binary truth (the imbalanced-target metric)."""
    labels, scores = _validated_binary_pair(truth, score)
    return float(average_precision_score(labels, scores))


def _f1(truth: np.ndarray, score: np.ndarray) -> float:
    """Positive-class F1 at the declared binary score cutoff."""
    labels, scores = _validated_binary_pair(truth, score)
    return float(
        f1_score(
            labels.astype(int), (scores >= BINARY_DECISION_THRESHOLD).astype(int)
        )
    )


# Classification members stay out of the default vocabulary (a regression call
# must not silently score a binary metric on continuous truth); they are named
# explicitly by the classification callers.
_CLASSIFICATION_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "pr_auc": _pr_auc,
    "f1": _f1,
}

_METRIC_FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    **_REGRESSION_METRICS,
    **_CLASSIFICATION_METRICS,
}

# The metric vocabulary as a set, for callers that validate a configured name
# before scoring (e.g. a CV decision rule naming its metric).
METRIC_NAMES: frozenset[str] = frozenset(_METRIC_FUNCTIONS)

# Which end of each metric is better — the single declaration every ranking
# reads, so a new metric cannot be ranked by a hardcoded default direction.
METRIC_DIRECTIONS: dict[str, str] = {
    "mae": "min",
    "rmse": "min",
    "r2": "max",
    "mape": "min",
    "max_error": "min",
    "median_ae": "min",
    "explained_var": "max",
    "pr_auc": "max",
    "f1": "max",
}
if set(METRIC_DIRECTIONS) != set(_METRIC_FUNCTIONS):
    raise RuntimeError(
        "METRIC_DIRECTIONS and _METRIC_FUNCTIONS disagree: "
        f"{sorted(set(METRIC_DIRECTIONS) ^ set(_METRIC_FUNCTIONS))}"
    )

# The metrics ``compute_metrics`` returns when the caller names none: the
# regression vocabulary, unchanged from the pre-classification behaviour.
_DEFAULT_METRICS: tuple[str, ...] = tuple(_REGRESSION_METRICS)



def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    decimals: int = METRIC_DECIMALS,
    metrics: Sequence[str] | None = None,
) -> dict[str, float]:
    """Score the regression vocabulary, optionally a named subset.

    ``metrics`` selects a subset (e.g. ``("rmse", "mae")`` for a per-origin or
    per-step block, where a mape gate would fire on a block that has no
    meaningful percentage error); absent, every metric is returned.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if not np.all(np.isfinite(y_true)):
        raise ValueError("y_true contains NaN or infinite values")
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("y_pred contains NaN or infinite values")
    if metrics is None:
        selected = {name: _METRIC_FUNCTIONS[name] for name in _DEFAULT_METRICS}
    else:
        unknown = sorted(set(metrics) - set(_METRIC_FUNCTIONS))
        if unknown:
            raise ValueError(
                f"unknown metric(s) {unknown}; valid metrics: "
                f"{sorted(_METRIC_FUNCTIONS)}"
            )
        selected = {name: _METRIC_FUNCTIONS[name] for name in metrics}
    return {
        name: round(float(function(y_true, y_pred)), decimals)
        for name, function in selected.items()
    }


class MetricBootstrap:
    """Percentile bootstrap confidence intervals for the holdout metrics.

    Resamples the scored rows with replacement (NumPy only — no new
    dependency), recomputes RMSE/MAE/MAPE per draw, and returns the
    ``level``-percentile interval. The point estimate is the full-holdout
    metric from :func:`compute_metrics`; the interval says how far sampling
    noise could move it.
    """

    def __init__(self, config: BootstrapConfig) -> None:
        self._config = config

    def intervals(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, MetricInterval]:
        truth = np.asarray(y_true, dtype=float)
        prediction = np.asarray(y_pred, dtype=float)
        if truth.shape != prediction.shape or truth.ndim != 1:
            raise ValueError(
                "y_true and y_pred must be one-dimensional with the same shape, "
                f"got {truth.shape} and {prediction.shape}"
            )
        if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
            raise ValueError("y_true and y_pred must be finite to bootstrap a metric")
        require_mape_defined(truth)
        generator = np.random.default_rng(self._config.seed)
        draws = np.empty((len(BOOTSTRAP_METRICS), self._config.iterations))
        for index in range(self._config.iterations):
            sample = generator.integers(0, truth.size, size=truth.size)
            for position, metric in enumerate(BOOTSTRAP_METRICS):
                draws[position, index] = _METRIC_FUNCTIONS[metric](
                    truth[sample], prediction[sample]
                )
        tail = (1.0 - self._config.level) / 2.0 * 100.0
        bounds = np.percentile(draws, [tail, 100.0 - tail], axis=1)
        return {
            metric: MetricInterval(
                lower=round(float(bounds[0, position]), METRIC_DECIMALS),
                upper=round(float(bounds[1, position]), METRIC_DECIMALS),
            )
            for position, metric in enumerate(BOOTSTRAP_METRICS)
        }

    def improvement_interval(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_reference: np.ndarray,
        metric: str = "rmse",
    ) -> MetricInterval:
        """Paired percentile bootstrap of ``reference - candidate`` error.

        Both predictions are scored on the same resampled rows, so the interval
        describes the improvement itself (positive = the candidate beats the
        reference), not two independent intervals. Used by the CV decision and
        the classification baseline.
        """
        if metric not in _METRIC_FUNCTIONS:
            raise ValueError(
                f"improvement metric must be one of {sorted(_METRIC_FUNCTIONS)}, "
                f"got {metric!r}"
            )
        truth = np.asarray(y_true, dtype=float)
        prediction = np.asarray(y_pred, dtype=float)
        reference = np.asarray(y_reference, dtype=float)
        if not (truth.shape == prediction.shape == reference.shape) or truth.ndim != 1:
            raise ValueError(
                "y_true, y_pred and y_reference must be one-dimensional with the "
                f"same shape, got {truth.shape}, {prediction.shape} and {reference.shape}"
            )
        if not (
            np.all(np.isfinite(truth))
            and np.all(np.isfinite(prediction))
            and np.all(np.isfinite(reference))
        ):
            raise ValueError("y_true, y_pred and y_reference must be finite")
        if metric == "mape":
            require_mape_defined(truth)
        function = _METRIC_FUNCTIONS[metric]
        generator = np.random.default_rng(self._config.seed)
        draws = np.empty(self._config.iterations)
        for index in range(self._config.iterations):
            sample = generator.integers(0, truth.size, size=truth.size)
            draws[index] = function(truth[sample], reference[sample]) - function(
                truth[sample], prediction[sample]
            )
        tail = (1.0 - self._config.level) / 2.0 * 100.0
        lower, upper = np.percentile(draws, [tail, 100.0 - tail])
        return MetricInterval(
            lower=round(float(lower), METRIC_DECIMALS),
            upper=round(float(upper), METRIC_DECIMALS),
        )


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
