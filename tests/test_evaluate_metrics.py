"""Confusion-at-cutoff and threshold tuning metrics."""

from __future__ import annotations

import numpy as np
import pytest

from broadway.evaluate.metrics import compute_metrics, confusion_at, tune_threshold


def test_confusion_at_counts_and_rates() -> None:
    y_true = np.array([1, 1, 1, 1, 0, 0])
    y_pred = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3])
    result = confusion_at(y_true, y_pred, threshold=0.5)
    assert result == {
        "tp": 2,
        "fp": 1,
        "tn": 1,
        "fn": 2,
        "precision": pytest.approx(0.6667),
        "recall": pytest.approx(0.5),
        "f1": pytest.approx(0.5714),
        "accuracy": pytest.approx(0.5),
    }


def test_tune_threshold_picks_configured_objective_optimum() -> None:
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.1])
    thresholds = [0.2, 0.5, 0.85]

    chosen_f1, rows_f1 = tune_threshold(y_true, y_pred, thresholds, "f1")
    assert chosen_f1 == pytest.approx(0.5)
    assert [row.threshold for row in rows_f1] == pytest.approx(thresholds)
    assert rows_f1[1].f1 == pytest.approx(0.8571)

    chosen_precision, _ = tune_threshold(y_true, y_pred, thresholds, "precision")
    assert chosen_precision == pytest.approx(0.85)


def test_mape_is_a_fraction_and_fails_loud_on_zero_truth() -> None:
    metrics = compute_metrics(np.array([10.0, 20.0]), np.array([11.0, 18.0]))
    assert metrics["mape"] == pytest.approx(0.1)  # 10% reported as a fraction
    with pytest.raises(ValueError, match="zero"):
        compute_metrics(np.array([0.0, 20.0]), np.array([1.0, 18.0]))


def test_metric_subset_selects_only_requested_metrics() -> None:
    truth = np.array([10.0, 20.0])
    prediction = np.array([11.0, 18.0])
    subset = compute_metrics(truth, prediction, metrics=("rmse", "mae"))
    assert set(subset) == {"rmse", "mae"}
    with pytest.raises(ValueError, match="unknown metric"):
        compute_metrics(truth, prediction, metrics=("bogus",))
