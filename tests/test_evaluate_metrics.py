"""Confusion-at-cutoff and threshold tuning metrics."""

from __future__ import annotations

import numpy as np
import pytest

from broadway.evaluate.metrics import confusion_at, tune_threshold


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
