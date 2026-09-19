"""Extended regression metrics and binarized ROC/PR AUC."""

from __future__ import annotations

import numpy as np
import pytest

from broadway.evaluate.metrics import binary_metrics, compute_metrics


def test_compute_metrics_extended_keys_present_and_positive() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    metrics = compute_metrics(y_true, y_pred)
    for key in ("mape", "max_error", "median_ae", "explained_var"):
        assert key in metrics
        assert metrics[key] > 0.0


def test_compute_metrics_existing_keys_unchanged() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(0.15)
    assert metrics["rmse"] == pytest.approx(0.1581)
    assert metrics["r2"] == pytest.approx(0.98)


def test_binary_metrics_perfect_separation_is_one() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([0.1, 0.2, 0.9, 0.95, 0.99])
    metrics = binary_metrics(y_true, y_pred, threshold=3.0)
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0


def test_binary_metrics_threshold_respected() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8])
    low = binary_metrics(y_true, y_pred, threshold=2.0)
    high = binary_metrics(y_true, y_pred, threshold=3.5)
    assert low["roc_auc"] > high["roc_auc"]
    assert low["pr_auc"] > high["pr_auc"]


def test_compute_metrics_nan_raises() -> None:
    with pytest.raises(ValueError):
        compute_metrics(np.array([1.0, np.nan]), np.array([1.0, 2.0]))


def test_compute_metrics_inf_raises() -> None:
    with pytest.raises(ValueError):
        compute_metrics(np.array([1.0, np.inf]), np.array([1.0, 2.0]))


def test_binary_metrics_nan_raises() -> None:
    with pytest.raises(ValueError):
        binary_metrics(np.array([1.0, np.nan]), np.array([1.0, 2.0]), threshold=1.0)


def test_binary_metrics_inf_raises() -> None:
    with pytest.raises(ValueError):
        binary_metrics(np.array([1.0, 2.0]), np.array([1.0, np.inf]), threshold=1.0)
