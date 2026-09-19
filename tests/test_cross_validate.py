"""cross_validate tests — sklearn.model_selection.cross_validate swap.

The golden dict below was captured from the pre-swap hand-written KFold
loop on the fixture (sklearn 1.7.2). Hardcoded goldens are intentional
brittleness: a sklearn-version shift must fail loud.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate as sklearn_cross_validate
from sklearn.pipeline import Pipeline

from broadway.evaluate.metrics import compute_metrics
from broadway.evaluate.validation import _SCORING, _make_cv, _mean_metrics, cross_validate


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    return make_regression(n_samples=300, n_features=5, noise=10.0, random_state=42)


def test_cross_validate_matches_hand_loop_golden() -> None:
    X, y = _fixture()
    result = cross_validate(LinearRegression(), X, y, cv_folds=5, random_state=42, cv_kind="kfold")
    assert result == {
        "mae": 7.7527,
        "rmse": 9.8213,
        "r2": 0.9912,
        "mape": 0.2867,
        "max_error": 25.1997,
        "median_ae": 6.3965,
        "explained_var": 0.9913,
    }


def test_mean_metrics_sign_and_rounding() -> None:
    scores = {
        "test_mae": np.array([-1.111, -2.222]),
        "test_rmse": np.array([-3.0, -4.0]),
        "test_r2": np.array([0.5, 0.7]),
    }
    assert _mean_metrics(scores, decimals=2) == {"mae": 1.67, "rmse": 3.5, "r2": 0.6}


def test_metric_vocabulary_parity_holdout_vs_cv() -> None:
    # evaluate/module.py consumes both paths — compute_metrics on the holdout
    # split and _SCORING through cross_validate — so the vocabularies are
    # coupled by design; drift here silently diverges holdout vs CV metrics.
    X, y = make_regression(n_samples=40, n_features=3, noise=1.0, random_state=7)
    y_pred = LinearRegression().fit(X, y).predict(X)
    assert set(compute_metrics(y, y_pred)) == {name for name in _SCORING}


def test_cross_validate_reproducible() -> None:
    X, y = _fixture()
    first = cross_validate(LinearRegression(), X, y, cv_folds=5, random_state=42, cv_kind="kfold")
    second = cross_validate(LinearRegression(), X, y, cv_folds=5, random_state=42, cv_kind="kfold")
    assert first == second


def test_cross_validate_different_seed_differs() -> None:
    X, y = _fixture()
    seed_1 = cross_validate(LinearRegression(), X, y, cv_folds=5, random_state=1, cv_kind="kfold")
    seed_2 = cross_validate(LinearRegression(), X, y, cv_folds=5, random_state=2, cv_kind="kfold")
    assert seed_1 != seed_2


def test_cross_validate_insufficient_samples_fails_loud() -> None:
    X, y = make_regression(n_samples=3, n_features=5, noise=10.0, random_state=42)
    with pytest.raises(ValueError):
        cross_validate(LinearRegression(), X, y, cv_folds=5, random_state=42, cv_kind="kfold")


def test_cross_validate_nan_in_x_raises_instead_of_persisting_nan_metrics() -> None:
    X, y = _fixture()
    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="X contains NaN or infinite values"):
        cross_validate(LinearRegression(), X_bad, y, cv_folds=5, random_state=42, cv_kind="kfold")


def test_cross_validate_inf_in_x_raises_instead_of_persisting_nan_metrics() -> None:
    X, y = _fixture()
    X_bad = X.copy()
    X_bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="X contains NaN or infinite values"):
        cross_validate(LinearRegression(), X_bad, y, cv_folds=5, random_state=42, cv_kind="kfold")


def test_cross_validate_output_types_are_json_safe() -> None:
    X, y = _fixture()
    result = cross_validate(LinearRegression(), X, y, cv_folds=5, random_state=42, cv_kind="kfold")
    for value in result.values():
        assert isinstance(value, float)
        assert not isinstance(value, np.floating)
    json.dumps(result)


def test_cross_validate_accepts_pipeline_not_just_bare_estimator() -> None:
    X, y = _fixture()
    pipeline = Pipeline([("passthrough", "passthrough"), ("model", LinearRegression())])
    result = cross_validate(pipeline, X, y, cv_folds=5, random_state=42, cv_kind="kfold")
    assert set(result) == {
        "mae",
        "rmse",
        "r2",
        "mape",
        "max_error",
        "median_ae",
        "explained_var",
    }


def test_make_cv_time_series_split_ignores_random_state() -> None:
    X, y = _fixture()
    first = _make_cv("time_series_split", 5, 1)
    second = _make_cv("time_series_split", 5, 42)
    first_splits = [(train.tolist(), val.tolist()) for train, val in first.split(X)]
    second_splits = [(train.tolist(), val.tolist()) for train, val in second.split(X)]
    assert first_splits == second_splits
    result_1 = cross_validate(
        LinearRegression(), X, y, cv_folds=5, random_state=1, cv_kind="time_series_split"
    )
    result_42 = cross_validate(
        LinearRegression(), X, y, cv_folds=5, random_state=42, cv_kind="time_series_split"
    )
    assert result_1 == result_42


def test_make_cv_time_series_split_folds_are_contiguous() -> None:
    indices = np.arange(100)
    for train_idx, val_idx in _make_cv("time_series_split", 5, 42).split(indices):
        assert max(train_idx) < min(val_idx)
    kfold_splits = list(_make_cv("kfold", 5, 42).split(indices))
    assert any(max(train_idx) >= min(val_idx) for train_idx, val_idx in kfold_splits)


def test_cross_validate_time_series_split_end_to_end() -> None:
    X, y = _fixture()
    result = cross_validate(
        LinearRegression(), X, y, cv_folds=5, random_state=42, cv_kind="time_series_split"
    )
    assert set(result) == {
        "mae",
        "rmse",
        "r2",
        "mape",
        "max_error",
        "median_ae",
        "explained_var",
    }
    assert all(np.isfinite(value) for value in result.values())


def test_cross_validate_unequal_fold_sizes_unweighted_aggregation() -> None:
    X, y = make_regression(n_samples=17, n_features=5, noise=10.0, random_state=42)
    cv = _make_cv("time_series_split", 5, 42)
    per_fold: dict[str, list[float]] = {}
    for train_idx, val_idx in cv.split(X):
        fold = sklearn_cross_validate(
            LinearRegression(),
            X,
            y,
            cv=[(train_idx, val_idx)],
            scoring=_SCORING,
        )
        for name, values in fold.items():
            per_fold.setdefault(name, []).append(float(values[0]))
    expected = _mean_metrics(
        {name: np.asarray(values) for name, values in per_fold.items()}, decimals=4
    )
    result = cross_validate(
        LinearRegression(), X, y, cv_folds=5, random_state=42, cv_kind="time_series_split"
    )
    assert result == expected


def test_make_cv_unknown_kind_raises() -> None:
    with pytest.raises(ValueError):
        _make_cv("bogus", 5, 42)
