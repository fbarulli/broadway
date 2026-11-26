"""cross_validate tests — sklearn.model_selection.cross_validate swap.

The golden dict below was captured from the pre-swap hand-written KFold
loop on the fixture (sklearn 1.7.2). Hardcoded goldens are intentional
brittleness: a sklearn-version shift must fail loud.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate as sklearn_cross_validate
from sklearn.pipeline import Pipeline

from broadway.config.schema import TrainStep
from broadway.evaluate.contracts import BootstrapConfig
from broadway.evaluate.metrics import compute_metrics
from broadway.evaluate.validation import (
    _SCORING,
    TimeSeriesCV,
    _make_cv,
    _mean_metrics,
    cross_validate,
)
from broadway.reports.decision import DecisionRule


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


def test_expanding_cv_gap_never_trains_on_future_rows() -> None:
    X, y = _fixture()
    gap = 10
    for train_idx, val_idx in _make_cv("expanding", 5, 42, gap).split(X):
        assert max(train_idx) + gap < min(val_idx)
    result = cross_validate(
        LinearRegression(),
        X,
        y,
        cv_folds=5,
        random_state=42,
        cv_kind="expanding",
        gap=gap,
    )
    assert all(np.isfinite(value) for value in result.values())
    with pytest.raises(ValueError, match="only applies to an expanding"):
        _make_cv("kfold", 5, 42, gap=1)
    with pytest.raises(ValueError, match="gap must be"):
        _make_cv("expanding", 5, 42, gap=-1)


# --------------------------------------------------------------------------- #
# Time-series CV: rolling windows, seasonal blocks, purge/horizon, datetime cuts
# --------------------------------------------------------------------------- #


def _train_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "random_state": 42,
        "n_jobs": 1,
        "cv_folds": 3,
        "cv_kind": "expanding",
        "cv_gap": 1,
        "model_file": "model.pkl",
        "n_estimators": 10,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "quantile_tail": 0.9,
        "output_dir": "artifacts/training",
        "output_file": "training_result.json",
    }
    kwargs.update(overrides)
    return kwargs


def _seasonal_series(size: int) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    rng = np.random.default_rng(0)
    hours = np.arange(size) % 24
    truth = 50.0 + 20.0 * np.sin(2.0 * np.pi * hours / 24.0) + rng.normal(scale=2.0, size=size)
    features = np.column_stack(
        [
            np.roll(truth, 1),
            np.sin(2.0 * np.pi * hours / 24.0),
            np.cos(2.0 * np.pi * hours / 24.0),
        ]
    )
    dates = pd.date_range("2024-01-01", periods=size, freq="h")
    return features, truth, dates


def test_rolling_cv_keeps_a_fixed_training_window() -> None:
    X = np.arange(1200).reshape(-1, 1)
    rolling = list(_make_cv("rolling", 3, 42, gap=24, train_size=100).split(X))
    assert [len(train) for train, _ in rolling] == [100, 100, 100]
    expanding = list(_make_cv("expanding", 3, 42, gap=24).split(X))
    assert [len(train) for train, _ in expanding] == [276, 576, 876]
    assert _make_cv("rolling", 3, 42, gap=24, train_size=100).get_n_splits() == 3
    with pytest.raises(ValueError, match="does not fit"):
        list(_make_cv("rolling", 3, 42, gap=24, train_size=500).split(X))
    with pytest.raises(ValueError, match="explicit train_size"):
        _make_cv("rolling", 3, 42, gap=24)
    with pytest.raises(ValueError, match="more than 5 time steps"):
        list(_make_cv("expanding", 5, 42).split(np.arange(3).reshape(-1, 1)))
    with pytest.raises(ValueError, match="leaves no training rows"):
        list(_make_cv("expanding", 2, 42, gap=1000).split(np.arange(100).reshape(-1, 1)))
    with pytest.raises(ValidationError, match="requires cv_train_size"):
        TrainStep(**_train_kwargs(cv_kind="rolling"))


def test_seasonal_cv_bounds_folds_on_whole_periods() -> None:
    X = np.arange(600).reshape(-1, 1)
    splitter = _make_cv("expanding", 5, 42, gap=1, seasonal_period=24)
    folds = list(splitter.split(X))
    assert splitter.test_size % 24 == 0
    for _, test in folds:
        assert test[0] % 24 == 0
        assert (test[-1] + 1) % 24 == 0
        assert len(test) % 24 == 0
    with pytest.raises(ValueError, match="whole 24-step periods"):
        list(
            _make_cv("expanding", 5, 42, gap=1, seasonal_period=24).split(
                np.arange(100).reshape(-1, 1)
            )
        )
    with pytest.raises(ValueError, match="whole number of 24-step"):
        list(
            _make_cv("rolling", 3, 42, gap=1, train_size=30, seasonal_period=24).split(
                X
            )
        )


def test_cv_gap_below_horizon_is_refused() -> None:
    with pytest.raises(ValidationError, match="cv_gap"):
        TrainStep(**_train_kwargs(cv_kind="expanding", cv_gap=1, cv_horizon=24))
    step = TrainStep(**_train_kwargs(cv_kind="expanding", cv_gap=24, cv_horizon=24))
    assert (step.cv_gap, step.cv_horizon) == (24, 24)
    with pytest.raises(ValidationError, match="only apply"):
        TrainStep(**_train_kwargs(cv_kind="kfold", cv_horizon=2))


def test_datetime_cuts_use_the_time_grid_and_fall_back_loudly(caplog) -> None:
    kept_positions = np.delete(np.arange(125), [100, 101, 102, 103, 104])
    dates = pd.date_range("2024-01-01", periods=125, freq="h")[kept_positions]
    X = np.arange(kept_positions.size).reshape(-1, 1)

    timed = _make_cv("expanding", 4, 42, gap=1, dates=dates)
    timed_folds = list(timed.split(X))
    assert timed.time_indexed is True
    assert timed.test_size == 25  # 125 grid steps / 5 blocks
    assert len(timed_folds[-1][1]) == 20  # 25 positions minus 5 missing hours

    with caplog.at_level(logging.INFO):
        row_order = _make_cv("expanding", 4, 42, gap=1)
        row_folds = list(row_order.split(X))
    assert row_order.time_indexed is False
    assert row_order.test_size == 24  # 120 rows / 5 blocks
    assert len(row_folds[-1][1]) == 24
    assert "row order" in caplog.text

    irregular = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        np.cumsum(np.tile([5, 7], 60)), unit="min"
    )
    assert _make_cv("expanding", 4, 42, gap=1, dates=irregular).time_indexed is False
    duplicate = pd.date_range("2024-01-01", periods=119, freq="h")
    duplicate = duplicate.insert(5, duplicate[5])
    assert _make_cv("expanding", 4, 42, gap=1, dates=duplicate).time_indexed is False
    with pytest.raises(ValueError, match="non-decreasing"):
        _make_cv("expanding", 4, 42, gap=1, dates=dates[::-1])
    with pytest.raises(ValueError, match="missing values"):
        _make_cv("expanding", 4, 42, gap=1, dates=pd.Series([pd.NaT, pd.Timestamp("2024-01-01")]))
    single = pd.Series([pd.Timestamp("2024-01-01")])
    assert _make_cv("expanding", 4, 42, gap=1, dates=single).time_indexed is False
    with pytest.raises(ValueError, match="time index must align"):
        list(_make_cv("expanding", 4, 42, gap=1, dates=dates[:100]).split(X))


def test_multi_step_cv_scores_every_horizon_and_origin() -> None:
    features, truth, dates = _seasonal_series(400)
    step = TrainStep(**_train_kwargs(cv_folds=3, cv_gap=24, cv_horizon=24))
    result, decision = TimeSeriesCV(
        step, "demand", None, BootstrapConfig(iterations=20, seed=1)
    ).run(LinearRegression(), features, truth, dates=dates)
    assert decision is None
    assert list(result.horizon_metrics) == list(range(1, 25))
    assert result.origins and all(origin.n_steps == 24 for origin in result.origins)
    # 3 folds x 100 test rows -> 4 whole 24-step origins each; 12 rows left over.
    assert len(result.origins) == 12
    assert sum(origin.n_steps for origin in result.origins) == 288
    assert all(np.isfinite(origin.rmse) for origin in result.origins)

    single = TrainStep(**_train_kwargs(cv_folds=3, cv_gap=1, cv_horizon=1))
    single_result, _ = TimeSeriesCV(single, "demand", None, BootstrapConfig(iterations=5)).run(
        LinearRegression(), features, truth, dates=dates
    )
    assert single_result.origins == []
    assert list(single_result.horizon_metrics) == [1]


def test_cv_decision_ships_a_candidate_that_beats_the_seasonal_naive() -> None:
    features, truth, dates = _seasonal_series(600)
    rule = DecisionRule(
        metric="rmse",
        direction="max",
        min_effect=0.5,
        min_sample_size=100,
        action="ship the candidate",
        owner="probe-owner",
    )
    step = TrainStep(**_train_kwargs(cv_folds=5, cv_gap=24, cv_horizon=24, cv_seasonal_period=24))
    result, decision = TimeSeriesCV(
        step, "demand", rule, BootstrapConfig(iterations=100, seed=7)
    ).run(LinearRegression(), features, truth, dates=dates)
    assert decision is not None
    assert decision.verdict == "act"
    assert decision.value == pytest.approx(result.improvement["rmse"])
    assert decision.interval_low <= decision.value <= decision.interval_high
    assert decision.evidence.baseline.startswith("seasonal-naive")
    assert len(decision.evidence.segments) == step.cv_folds
    assert decision.owner == "probe-owner"

    bad_rule = rule.model_copy(update={"metric": "bogus"})
    with pytest.raises(ValueError, match="not in the scored vocabulary"):
        TimeSeriesCV(step, "demand", bad_rule, BootstrapConfig(iterations=5)).run(
            LinearRegression(), features, truth, dates=dates
        )
