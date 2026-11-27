"""Public behavior of the 08_timeseries lane: series, record and step page."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from broadway.config.schema import (
    BootstrapConfig,
    PermutationConfig,
    SeriesSpec,
)
from broadway.reports import experiments_dashboard as ui
from broadway.reports import story_page
from broadway.reports.decision import Decision, DecisionRule
from broadway.reports.story_models import StoryConfig, StoryStepMap
from broadway.timeseries.analysis import SeriesDecisionBuilder
from broadway.timeseries.backtest import (
    BacktestResult,
    FoldResult,
    SeasonalNaive,
)
from broadway.timeseries.series import DailySeriesBuilder


def _daily_frame() -> pd.DataFrame:
    """Four weeks of orders; each week adds 10 to the previous weekly pattern."""
    dates = pd.date_range("2020-01-06", periods=28, freq="D")  # a Monday
    rows: list[dict[str, object]] = []
    for index, day in enumerate(dates):
        week, weekday = divmod(index, 7)
        count = 10 + weekday * 10 + week * 10
        for _ in range(count):
            rows.append({"ts": day, "payment": 100.0})
    return pd.DataFrame(rows)


def test_daily_series_zero_fills_and_has_the_known_seasonal_naive_mae() -> None:
    frame = _daily_frame()
    builder = DailySeriesBuilder(
        datetime_column="ts",
        grain_column="order_id",
        start=date(2020, 1, 6),
        end=date(2020, 2, 2),
    )
    count_spec = SeriesSpec(label="daily orders", unit="orders/day", aggregation="count")
    series = builder.build(frame, "orders", count_spec)

    assert series.n == 28
    assert series.values[:7].tolist() == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    # Week 3 is last week's pattern + 10, so same-weekday forecasting misses by 10.
    forecast = SeasonalNaive(period=7).forecast(series.values[:21], horizon=7)
    mae = float(np.mean(np.abs(series.values[21:] - forecast)))
    assert mae == pytest.approx(10.0)

    # A missing calendar day is a zero on the grid, never an interpolated point.
    sparse = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2020-01-06", "2020-01-08"]),
            "payment": [5.0, 7.0],
        }
    )
    sparse_builder = DailySeriesBuilder(
        datetime_column="ts",
        grain_column="order_id",
        start=date(2020, 1, 6),
        end=date(2020, 1, 8),
    )
    revenue_spec = SeriesSpec(
        label="daily revenue", unit="BRL/day", aggregation="sum", column="payment"
    )
    revenue = sparse_builder.build(sparse, "revenue", revenue_spec)
    assert revenue.values.tolist() == [5.0, 0.0, 7.0]


def _result() -> BacktestResult:
    """Two folds: baseline 10 -> model 8, and baseline 20 -> model 14."""
    dates = pd.date_range("2020-01-06", periods=28, freq="D")
    baseline = np.asarray([10.0] * 14 + [20.0] * 14)
    model = np.asarray([8.0] * 14 + [14.0] * 14)
    fold_ids = np.asarray([1] * 14 + [2] * 14)
    folds = (
        FoldResult(
            index=1,
            train_start=dates[0],
            train_end=dates[6],
            test_start=dates[7],
            test_end=dates[13],
            train_size=7,
            test_size=14,
            baseline_mae=10.0,
            model_mae=8.0,
            baseline_rmse=10.0,
            model_rmse=8.0,
            relative_improvement=0.2,
        ),
        FoldResult(
            index=2,
            train_start=dates[0],
            train_end=dates[13],
            test_start=dates[14],
            test_end=dates[27],
            train_size=14,
            test_size=14,
            baseline_mae=20.0,
            model_mae=14.0,
            baseline_rmse=20.0,
            model_rmse=14.0,
            relative_improvement=0.3,
        ),
    )
    return BacktestResult(
        folds=folds,
        dates=dates,
        horizon_positions=np.tile(np.arange(14), 2),
        fold_ids=fold_ids,
        baseline_errors=baseline,
        model_errors=model,
    )


def _decision() -> Decision:
    rule = DecisionRule(
        metric="relative_mae_improvement",
        direction="max",
        min_effect=0.05,
        null_quantile=0.95,
        min_sample_size=28,
        action="keep the seasonal-naive baseline unless the interval clears the null",
        owner="olist",
    )
    builder = SeriesDecisionBuilder(
        rule,
        BootstrapConfig(iterations=50, seed=7, level=0.95),
        PermutationConfig(iterations=50, seed=7),
    )
    return builder.build(
        result=_result(),
        question="Does the model beat the baseline?",
        provenance={"dataset": "fixture", "series": "orders"},
    )


def test_series_decision_record_validates_through_the_canonical_model() -> None:
    decision = _decision()

    validated = Decision.model_validate_json(decision.model_dump_json())

    assert validated.metric == "relative_mae_improvement"
    assert validated.sample_size == 28
    assert validated.interval_low <= validated.value <= validated.interval_high
    assert validated.null_threshold is not None
    assert validated.verdict in {"act", "investigate", "no_action"}


def test_timeseries_step_page_renders_the_published_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "experiments"
    step_dir = root / "results" / "pipeline" / "08_timeseries"
    step_dir.mkdir(parents=True)
    decision = _decision()
    (step_dir / "08_timeseries_decision.json").write_text(
        decision.model_dump_json(), encoding="utf-8"
    )
    (step_dir / "08_timeseries.png").write_bytes(b"png")
    config = StoryConfig.fallback().model_copy(
        update={"steps": StoryStepMap(configured=("08_timeseries",), planned=())}
    )
    monkeypatch.setattr(story_page, "_CONFIG", config)
    monkeypatch.setattr(ui, "EXPERIMENTS_ROOT", root)
    monkeypatch.setenv("BROADWAY_PROGRESS_DIR", str(tmp_path / "progress"))

    rows = ui.census_rows("pipeline")
    assert rows == [
        {
            "experiment": "08_timeseries",
            "question": "Does the model beat the baseline?",
            "status": decision.verdict,
            "artifacts": 2,
        }
    ]
    response = ui.experiment_page("08_timeseries", "pipeline")
    assert response.status_code == 200
    body = bytes(response.body).decode("utf-8")
    assert "relative_mae_improvement" in body
    assert 'class="decision act"' in body
    assert '<img src="/results/pipeline/08_timeseries/08_timeseries.png"' in body
