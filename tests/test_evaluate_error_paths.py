"""Public error/validation behavior of the evaluate metrics and error ledger."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from broadway.config.loader import load_config
from broadway.evaluate.contracts import DecisionConfig, ErrorAnalysisConfig, ThresholdGrid
from broadway.evaluate.metrics import confusion_at, tune_threshold
from broadway.evaluate.module import HoldoutErrorAnalysis
from broadway.features.plan import ColumnPlan


def _plan() -> ColumnPlan:
    frame = pd.DataFrame({"rooms": [1, 2], "area": [10.0, 20.0], "price": [100.0, 200.0]})
    return ColumnPlan.from_frame_columns(frame, ["rooms", "area"], target="price")


def _features() -> pd.DataFrame:
    return pd.DataFrame({"rooms": [1, 2, 3], "area": [10.0, 20.0, 30.0]})


def test_error_ledger_writes_only_configured_segment_columns(tmp_path: Path) -> None:
    analysis = HoldoutErrorAnalysis(
        ErrorAnalysisConfig(segment_columns=["rooms"]), None, _plan(), "price"
    )

    analysis.write(
        tmp_path,
        _features(),
        np.array([100.0, 200.0, 300.0]),
        np.array([110.0, 190.0, 330.0]),
    )

    ledger = pd.read_parquet(tmp_path / "errors.parquet")
    assert list(ledger.columns) == ["y_true", "y_pred", "residual", "abs_error", "rooms"]
    assert not (tmp_path / "thresholds.json").exists()


def test_error_ledger_rejects_segment_columns_outside_the_plan(tmp_path: Path) -> None:
    analysis = HoldoutErrorAnalysis(
        ErrorAnalysisConfig(segment_columns=["rooms", "ghost"]), None, _plan(), "price"
    )

    with pytest.raises(ValueError, match="not in the experiment column plan"):
        analysis.write(
            tmp_path,
            _features(),
            np.array([100.0, 200.0, 300.0]),
            np.array([110.0, 190.0, 330.0]),
        )


def test_error_analysis_rejects_a_decision_for_another_column() -> None:
    decision = DecisionConfig(
        column="rooms", threshold=2.0, objective="f1", grid=ThresholdGrid(min=0.0, max=1.0, steps=2)
    )

    with pytest.raises(ValueError, match="is not the scored target"):
        HoldoutErrorAnalysis(ErrorAnalysisConfig(), decision, _plan(), "price")


def test_error_analysis_from_config_requires_dataset_experiment_and_evaluate() -> None:
    cfg = load_config(
        "evaluate", dataset="test", experiment="baseline", environment="development"
    )

    with pytest.raises(ValueError, match="requires dataset, experiment, and evaluate"):
        HoldoutErrorAnalysis.from_config(cfg.model_copy(update={"dataset": None}))


def test_confusion_at_rejects_malformed_binary_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        confusion_at(np.array([1, 0]), np.array([0.4]), threshold=0.5)
    with pytest.raises(ValueError, match="one-dimensional"):
        confusion_at(np.array([[1, 0]]), np.array([[0.4, 0.6]]), threshold=0.5)
    with pytest.raises(ValueError, match="NaN or infinite"):
        confusion_at(np.array([np.nan, 0]), np.array([0.4, 0.6]), threshold=0.5)
    with pytest.raises(ValueError, match="binary truth labels"):
        confusion_at(np.array([2, 0]), np.array([0.4, 0.6]), threshold=0.5)
    with pytest.raises(ValueError, match="y_pred contains"):
        confusion_at(np.array([1, 0]), np.array([np.inf, 0.6]), threshold=0.5)


def test_confusion_at_on_an_empty_cutoff_reports_zero_rates() -> None:
    result = confusion_at(np.array([]), np.array([]), threshold=0.5)

    assert result["tp"] == result["fp"] == result["tn"] == result["fn"] == 0
    assert result["accuracy"] == 0.0
    assert result["f1"] == 0.0


def test_tune_threshold_validates_objective_and_grid() -> None:
    truth = np.array([1, 0])
    scores = np.array([0.9, 0.1])

    with pytest.raises(ValueError, match="unknown tuning objective"):
        tune_threshold(truth, scores, [0.5], "auc")
    with pytest.raises(ValueError, match="at least one candidate cutoff"):
        tune_threshold(truth, scores, [], "f1")
    with pytest.raises(ValueError, match="thresholds contains NaN or infinite"):
        tune_threshold(truth, scores, [0.5, np.nan], "f1")
