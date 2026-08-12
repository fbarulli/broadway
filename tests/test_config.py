from __future__ import annotations

import pytest

from broadway.config.loader import load_config
from broadway.etl.process_config import (
    min_trip_duration_minutes,
    max_trip_duration_minutes,
    min_trip_distance,
    max_trip_distance,
    rename_map,
)


@pytest.fixture
def train_cfg():
    cfg = load_config("train", dataset="taxi", experiment="taxi")
    assert cfg.train is not None
    return cfg.train


@pytest.fixture
def evaluate_cfg():
    cfg = load_config("evaluate", dataset="taxi", experiment="taxi")
    assert cfg.evaluate is not None
    return cfg.evaluate


@pytest.fixture
def etl_cfg():
    cfg = load_config("etl", dataset="taxi", experiment="taxi")
    assert cfg.etl is not None
    return cfg.etl


@pytest.fixture
def stats_cfg():
    cfg = load_config("stats", dataset="taxi", experiment="taxi")
    assert cfg.stats is not None
    return cfg.stats


def test_load_train_config(train_cfg) -> None:
    assert train_cfg.model_file
    assert train_cfg.random_state


def test_load_evaluate_config(evaluate_cfg) -> None:
    assert evaluate_cfg.target_metric
    assert evaluate_cfg.promotion_threshold


def test_load_etl_with_pipeline_fields(etl_cfg) -> None:
    assert etl_cfg.min_trip_distance == min_trip_distance
    assert etl_cfg.max_trip_distance == max_trip_distance
    assert etl_cfg.min_trip_duration_minutes == min_trip_duration_minutes
    assert etl_cfg.max_trip_duration_minutes == max_trip_duration_minutes
    assert etl_cfg.rename_map == rename_map


def test_missing_dataset_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("train", dataset="nonexistent", experiment="taxi")


def test_missing_experiment_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("full", dataset="taxi", experiment="nonexistent")


def test_invalid_step_raises() -> None:
    with pytest.raises(ValueError, match="unknown step"):
        load_config("bogus")


def test_stats_config_fields(stats_cfg) -> None:
    assert stats_cfg.min_rows_for_sampling
    assert stats_cfg.group_values
