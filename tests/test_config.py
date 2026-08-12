from __future__ import annotations

import pytest
from pydantic import ValidationError

from broadway.config.loader import load_config


def test_load_train_config() -> None:
    cfg = load_config("train", dataset="taxi", experiment="taxi")
    assert cfg.dataset is not None
    assert cfg.dataset.name == "taxi"
    assert cfg.train is not None
    assert cfg.train.model_file == "model.pkl"


def test_load_evaluate_config() -> None:
    cfg = load_config("evaluate", dataset="taxi", experiment="taxi")
    assert cfg.evaluate is not None
    assert cfg.evaluate.target_metric == "rmse"


def test_load_etl_with_pipeline_fields() -> None:
    cfg = load_config("etl", dataset="taxi", experiment="taxi")
    assert cfg.etl is not None
    assert cfg.etl.min_trip_distance == 0.0
    assert cfg.etl.max_trip_distance == 50.0
    assert cfg.etl.rename_map["PULocationID"] == "pickup_location_id"


def test_missing_dataset_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("train", dataset="nonexistent", experiment="taxi")


def test_missing_experiment_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("full", dataset="taxi", experiment="nonexistent")


def test_invalid_step_raises() -> None:
    with pytest.raises(ValueError, match="unknown step"):
        load_config("bogus")


def test_stats_config_fields() -> None:
    cfg = load_config("stats", dataset="taxi", experiment="taxi")
    assert cfg.stats is not None
    assert cfg.stats.data_path == "data/processed/training_data.parquet"
    assert cfg.stats.min_rows_for_sampling == 10000
    assert "Manhattan" in cfg.stats.group_values
