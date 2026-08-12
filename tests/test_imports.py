from __future__ import annotations

import subprocess

import pytest

from broadway.config.loader import load_config
from broadway.config.schema import FeaturesStep


@pytest.fixture
def features_cfg() -> FeaturesStep:
    cfg = load_config("features", dataset="taxi", experiment="taxi")
    assert cfg.features is not None
    return cfg.features


def test_core_imports() -> None:
    from broadway.config.loader import load_config
    from broadway.pipeline import run
    from broadway.cli import main

    assert callable(load_config)
    assert callable(run)
    assert callable(main)


def test_features_imports(features_cfg) -> None:
    from broadway.features.schema import ENGINEERED_FEATURES, RAW_FEATURES, TARGET
    from broadway.features.ml_pipeline import FeaturePipeline

    assert isinstance(ENGINEERED_FEATURES, list)
    assert isinstance(RAW_FEATURES, list)
    assert isinstance(TARGET, str)
    pipeline = FeaturePipeline(
        lookup_path=features_cfg.lookup_path,
        encoding_smoothing=features_cfg.encoding_smoothing,
        frequency_fill=features_cfg.frequency_fill,
        rush_hour_morning_start=features_cfg.rush_hour_morning_start,
        rush_hour_morning_end=features_cfg.rush_hour_morning_end,
        rush_hour_evening_start=features_cfg.rush_hour_evening_start,
        rush_hour_evening_end=features_cfg.rush_hour_evening_end,
        night_start=features_cfg.night_start,
        night_end=features_cfg.night_end,
        passenger_count_min=features_cfg.passenger_count_min,
        passenger_count_max=features_cfg.passenger_count_max,
    )
    assert not pipeline.fitted


def test_etl_imports() -> None:
    from broadway.etl.process import process_data
    from broadway.etl.process_config import rename_map

    assert callable(process_data)
    assert isinstance(rename_map, dict)


def test_config_loads() -> None:
    cfg = load_config("train", dataset="taxi", experiment="taxi")
    assert cfg.dataset is not None
    assert cfg.dataset.name
    assert cfg.train is not None


def test_no_logistics_ml_references() -> None:
    result = subprocess.run(
        ["grep", "-r", "logistics_ml", "src/"], capture_output=True, text=True
    )
    assert result.returncode != 0, f"stale logistics_ml references:\n{result.stdout}"
