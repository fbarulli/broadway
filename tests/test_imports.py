from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def test_core_imports() -> None:
    from broadway.config.loader import load_config
    from broadway.pipeline import run
    from broadway.cli import main

    assert callable(load_config)
    assert callable(run)
    assert callable(main)


def test_features_imports() -> None:
    from broadway.features.schema import ENGINEERED_FEATURES, RAW_FEATURES, TARGET
    from broadway.features.ml_pipeline import FeaturePipeline

    assert isinstance(ENGINEERED_FEATURES, list)
    assert isinstance(RAW_FEATURES, list)
    assert isinstance(TARGET, str)
    pipeline = FeaturePipeline(
        lookup_path="data/raw/taxi_zone_lookup.csv",
        encoding_smoothing=50,
        frequency_fill=0,
        rush_hour_morning_start=7,
        rush_hour_morning_end=9,
        rush_hour_evening_start=16,
        rush_hour_evening_end=19,
        night_start=22,
        night_end=5,
        passenger_count_min=1,
        passenger_count_max=6,
    )
    assert not pipeline.fitted


def test_etl_imports() -> None:
    from broadway.etl.process import process_data
    from broadway.etl.process_config import rename_map

    assert callable(process_data)
    assert isinstance(rename_map, dict)


def test_config_loads() -> None:
    from broadway.config.loader import load_config

    cfg = load_config("train", dataset="taxi", experiment="taxi")
    assert cfg.dataset is not None
    assert cfg.dataset.name == "taxi"
    assert cfg.train is not None


def test_no_logistics_ml_references() -> None:
    root = Path("src")
    py_files = list(root.rglob("*.py"))
    assert len(py_files) > 0, "no Python files found under src/"

    result = subprocess.run(
        ["grep", "-r", "logistics_ml", "src/"], capture_output=True, text=True
    )
    assert result.returncode != 0, f"stale logistics_ml references:\n{result.stdout}"
