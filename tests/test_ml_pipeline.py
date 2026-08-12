from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from broadway.config.loader import load_config
from broadway.config.schema import FeaturesStep
from project.ml_pipeline import FeaturePipeline
from broadway.features.schema import TARGET
from project.features import ENGINEERED_FEATURES


@pytest.fixture
def features_cfg() -> FeaturesStep:
    cfg = load_config("features", dataset="taxi", experiment="taxi")
    assert cfg.features is not None
    return cfg.features


@pytest.fixture
def lookup_csv(tmp_path: Path) -> str:
    path = tmp_path / "zones.csv"
    pd.DataFrame(
        {
            "LocationID": [1, 2, 3],
            "Borough": ["Manhattan", "Brooklyn", "Queens"],
        }
    ).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def taxi_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pickup_datetime": pd.date_range("2024-01-01", periods=50, freq="h"),
            "pickup_location_id": pd.Series([1, 2, 3, 1, 2] * 10, dtype="int32"),
            "dropoff_location_id": pd.Series([2, 3, 1, 3, 2] * 10, dtype="int32"),
            "trip_distance": [1.0 + i * 0.3 for i in range(50)],
            "passenger_count": [1.0] * 50,
            TARGET: [5.0 + i * 0.5 for i in range(50)],
        }
    )


@pytest.fixture
def pipeline(features_cfg: FeaturesStep, lookup_csv: str) -> FeaturePipeline:
    kwargs: dict[str, Any] = features_cfg.model_dump(
        exclude={"pipeline_file", "borough_column", "borough_lookup_column", "lookup_path"}
    )
    return FeaturePipeline(lookup_path=lookup_csv, **kwargs)


def test_fit_learns_encodings(pipeline: FeaturePipeline, taxi_data: pd.DataFrame) -> None:
    pipeline.fit(taxi_data)
    assert pipeline.fitted
    assert pipeline.route_stats is not None
    assert len(pipeline.route_stats) > 0
    assert pipeline.route_frequency is not None
    assert pipeline.global_mean is not None


def test_transform_produces_engineered_features(
    pipeline: FeaturePipeline, taxi_data: pd.DataFrame
) -> None:
    pipeline.fit(taxi_data)
    result = pipeline.transform(taxi_data)

    assert len(result) == len(taxi_data)
    for feat in ENGINEERED_FEATURES:
        assert feat in result.columns, f"missing feature: {feat}"
    assert TARGET not in result.columns


def test_fit_transform(pipeline: FeaturePipeline, taxi_data: pd.DataFrame) -> None:
    result = pipeline.fit_transform(taxi_data)
    assert len(result) == len(taxi_data)
    assert pipeline.fitted


def test_transform_before_fit_raises(
    pipeline: FeaturePipeline, taxi_data: pd.DataFrame
) -> None:
    with pytest.raises(RuntimeError, match="must be fit"):
        pipeline.transform(taxi_data)
