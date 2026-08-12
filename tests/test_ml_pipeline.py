from __future__ import annotations

import pandas as pd
import pytest

from broadway.config.loader import load_config
from broadway.features.ml_pipeline import FeaturePipeline
from broadway.features.schema import ENGINEERED_FEATURES, TARGET


@pytest.fixture
def features_cfg():
    cfg = load_config("features", dataset="taxi", experiment="taxi")
    assert cfg.features is not None
    return cfg.features


@pytest.fixture
def lookup_csv(tmp_path) -> str:
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
    n = 50
    return pd.DataFrame(
        {
            "pickup_datetime": pd.date_range("2024-01-01", periods=n, freq="h"),
            "pickup_location_id": [1, 2, 3, 1, 2] * (n // 5),
            "dropoff_location_id": [2, 3, 1, 3, 2] * (n // 5),
            "trip_distance": [1.0 + i * 0.3 for i in range(n)],
            "passenger_count": [1.0] * n,
            TARGET: [5.0 + i * 0.5 for i in range(n)],
        }
    )


@pytest.fixture
def pipeline(features_cfg, lookup_csv) -> FeaturePipeline:
    return FeaturePipeline(
        lookup_path=lookup_csv,
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


def test_fit_learns_encodings(pipeline, taxi_data) -> None:
    pipeline.fit(taxi_data)
    assert pipeline.fitted
    assert pipeline.route_stats is not None
    assert len(pipeline.route_stats) > 0
    assert pipeline.route_frequency is not None
    assert pipeline.global_mean is not None


def test_transform_produces_engineered_features(pipeline, taxi_data) -> None:
    pipeline.fit(taxi_data)
    result = pipeline.transform(taxi_data)

    assert len(result) == len(taxi_data)
    for feat in ENGINEERED_FEATURES:
        assert feat in result.columns, f"missing feature: {feat}"
    assert TARGET not in result.columns


def test_fit_transform(pipeline, taxi_data) -> None:
    result = pipeline.fit_transform(taxi_data)
    assert len(result) == len(taxi_data)
    assert pipeline.fitted


def test_transform_before_fit_raises(pipeline, taxi_data) -> None:
    with pytest.raises(RuntimeError, match="must be fit"):
        pipeline.transform(taxi_data)
