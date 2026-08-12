"""Pandera schema tests: synthetic fixtures, no real data load."""

from __future__ import annotations

import pandas as pd
import pandera.errors
import pytest

from broadway.features.schema import (
    ENGINEERED_FEATURES,
    EngineeredFeaturesSchema,
)
from projects.taxi import data
from projects.taxi.schemas import TaxiRawSchema


def _valid_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            data.DATETIME_COL: pd.to_datetime(
                ["2024-01-01 00:00:00", "2024-01-01 01:00:00", "2024-01-01 02:00:00"]
            ),
            data.PASSENGER_COUNT_COL: pd.Series([1.0, 2.0, 3.0], dtype="float32"),
            data.TRIP_DISTANCE_COL: pd.Series([1.5, 2.5, 3.5], dtype="float32"),
            data.PICKUP_LOCATION_COL: pd.Series([1, 2, 3], dtype="int32"),
            data.DROPOFF_LOCATION_COL: pd.Series([4, 5, 6], dtype="int32"),
            data.TARGET_COL: pd.Series([10.0, 20.0, 30.0], dtype="float32"),
            data.ZONE_ID_COL: pd.Series([1, 2, 3], dtype="int64"),
            data.PICKUP_BOROUGH_COL: ["Manhattan", "Brooklyn", "Queens"],
        }
    )


_ENGINEERED_DTYPES = {
    "pickup_hour": "int32",
    "pickup_day_of_week": "int32",
    "pickup_month": "int32",
    "passenger_count": "float32",
    "trip_distance": "float32",
    "pickup_location_id": "int32",
    "dropoff_location_id": "int32",
    "is_weekend": "int8",
    "rush_hour": "int8",
    "is_night": "int8",
    "log_distance": "float32",
    "same_borough": "int8",
    "route_avg_duration": "float64",
    "route_frequency": "int32",
}


def _valid_engineered() -> pd.DataFrame:
    return pd.DataFrame(
        {
            col: pd.Series([0, 1, 2], dtype=dtype)
            for col, dtype in _ENGINEERED_DTYPES.items()
        }
    )


def test_taxi_raw_schema_validates_and_rejects() -> None:
    valid = _valid_raw()
    TaxiRawSchema.validate(valid)

    with pytest.raises(pandera.errors.SchemaError):
        TaxiRawSchema.validate(valid.drop(columns=[data.TRIP_DISTANCE_COL]))

    wrong = valid.copy()
    wrong[data.PICKUP_LOCATION_COL] = wrong[data.PICKUP_LOCATION_COL].astype("float64")
    with pytest.raises(pandera.errors.SchemaError):
        TaxiRawSchema.validate(wrong)


def test_engineered_features_schema_validates_and_rejects() -> None:
    valid = _valid_engineered()
    assert list(valid.columns) == ENGINEERED_FEATURES
    EngineeredFeaturesSchema.validate(valid)

    with pytest.raises(pandera.errors.SchemaError):
        EngineeredFeaturesSchema.validate(valid.drop(columns=["log_distance"]))

    wrong = valid.copy()
    wrong["pickup_hour"] = wrong["pickup_hour"].astype("float64")
    with pytest.raises(pandera.errors.SchemaError):
        EngineeredFeaturesSchema.validate(wrong)
