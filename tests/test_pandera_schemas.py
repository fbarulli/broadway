"""Pandera schema tests: synthetic fixtures, no real data load."""

from __future__ import annotations

import pandas as pd
import pandera.errors
import pytest

from broadway.config.schema import ColumnSchema, ColumnRole, DatasetContract
from broadway.contracts.pandera import build_raw_schema
from project.features import ENGINEERED_FEATURES, ENGINEERED_SCHEMA

_RAW_COLUMNS = {
    "pickup_datetime": ColumnSchema(
        dtype="datetime64[us]", null_count=0, role=ColumnRole.DATETIME
    ),
    "passenger_count": ColumnSchema(
        dtype="float64", null_count=0, role=ColumnRole.FEATURE
    ),
    "trip_distance": ColumnSchema(
        dtype="float64", null_count=0, role=ColumnRole.FEATURE
    ),
    "pickup_location_id": ColumnSchema(
        dtype="int32", null_count=0, role=ColumnRole.FEATURE
    ),
    "dropoff_location_id": ColumnSchema(
        dtype="int32", null_count=0, role=ColumnRole.FEATURE
    ),
    "trip_duration_minutes": ColumnSchema(
        dtype="float64", null_count=0, role=ColumnRole.TARGET
    ),
}


def _contract() -> DatasetContract:
    return DatasetContract(
        name="taxi",
        path="data/processed/training_data.parquet",
        target="trip_duration_minutes",
        task="regression",
        datetime_column="pickup_datetime",
        columns=_RAW_COLUMNS,
        lookup_tables={},
        row_count=1,
    )


def _valid_raw(contract: DatasetContract) -> pd.DataFrame:
    frames: dict[str, pd.Series] = {}
    for name, col in contract.columns.items():
        if col.dtype.startswith("datetime"):
            frames[name] = pd.to_datetime(
                ["2024-01-01 00:00:00", "2024-01-01 01:00:00", "2024-01-01 02:00:00"]
            )
        elif col.dtype.startswith("int"):
            frames[name] = pd.Series([1, 2, 3], dtype=col.dtype)
        else:
            frames[name] = pd.Series([1.0, 2.0, 3.0], dtype=col.dtype)
    return pd.DataFrame(frames)


_ENGINEERED_DTYPES = {
    "pickup_hour": "int32",
    "pickup_day_of_week": "int32",
    "pickup_month": "int32",
    "passenger_count": "float64",
    "trip_distance": "float64",
    "pickup_location_id": "int32",
    "dropoff_location_id": "int32",
    "is_weekend": "int8",
    "rush_hour": "int8",
    "is_night": "int8",
    "log_distance": "float64",
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


def test_build_raw_schema_validates_and_rejects() -> None:
    contract = _contract()
    schema = build_raw_schema(contract)
    assert list(schema.columns.keys()) == list(contract.columns.keys())

    valid = _valid_raw(contract)
    schema.validate(valid)

    with pytest.raises(pandera.errors.SchemaError):
        schema.validate(valid.drop(columns=["trip_distance"]))

    wrong = valid.copy()
    wrong["pickup_location_id"] = wrong["pickup_location_id"].astype("float64")
    with pytest.raises(pandera.errors.SchemaError):
        schema.validate(wrong)


def test_engineered_features_schema_validates_and_rejects() -> None:
    valid = _valid_engineered()
    assert list(valid.columns) == list(ENGINEERED_FEATURES)
    ENGINEERED_SCHEMA.validate(valid)

    with pytest.raises(pandera.errors.SchemaError):
        ENGINEERED_SCHEMA.validate(valid.drop(columns=["log_distance"]))

    wrong = valid.copy()
    wrong["pickup_hour"] = wrong["pickup_hour"].astype("float64")
    with pytest.raises(pandera.errors.SchemaError):
        ENGINEERED_SCHEMA.validate(wrong)
