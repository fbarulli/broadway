from __future__ import annotations

import pandas as pd
import pytest

from broadway.etl.process import (
    compute_trip_duration,
    filter_valid_duration,
    filter_valid_trips,
    read_raw_data,
    rename_columns,
    select_and_clean_columns,
)
from broadway.features.schema import TARGET


@pytest.fixture
def raw_trips() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_distance": [2.5, 0.0, 60.0, 3.0, 1.0],
            "tpep_pickup_datetime": pd.to_datetime(
                ["2024-01-01 10:00", "2024-01-01 11:00", "2024-01-01 12:00",
                 "2024-01-01 13:00", "2024-01-01 14:00"]
            ),
            "tpep_dropoff_datetime": pd.to_datetime(
                ["2024-01-01 10:20", "2024-01-01 10:50", "2024-01-01 13:00",
                 "2024-01-01 13:30", "2024-01-01 14:10"]
            ),
            "PULocationID": [1, 2, 3, 4, 5],
            "DOLocationID": [1, 2, 3, 4, 5],
            "passenger_count": [1.0, 2.0, 1.0, 3.0, 1.0],
        }
    )


def test_read_raw_data(raw_trips: pd.DataFrame, tmp_path) -> None:
    f = tmp_path / "yellow_tripdata_2024-01.parquet"
    raw_trips.to_parquet(f)
    result = read_raw_data([f])
    assert len(result) == 5
    assert list(result.columns) == list(raw_trips.columns)


def test_filter_valid_trips(raw_trips: pd.DataFrame) -> None:
    filtered = filter_valid_trips(raw_trips)
    assert len(filtered) == 3
    assert float(filtered["trip_distance"].min()) > 0.0
    assert float(filtered["trip_distance"].max()) < 50.0


def test_compute_trip_duration(raw_trips: pd.DataFrame) -> None:
    valid = raw_trips.head(1).copy()
    result = compute_trip_duration(valid)
    assert TARGET in result.columns
    assert float(result[TARGET].iloc[0]) == pytest.approx(20.0)


def test_filter_valid_duration() -> None:
    df = pd.DataFrame(
        {TARGET: [0.0, 10.0, 200.0, 30.0, 90.0]}
    )
    filtered = filter_valid_duration(df)
    assert len(filtered) == 3
    assert float(filtered[TARGET].min()) >= 1.0
    assert float(filtered[TARGET].max()) <= 180.0


def test_rename_columns() -> None:
    df = pd.DataFrame({"PULocationID": [1], "DOLocationID": [2]})
    result = rename_columns(df)
    assert "pickup_location_id" in result.columns
    assert "dropoff_location_id" in result.columns


def test_select_and_clean_columns(raw_trips: pd.DataFrame) -> None:
    renamed = rename_columns(raw_trips)
    renamed[TARGET] = [10.0, 20.0, None, 30.0, 40.0]
    result = select_and_clean_columns(renamed)
    assert TARGET in result.columns
    assert result[TARGET].notna().all()
    assert len(result) == 4
