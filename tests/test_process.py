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
from broadway.etl.process_config import (
    min_trip_duration_minutes,
    max_trip_duration_minutes,
    min_trip_distance,
    max_trip_distance,
    rename_map,
)
from broadway.features.schema import TARGET


@pytest.fixture
def raw_trips() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_distance": [2.5, min_trip_distance, max_trip_distance + 10, 3.0, 1.0],
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
    assert len(result) == len(raw_trips)
    assert list(result.columns) == list(raw_trips.columns)


def test_filter_valid_trips(raw_trips: pd.DataFrame) -> None:
    filtered = filter_valid_trips(raw_trips)
    assert len(filtered) < len(raw_trips)
    assert float(filtered["trip_distance"].min()) > min_trip_distance
    assert float(filtered["trip_distance"].max()) < max_trip_distance


def test_compute_trip_duration(raw_trips: pd.DataFrame) -> None:
    valid = raw_trips.head(1).copy()
    result = compute_trip_duration(valid)
    assert TARGET in result.columns
    assert float(result[TARGET].iloc[0]) == pytest.approx(20.0)


def test_filter_valid_duration() -> None:
    df = pd.DataFrame({
        TARGET: [
            min_trip_duration_minutes - 1,
            min_trip_duration_minutes + 1,
            max_trip_duration_minutes + 1,
            (min_trip_duration_minutes + max_trip_duration_minutes) / 2,
            (min_trip_duration_minutes + max_trip_duration_minutes) / 2,
        ]
    })
    filtered = filter_valid_duration(df)
    assert len(filtered) == 3
    assert float(filtered[TARGET].min()) >= min_trip_duration_minutes
    assert float(filtered[TARGET].max()) <= max_trip_duration_minutes


def test_rename_columns() -> None:
    df = pd.DataFrame({old: [1] for old in rename_map})
    result = rename_columns(df)
    assert "pickup_location_id" in result.columns
    assert "dropoff_location_id" in result.columns


def test_select_and_clean_columns(raw_trips: pd.DataFrame) -> None:
    renamed = rename_columns(raw_trips)
    renamed[TARGET] = [10.0, 20.0, None, 30.0, 40.0]
    result = select_and_clean_columns(renamed)
    assert TARGET in result.columns
    assert result[TARGET].notna().all()
    assert len(result) < len(renamed)
