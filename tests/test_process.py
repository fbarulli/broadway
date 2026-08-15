from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import broadway.etl.process_config as process_config
import broadway.lineage.records as lineage_records
from broadway.config.loader import load_config
from broadway.config.schema import DatasetContract
from broadway.etl.process import (
    compute_trip_duration,
    filter_valid_duration,
    filter_valid_passenger_count,
    filter_valid_trips,
    process_data,
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
from broadway.lineage.models import LineageRecord


@pytest.fixture
def contract() -> DatasetContract:
    cfg = load_config("contracts", dataset="taxi", experiment="taxi")
    assert cfg.dataset is not None
    return cfg.dataset


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
            "PULocationID": pd.Series([1, 2, 3, 4, 5], dtype="int32"),
            "DOLocationID": pd.Series([1, 2, 3, 4, 5], dtype="int32"),
            "passenger_count": [1.0, 2.0, 1.0, 3.0, 1.0],
            "fare_amount": [20.0, 15.0, 8.0, 30.0, 12.0],
        }
    )


def test_read_raw_data(raw_trips: pd.DataFrame, tmp_path: Path) -> None:
    f = tmp_path / "yellow_tripdata_2024-01.parquet"
    raw_trips.to_parquet(f)
    result = read_raw_data([f])
    assert len(result) == len(raw_trips)
    assert set(result.columns) == set(raw_trips.columns)


def test_filter_valid_trips(raw_trips: pd.DataFrame) -> None:
    filtered = filter_valid_trips(raw_trips)
    assert len(filtered) < len(raw_trips)
    assert float(filtered["trip_distance"].min()) > min_trip_distance
    assert float(filtered["trip_distance"].max()) < max_trip_distance


def test_compute_trip_duration(raw_trips: pd.DataFrame) -> None:
    valid = raw_trips.head(1).copy()
    result = compute_trip_duration(valid, TARGET)
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
    filtered = filter_valid_duration(df, TARGET)
    assert len(filtered) == 3
    assert float(filtered[TARGET].min()) >= min_trip_duration_minutes
    assert float(filtered[TARGET].max()) <= max_trip_duration_minutes


def test_filter_valid_passenger_count_drops_nan_and_invalid() -> None:
    df = pd.DataFrame({"passenger_count": [1.0, 2.0, 3.5, 0.0, 9.0, float("nan")]})
    out = filter_valid_passenger_count(df)
    assert set(out["passenger_count"].tolist()) == {1.0, 2.0}


def test_rename_columns() -> None:
    df = pd.DataFrame({old: [1] for old in rename_map})
    result = rename_columns(df)
    assert "pickup_location_id" in result.columns
    assert "dropoff_location_id" in result.columns


def test_select_and_clean_columns(raw_trips: pd.DataFrame, contract: DatasetContract) -> None:
    renamed = rename_columns(raw_trips)
    renamed[TARGET] = [10.0, 20.0, None, 30.0, 40.0]
    result = select_and_clean_columns(renamed, contract)
    assert TARGET in result.columns
    assert result[TARGET].notna().all()
    assert len(result) < len(renamed)
    assert set(result.columns) == set(contract.columns.keys())


def test_process_data_writes_ingest_lineage(
    raw_trips: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    processed_dir = tmp_path / "processed"
    lineage_dir = tmp_path / "lineage"

    raw_trips.to_parquet(raw_dir / "yellow_tripdata_2024-01.parquet")

    monkeypatch.setattr(process_config, "raw_dir", str(raw_dir))
    monkeypatch.setattr(process_config, "processed_dir", str(processed_dir))
    monkeypatch.setattr(lineage_records, "LINEAGE_DIR", lineage_dir)

    process_data("taxi")

    training_path = processed_dir / "training_data.parquet"
    assert training_path.exists()
    assert len(pd.read_parquet(training_path)) > 0

    record = LineageRecord.model_validate_json(
        (lineage_dir / "records" / "ingest_taxi.json").read_text(encoding="utf-8")
    )
    assert record.artifact == str(training_path)
    assert record.parents == ["dataset:taxi"]
