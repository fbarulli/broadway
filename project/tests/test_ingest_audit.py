"""Ingest TransformAudit (FX-A03a): the persisted record carries a balanced ledger."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

import broadway.lineage.records as lineage_records
from broadway.lineage.models import LineageRecord, TransformAudit
from project.etl import process_config
from project.etl.process import process_data

_BALANCED_DROPS = re.compile(r"-(\d+) rows$")


def _known_attrition_frame() -> pd.DataFrame:
    """8 rows: -2 trips filter, -1 duration, -1 passenger_count, -1 duplicate."""
    return pd.DataFrame(
        {
            "trip_distance": [2.5, 3.0, 55.0, 4.0, 1.0, 4.0, 5.0, 5.0],
            "tpep_pickup_datetime": pd.to_datetime(
                [
                    "2024-01-01 10:00:00", "2024-01-01 11:00:00", "2024-01-01 12:00:00",
                    "2023-12-31 10:00:00", "2024-01-01 13:00:00", "2024-01-01 14:00:00",
                    "2024-01-01 15:00:00", "2024-01-01 15:00:00",
                ]
            ),
            "tpep_dropoff_datetime": pd.to_datetime(
                [
                    "2024-01-01 10:20:00", "2024-01-01 12:00:00", "2024-01-01 12:30:00",
                    "2023-12-31 10:30:00", "2024-01-01 13:00:30", "2024-01-01 15:00:00",
                    "2024-01-01 17:00:00", "2024-01-01 17:00:00",
                ]
            ),
            "PULocationID": pd.Series([1, 2, 3, 4, 5, 6, 7, 7], dtype="int32"),
            "DOLocationID": pd.Series([1, 3, 2, 5, 4, 6, 8, 8], dtype="int32"),
            "passenger_count": [1.0, 2.0, 1.0, 3.0, float("nan"), 8.0, 3.0, 3.0],
            "fare_amount": [20.0, 15.0, 8.0, 30.0, 12.0, 18.0, 25.0, 25.0],
        }
    )


def _valid_frame(n: int) -> pd.DataFrame:
    """n pairwise-distinct rows that pass every filter stage untouched."""
    pickups = pd.to_datetime("2024-01-01 08:00") + pd.to_timedelta(range(n), unit="h")
    return pd.DataFrame(
        {
            "trip_distance": [2.0 + i for i in range(n)],
            "tpep_pickup_datetime": pickups,
            "tpep_dropoff_datetime": pickups + pd.Timedelta(minutes=30),
            "PULocationID": pd.Series(range(100, 100 + n), dtype="int32"),
            "DOLocationID": pd.Series(range(200, 200 + n), dtype="int32"),
            "passenger_count": [float((i % 6) + 1) for i in range(n)],
            "fare_amount": [10.0 + i for i in range(n)],
        }
    )


def _write_raw(tmp_path: Path, df: pd.DataFrame) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(raw_dir / "yellow_tripdata_2024-01.parquet")


@pytest.fixture
def ingest_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(process_config, "raw_dir", str(tmp_path / "raw"))
    monkeypatch.setattr(process_config, "processed_dir", str(tmp_path / "processed"))
    monkeypatch.setattr(lineage_records, "LINEAGE_DIR", tmp_path / "lineage")
    monkeypatch.delenv("CI", raising=False)
    return tmp_path


def _read_audit(tmp_path: Path) -> tuple[TransformAudit, pd.DataFrame]:
    record = LineageRecord.model_validate_json(
        (tmp_path / "lineage" / "records" / "ingest_taxi.json").read_text(encoding="utf-8")
    )
    processed = pd.read_parquet(tmp_path / "processed" / "training_data.parquet")
    assert record.audit is not None  # the audit rides the written record
    return record.audit, processed


def _assert_balanced(audit: TransformAudit) -> None:
    explained = sum(
        int(m.group(1)) for r in audit.reasons if (m := _BALANCED_DROPS.search(r))
    )
    assert audit.rows_in - explained == audit.rows_out
    assert audit.rows_dropped_total == audit.rows_in - audit.rows_out
    assert audit.rows_dropped_unexplained == 0


def test_ingest_record_carries_balanced_audit(ingest_env: Path) -> None:
    _write_raw(ingest_env, _known_attrition_frame())
    process_data("taxi")
    audit, processed = _read_audit(ingest_env)
    assert audit.rows_in == 8
    assert audit.rows_out == 3 == len(processed)
    assert audit.reasons == [
        "filter_valid_trips: -2 rows",
        "filter_valid_duration: -1 rows",
        "filter_valid_passenger_count: -1 rows",
        "duplicates: -1 rows",
    ]
    _assert_balanced(audit)
    assert audit.columns_added == [
        "dropoff_location_id", "pickup_datetime", "pickup_location_id", "trip_duration_minutes",
    ]
    assert set(audit.columns_removed) == {
        "tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID"
    }


def test_audit_zeros_on_zero_drop_run(ingest_env: Path) -> None:
    _write_raw(ingest_env, _valid_frame(2))
    process_data("taxi")
    audit, processed = _read_audit(ingest_env)
    assert audit.reasons == []
    assert audit.rows_in == audit.rows_out == len(processed) == 2
    assert audit.rows_dropped_total == 0
    assert audit.rows_dropped_unexplained == 0


def test_ci_sampling_reorder_becomes_visible_in_ledger(
    ingest_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_raw(ingest_env, _valid_frame(6))
    monkeypatch.setattr(process_config, "ci_sample_size", 3)
    monkeypatch.setenv("CI", "true")
    process_data("taxi")
    audit, processed = _read_audit(ingest_env)
    assert audit.reasons == ["ci_sample: -3 rows"]
    _assert_balanced(audit)
    assert audit.rows_out == len(processed) == 3


def test_nan_twins_account_at_dropna_through_full_pipeline(ingest_env: Path) -> None:
    frame = _known_attrition_frame()
    frame.loc[6, "fare_amount"] = float("nan")  # twins become dropna casualties,
    frame.loc[7, "fare_amount"] = float("nan")  # so dedup never fires on them
    _write_raw(ingest_env, frame)
    process_data("taxi")
    audit, _processed = _read_audit(ingest_env)
    joined = " ".join(audit.reasons)
    assert "dropna: -2 rows" in joined
    assert "duplicates:" not in joined
    _assert_balanced(audit)
