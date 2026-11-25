"""Tests for the extended splitter helpers: chronological_split and calibration_split."""

from __future__ import annotations

import pandas as pd

from broadway.config.schema import (
    ColumnRole,
    ColumnSchema,
    DatasetContract,
    SplitConfig,
    TaskType,
)
from broadway.data.splitter import calibration_split, chronological_split


def test_chronological_split_returns_last_fraction_as_test() -> None:
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-06",
                    "2024-01-07",
                    "2024-01-08",
                    "2024-01-09",
                    "2024-01-10",
                ]
            ),
            "value": list(range(10)),
        }
    )
    train, test = chronological_split(df, "ts", test_fraction=0.3)
    assert len(train) == 7
    assert len(test) == 3
    assert len(train) + len(test) == len(df)
    assert list(test["value"]) == [7, 8, 9]


def test_chronological_split_sorts_and_keeps_dates_disjoint() -> None:
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                ["2024-03-05", "2024-01-05", "2024-02-15", "2024-01-01", "2024-03-01"]
            ),
            "value": [4, 1, 3, 0, 2],
        }
    )
    train, test = chronological_split(df, "ts", test_fraction=0.4)
    assert len(train) == 3
    assert len(test) == 2
    assert train["ts"].max() < test["ts"].min()
    assert set(test["ts"]) == {pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-05")}
    assert set(train["ts"]) == {pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-05"), pd.Timestamp("2024-02-15")}


def test_calibration_split_keeps_calibration_later_than_fit_in_time() -> None:
    dataset = DatasetContract(
        name="synthetic",
        path="synthetic.parquet",
        target="value",
        task=TaskType.REGRESSION,
        datetime_column="ts",
        columns={
            "ts": ColumnSchema(dtype="datetime64", null_count=0, role=ColumnRole.DATETIME),
            "value": ColumnSchema(dtype="float64", null_count=0, role=ColumnRole.TARGET),
        },
        lookup_tables={},
    )
    split_cfg = SplitConfig(type="time", validation_size=0.2)
    df = pd.DataFrame(
        {
            # Deliberately unsorted: the calibration split must order by time.
            "ts": pd.to_datetime(
                ["2024-01-05", "2024-01-02", "2024-01-08", "2024-01-01", "2024-01-03"]
            ),
            "value": [4, 1, 5, 0, 2],
        }
    )
    fit, calibration = calibration_split(df, dataset, split_cfg, 0.4, random_state=42)
    assert len(fit) == 3
    assert len(calibration) == 2
    assert fit["ts"].max() < calibration["ts"].min()
