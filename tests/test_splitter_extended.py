"""Tests for the extended splitter helpers: chronological_split and stratified_sample."""

from __future__ import annotations

import pandas as pd

from broadway.data.splitter import chronological_split, stratified_sample


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


def test_stratified_sample_caps_rows_per_group() -> None:
    df = pd.DataFrame(
        {
            "group": ["a"] * 5 + ["b"] * 3 + ["c"] * 10,
            "value": list(range(18)),
        }
    )
    sample = stratified_sample(df, "group", per_group=3, random_state=7)
    counts = sample["group"].value_counts().to_dict()
    assert counts == {"a": 3, "b": 3, "c": 3}
    assert len(sample) == 9
    assert list(sample.columns) == ["group", "value"]


def test_stratified_sample_keeps_smaller_groups_intact() -> None:
    df = pd.DataFrame(
        {
            "group": ["a"] * 5 + ["b"] * 2,
            "value": list(range(7)),
        }
    )
    sample = stratified_sample(df, "group", per_group=4, random_state=3)
    counts = sample["group"].value_counts().to_dict()
    assert counts == {"a": 4, "b": 2}
    assert len(sample) == 6


def test_stratified_sample_is_reproducible_and_resets_index() -> None:
    df = pd.DataFrame(
        {
            "group": ["a"] * 5 + ["b"] * 3 + ["c"] * 10,
            "value": list(range(18)),
        }
    )
    first = stratified_sample(df, "group", per_group=2, random_state=11)
    second = stratified_sample(df, "group", per_group=2, random_state=11)
    pd.testing.assert_frame_equal(first, second)
    assert first.index.equals(pd.RangeIndex(len(first)))
