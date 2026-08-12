"""Unit tests for ``projects/taxi/data.py`` against a small synthetic parquet."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from projects.taxi import data

SMALL_BOROUGH = "Staten Island"
SMALL_FULL_COUNT = 300

_BOROUGH_LOCATION_IDS: dict[str, list[int]] = {
    "Staten Island": [1],
    "Manhattan": [10, 11, 12],
    "Brooklyn": [20],
    "Queens": [30],
    "Bronx": [40],
}

_BOROUGH_COUNTS: dict[str, int] = {
    "Staten Island": SMALL_FULL_COUNT,
    "Manhattan": 35_000,
    "Brooklyn": 15_000,
    "Queens": 20_000,
    "Bronx": 12_000,
}


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = sum(_BOROUGH_COUNTS.values())

    location_ids: list[int] = []
    for borough, count in _BOROUGH_COUNTS.items():
        location_ids.extend(rng.choice(_BOROUGH_LOCATION_IDS[borough], size=count))

    df = pd.DataFrame(
        {
            data.DATETIME_COL: pd.date_range("2023-12-30 00:00:00", periods=n, freq="90s"),
            data.PICKUP_LOCATION_COL: np.asarray(location_ids, dtype=np.int64),
            data.TRIP_DISTANCE_COL: rng.uniform(0.5, 25.0, size=n).astype("float64"),
            data.DURATION_COL: rng.uniform(1.0, 120.0, size=n).astype("float64"),
        }
    )
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


@pytest.fixture
def synthetic_parquet(tmp_path: Path, synthetic_df: pd.DataFrame) -> Path:
    path = tmp_path / "synthetic_training.parquet"
    synthetic_df.to_parquet(path)
    return path


@pytest.fixture
def zones_csv(tmp_path: Path) -> Path:
    rows = [
        {"LocationID": loc, "Borough": borough}
        for borough, locs in _BOROUGH_LOCATION_IDS.items()
        for loc in locs
    ]
    path = tmp_path / "taxi_zone_lookup.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def patched_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_parquet: Path,
    zones_csv: Path,
) -> None:
    monkeypatch.setattr(data, "DATA_PATH", str(synthetic_parquet))
    monkeypatch.setattr(data, "LOOKUP_PATH", str(zones_csv))
    monkeypatch.setattr(data, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(data, "QUALITY_REPORT", tmp_path / "quality_report.json")
    monkeypatch.delenv("DATA_MODE", raising=False)


def test_generate_sample_cache_keeps_small_groups_in_full(
    patched_data: None,
    synthetic_df: pd.DataFrame,
) -> None:
    data.generate_sample_cache(mode="dev")

    cache = pd.read_parquet(data._cache_path("dev"))
    counts = cache.groupby(data.PICKUP_BOROUGH_COL).size()

    assert counts[SMALL_BOROUGH] == SMALL_FULL_COUNT
    assert SMALL_FULL_COUNT <= len(cache) <= data._sample_size("dev") + SMALL_FULL_COUNT
    assert len(cache) < len(synthetic_df)


def test_load_time_slice_is_contiguous_and_sorted(
    patched_data: None,
    synthetic_df: pd.DataFrame,
) -> None:
    start, end = data._time_slice_bounds("dev")
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)

    result = data.load_time_slice(mode="dev")

    datetimes = result[data.DATETIME_COL]
    assert not result.empty
    assert datetimes.is_monotonic_increasing
    assert datetimes.is_unique
    assert ((datetimes >= lo) & (datetimes < hi)).all()

    expected = synthetic_df[
        (synthetic_df[data.DATETIME_COL] >= lo) & (synthetic_df[data.DATETIME_COL] < hi)
    ]
    assert len(result) == len(expected)
    assert len(result) < len(synthetic_df)


def test_load_stratified_sample_missing_cache_raises(patched_data: None) -> None:
    with pytest.raises(FileNotFoundError):
        data.load_stratified_sample()


def test_load_borough_durations_returns_all_groups(patched_data: None) -> None:
    data.generate_sample_cache(mode="dev")

    durations = data.load_borough_durations(mode="dev")

    assert set(durations.keys()) == set(data.BOROUGHS)
    for borough in data.BOROUGHS:
        assert isinstance(durations[borough], np.ndarray)
    assert durations[SMALL_BOROUGH].shape[0] == SMALL_FULL_COUNT


def test_read_training_data_downcasts_dtypes(patched_data: None) -> None:
    df = data.read_training_data()

    int_cols = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c])]
    float_cols = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
    assert int_cols
    assert float_cols
    assert all(df[c].dtype == "int32" for c in int_cols)
    assert all(df[c].dtype == "float32" for c in float_cols)


@pytest.mark.parametrize("mode", ["dev", "live"])
def test_resolve_mode_explicit(mode: str) -> None:
    assert data._resolve_mode(mode) == mode


def test_resolve_mode_defaults_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATA_MODE", raising=False)
    assert data._resolve_mode(None) == os.getenv("DATA_MODE", "dev")

    monkeypatch.setenv("DATA_MODE", "live")
    assert data._resolve_mode(None) == "live"


def test_mode_dependent_outputs_differ() -> None:
    assert data._sample_size("dev") != data._sample_size("live")
    assert data._cache_path("dev") != data._cache_path("live")
    assert data._params_hash("dev") != data._params_hash("live")
