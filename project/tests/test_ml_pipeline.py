"""Golden regression for the taxi ``FeaturePipeline`` (Slice 6).

The pipeline's internals were re-expressed over the platform encoding
transformers (``broadway.features.transformers``). The literals below were
captured from the pre-change merge-based implementation on synthetic data, so
they pin the frozen feature names, dtypes, and encoded values exactly.

Float goldens no longer use bare ``==``: the dual-numpy lock (uv.lock pins
numpy 2.3.5 on darwin/x86_64 and 2.5.2 everywhere else) makes ~16th-digit
results stack-dependent (FIXES.md golden-float row: one fresh-clone failure
observed). Float fields therefore tolerate up to 8 double-precision ULPs at
each golden value's magnitude; column names, dtypes, integers, and strings
stay exact-equality until a byte-freeze flag exists.
"""

from __future__ import annotations

import math
from unittest import mock

import pandas as pd
import pytest

from project.ml_pipeline import FeaturePipeline

# Float tolerance: 8 double-precision ULPs at the golden magnitude — clears
# the observed cross-numpy 16th-digit drift with headroom while still pinning
# every other digit.
_ULP_BUDGET = 8

GOLDEN_COLUMNS = [
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_month",
    "passenger_count",
    "trip_distance",
    "pickup_location_id",
    "dropoff_location_id",
    "is_weekend",
    "rush_hour",
    "is_night",
    "log_distance",
    "same_borough",
    "route_avg_duration",
    "route_frequency",
]

GOLDEN_DTYPES = {
    "pickup_hour": "int32",
    "pickup_day_of_week": "int32",
    "pickup_month": "int32",
    "passenger_count": "int64",
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

# route (pickup, dropoff) = (1, 2) seen twice -> smoothed 42.27272727272727;
# (1, 3) and (4, 4) seen once -> 44.285714285714285 / 45.714285714285715;
# (8, 8) unseen at fit -> global mean 45.0 and frequency fill -1.
GOLDEN_ROUTE_AVG_DURATION = [42.27272727272727, 45.0, 44.285714285714285, 45.714285714285715]
GOLDEN_ROUTE_FREQUENCY = [2, -1, 1, 1]

GOLDEN_ROWS = [
    {
        "pickup_hour": 10, "pickup_day_of_week": 4, "pickup_month": 1,
        "passenger_count": 1, "trip_distance": 1.0,
        "pickup_location_id": 1, "dropoff_location_id": 2,
        "is_weekend": 0, "rush_hour": 0, "is_night": 0,
        "log_distance": 0.6931471805599453, "same_borough": 0,
        "route_avg_duration": 42.27272727272727, "route_frequency": 2,
    },
    {
        "pickup_hour": 11, "pickup_day_of_week": 4, "pickup_month": 1,
        "passenger_count": 2, "trip_distance": 2.0,
        "pickup_location_id": 8, "dropoff_location_id": 8,
        "is_weekend": 0, "rush_hour": 0, "is_night": 0,
        "log_distance": 1.0986122886681096, "same_borough": 0,
        "route_avg_duration": 45.0, "route_frequency": -1,
    },
    {
        "pickup_hour": 12, "pickup_day_of_week": 4, "pickup_month": 1,
        "passenger_count": 1, "trip_distance": 1.5,
        "pickup_location_id": 1, "dropoff_location_id": 3,
        "is_weekend": 0, "rush_hour": 0, "is_night": 0,
        "log_distance": 0.9162907318741551, "same_borough": 0,
        "route_avg_duration": 44.285714285714285, "route_frequency": 1,
    },
    {
        "pickup_hour": 13, "pickup_day_of_week": 4, "pickup_month": 1,
        "passenger_count": 3, "trip_distance": 2.5,
        "pickup_location_id": 4, "dropoff_location_id": 4,
        "is_weekend": 0, "rush_hour": 0, "is_night": 0,
        "log_distance": 1.252762968495368, "same_borough": 1,
        "route_avg_duration": 45.714285714285715, "route_frequency": 1,
    },
]


def _train_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pickup_datetime": pd.to_datetime([
                "2024-01-01 10:00", "2024-01-01 11:00", "2024-01-02 08:00",
                "2024-01-02 09:00", "2024-01-03 12:00", "2024-01-03 13:00",
                "2024-01-04 14:00", "2024-01-04 15:00",
            ]),
            "passenger_count": [1, 1, 2, 2, 1, 3, 1, 2],
            "trip_distance": [1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 1.2, 5.0],
            "pickup_location_id": pd.Series([1, 1, 1, 2, 3, 4, 5, 6], dtype="int32"),
            "dropoff_location_id": pd.Series([2, 2, 3, 1, 4, 4, 5, 6], dtype="int32"),
            "trip_duration_minutes": [10, 20, 30, 40, 50, 60, 70, 80],
        }
    )


def _transform_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pickup_datetime": pd.to_datetime([
                "2024-01-05 10:00", "2024-01-05 11:00",
                "2024-01-05 12:00", "2024-01-05 13:00",
            ]),
            "passenger_count": [1, 2, 1, 3],
            "trip_distance": [1.0, 2.0, 1.5, 2.5],
            "pickup_location_id": pd.Series([1, 8, 1, 4], dtype="int32"),
            "dropoff_location_id": pd.Series([2, 8, 3, 4], dtype="int32"),
            "trip_duration_minutes": [15, 100, 30, 60],
        }
    )


def _write_zones(path) -> None:
    pd.DataFrame(
        {
            "LocationID": [1, 2, 3, 4, 5, 6, 7],
            "Borough": [
                "Manhattan", "Brooklyn", "Queens", "Bronx",
                "Staten Island", "Manhattan", "Queens",
            ],
        }
    ).to_csv(path, index=False)


def _make_pipeline(lookup_path: str) -> FeaturePipeline:
    return FeaturePipeline(
        lookup_path=lookup_path,
        encoding_smoothing=20,
        frequency_fill=-1.0,
        rush_hour_morning_start=7,
        rush_hour_morning_end=9,
        rush_hour_evening_start=16,
        rush_hour_evening_end=19,
        night_start=22,
        night_end=5,
    )


def _float_close(actual: float, golden: float) -> bool:
    """ULP-budgeted equality (see module docstring for the derivation)."""
    return math.isclose(
        actual, golden, rel_tol=0.0, abs_tol=_ULP_BUDGET * math.ulp(abs(golden))
    )


def _list_close(actual: list[float], golden: list[float]) -> bool:
    """Element-wise ULP-budget compare for a float column."""
    return len(actual) == len(golden) and all(
        _float_close(a, g) for a, g in zip(actual, golden)
    )


def _records_match(frame: pd.DataFrame, golden_rows: list[dict[str, object]]) -> bool:
    """Row-for-row golden compare: floats within the ULP budget, rest exact."""
    records = frame.to_dict("records")
    if len(records) != len(golden_rows):
        return False
    for row, golden in zip(records, golden_rows):
        if set(row) != set(golden):
            return False
        for key, expected in golden.items():
            value = row[key]
            if isinstance(expected, float) and not _float_close(value, expected):
                return False
            if not isinstance(expected, float) and value != expected:
                return False
    return True


def test_transform_matches_pre_change_golden(tmp_path) -> None:
    zones_path = tmp_path / "zones.csv"
    _write_zones(zones_path)
    pipeline = _make_pipeline(str(zones_path))
    out = pipeline.fit(_train_frame()).transform(_transform_frame())

    assert out.columns.tolist() == GOLDEN_COLUMNS
    assert {col: str(dtype) for col, dtype in out.dtypes.items()} == GOLDEN_DTYPES
    assert _list_close(out["route_avg_duration"].tolist(), GOLDEN_ROUTE_AVG_DURATION)
    assert out["route_frequency"].tolist() == GOLDEN_ROUTE_FREQUENCY
    assert _records_match(out, GOLDEN_ROWS)


def test_transform_preserves_row_count(tmp_path) -> None:
    zones_path = tmp_path / "zones.csv"
    _write_zones(zones_path)
    pipeline = _make_pipeline(str(zones_path)).fit(_train_frame())
    frame = _transform_frame()
    assert len(pipeline.transform(frame)) == len(frame)


def test_row_count_guard_stays_armed(tmp_path) -> None:
    zones_path = tmp_path / "zones.csv"
    _write_zones(zones_path)
    pipeline = _make_pipeline(str(zones_path)).fit(_train_frame())
    with (
        mock.patch.object(
            pipeline._route_target_encoder,
            "transform",
            side_effect=lambda df: df.iloc[:-1],
        ),
        pytest.raises(RuntimeError, match="changed row count"),
    ):
        pipeline.transform(_transform_frame())


def test_transform_requires_fit(tmp_path) -> None:
    zones_path = tmp_path / "zones.csv"
    _write_zones(zones_path)
    pipeline = _make_pipeline(str(zones_path))
    with pytest.raises(RuntimeError, match="must be fit"):
        pipeline.transform(_transform_frame())
