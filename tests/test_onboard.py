from __future__ import annotations

import pandas as pd

from broadway.onboard.infer import infer
from broadway.onboard.models import InferenceReport


def test_infer_basic() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "value": [10.0, 20.0, 30.0, 40.0],
            "cat_int": [1, 2, 1, 2],
            "cat_obj": ["a", "b", "a", "b"],
            "ts": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "date_str": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        }
    )
    report = infer("test", df)
    assert isinstance(report, InferenceReport)
    assert report.row_count == 4
    assert report.columns["id"].suggested_role == "ignore"
    assert report.columns["ts"].datetime_candidate is True
    assert report.columns["ts"].suggested_role == "datetime"
    assert report.columns["date_str"].datetime_candidate is True
    assert report.columns["cat_int"].categorical is True
    assert report.columns["value"].categorical is False


def test_infer_date_string_detection() -> None:
    df = pd.DataFrame({"d": ["2024-01-01", "2024-01-02", "2024-01-03"]})
    report = infer("test", df)
    assert report.columns["d"].datetime_candidate is True


def test_infer_no_columns() -> None:
    report = infer("x", pd.DataFrame())
    assert isinstance(report, InferenceReport)
    assert report.columns == {}
    assert report.row_count == 0


def test_infer_non_date_string_not_datetime() -> None:
    df = pd.DataFrame({"s": ["a", "b", "c"]})
    report = infer("test", df)
    assert report.columns["s"].datetime_candidate is False
    assert report.columns["s"].categorical is True


def test_infer_mixed_format_datetime_no_parse_warning() -> None:
    import warnings

    df = pd.DataFrame(
        {
            "ts": [
                "2024-01-01",
                "2024-01-02 10:30:00",
                "2024-01-03",
                "2024-01-04 08:00:00",
            ]
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = infer("test", df)
    assert report.columns["ts"].datetime_candidate is True
    assert report.columns["ts"].suggested_role == "datetime"
    assert not any("format" in str(w.message) for w in caught)
