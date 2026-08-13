from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from broadway.config.loader import load_config
from broadway.eda.missing import null_counts, null_patterns
from broadway.eda.quality import constant_columns, duplicate_rows, outlier_counts_iqr
from broadway.eda.summary import summarize


@pytest.fixture
def eda_cfg():
    cfg = load_config("eda", dataset="test", experiment="baseline")
    assert cfg.eda is not None
    assert cfg.dataset is not None
    return cfg


@pytest.fixture
def test_sample():
    rng = np.random.default_rng(42)
    n = 1000
    df = pd.DataFrame(
        {
            "rooms": rng.integers(1, 7, n),
            "area": rng.integers(30, 200, n),
            "neighborhood": rng.choice(["A", "B", "C", "D"], n),
            "price": rng.integers(100, 1000, n),
        }
    )
    assert len(df) > 0
    return df


def test_summarize_returns_report(test_sample: pd.DataFrame) -> None:
    report = summarize(test_sample)

    assert isinstance(report, dict)
    assert report["row_count"] == 1000
    assert report["column_count"] == len(test_sample.columns)
    assert "columns" in report
    for col_name in test_sample.columns:
        assert col_name in report["columns"]
        col_info = report["columns"][col_name]
        assert "dtype" in col_info
        assert "null_count" in col_info
        assert "null_pct" in col_info
        assert "unique_count" in col_info


def test_constant_columns_none_in_varied_data(test_sample: pd.DataFrame) -> None:
    const = constant_columns(test_sample)
    assert isinstance(const, list)
    assert len(const) == 0


def test_duplicate_rows_reports_count(test_sample: pd.DataFrame) -> None:
    dup = duplicate_rows(test_sample)
    assert isinstance(dup, int)
    assert dup >= 0


def test_null_counts_per_column(test_sample: pd.DataFrame) -> None:
    counts = null_counts(test_sample)

    assert isinstance(counts, dict)
    for col_name in test_sample.columns:
        assert col_name in counts
        assert isinstance(counts[col_name], int)
        assert counts[col_name] >= 0


def test_null_patterns_valid_dataframe(test_sample: pd.DataFrame) -> None:
    patterns = null_patterns(test_sample)

    assert isinstance(patterns, pd.DataFrame)
    assert len(patterns) >= 1
    assert "count" in patterns.columns
    assert patterns["count"].sum() == len(test_sample)


def test_outlier_counts_iqr_on_real_data(
    test_sample: pd.DataFrame, eda_cfg
) -> None:
    multiplier = eda_cfg.eda.outlier_iqr_multiplier
    outliers = outlier_counts_iqr(test_sample, multiplier)

    assert isinstance(outliers, dict)
    numeric_cols = test_sample.select_dtypes(include="number").columns
    for col in numeric_cols:
        assert col in outliers
        assert isinstance(outliers[col], int)
        assert outliers[col] >= 0
