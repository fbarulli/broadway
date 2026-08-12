from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from broadway.config.loader import load_config
from broadway.eda.missing import null_counts, null_patterns
from broadway.eda.quality import constant_columns, duplicate_rows, outlier_counts_iqr
from broadway.eda.summary import summarize


@pytest.fixture
def eda_cfg():
    cfg = load_config("eda", dataset="taxi", experiment="taxi")
    assert cfg.eda is not None
    assert cfg.dataset is not None
    return cfg


@pytest.fixture
def taxi_sample():
    df = pd.read_parquet("data/processed/training_data.parquet").head(1000)
    assert len(df) > 0
    return df


def test_summarize_returns_report(taxi_sample: pd.DataFrame) -> None:
    report = summarize(taxi_sample)

    assert isinstance(report, dict)
    assert report["row_count"] == 1000
    assert report["column_count"] == len(taxi_sample.columns)
    assert "columns" in report
    for col_name in taxi_sample.columns:
        assert col_name in report["columns"]
        col_info = report["columns"][col_name]
        assert "dtype" in col_info
        assert "null_count" in col_info
        assert "null_pct" in col_info
        assert "unique_count" in col_info


def test_constant_columns_none_in_varied_data(taxi_sample: pd.DataFrame) -> None:
    const = constant_columns(taxi_sample)
    assert isinstance(const, list)
    assert len(const) == 0


def test_duplicate_rows_reports_count(taxi_sample: pd.DataFrame) -> None:
    dup = duplicate_rows(taxi_sample)
    assert isinstance(dup, int)
    assert dup >= 0


def test_null_counts_per_column(taxi_sample: pd.DataFrame) -> None:
    counts = null_counts(taxi_sample)

    assert isinstance(counts, dict)
    for col_name in taxi_sample.columns:
        assert col_name in counts
        assert isinstance(counts[col_name], int)
        assert counts[col_name] >= 0


def test_null_patterns_valid_dataframe(taxi_sample: pd.DataFrame) -> None:
    patterns = null_patterns(taxi_sample)

    assert isinstance(patterns, pd.DataFrame)
    assert len(patterns) >= 1
    assert "count" in patterns.columns
    assert patterns["count"].sum() == len(taxi_sample)


def test_outlier_counts_iqr_on_real_data(
    taxi_sample: pd.DataFrame, eda_cfg
) -> None:
    multiplier = eda_cfg.eda.outlier_iqr_multiplier
    outliers = outlier_counts_iqr(taxi_sample, multiplier)

    assert isinstance(outliers, dict)
    numeric_cols = taxi_sample.select_dtypes(include="number").columns
    for col in numeric_cols:
        assert col in outliers
        assert isinstance(outliers[col], int)
        assert outliers[col] >= 0
