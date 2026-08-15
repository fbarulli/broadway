"""Unit tests for ``broadway.data.loader.read_sample`` against a synthetic parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from broadway.config.schema import ColumnRole, ColumnSchema, DatasetContract, TaskType
from broadway.data.loader import read_sample


@pytest.fixture
def synthetic_parquet(tmp_path: Path) -> Path:
    n = 1000
    df = pd.DataFrame({"x": range(n), "y": [float(i * 2) for i in range(n)]})
    path = tmp_path / "synthetic.parquet"
    df.to_parquet(path)
    return path


@pytest.fixture
def contract(synthetic_parquet: Path) -> DatasetContract:
    return DatasetContract(
        name="synthetic",
        path=str(synthetic_parquet),
        target="y",
        task=TaskType.REGRESSION,
        datetime_column=None,
        columns={
            "x": ColumnSchema(dtype="int64", null_count=0, role=ColumnRole.FEATURE),
            "y": ColumnSchema(dtype="float64", null_count=0, role=ColumnRole.TARGET),
        },
        lookup_tables={},
    )


def test_read_sample_respects_sample_size(contract: DatasetContract) -> None:
    df = read_sample(contract, sample=500, seed=42)
    assert len(df) == 500


def test_read_sample_seed_reproducible(contract: DatasetContract) -> None:
    a = read_sample(contract, sample=500, seed=42)
    b = read_sample(contract, sample=500, seed=42)
    c = read_sample(contract, sample=500, seed=7)
    assert a.equals(b)
    assert len(c) == 500


def test_read_sample_column_pruning(contract: DatasetContract) -> None:
    df = read_sample(contract, sample=500, seed=42, columns=["x"])
    assert list(df.columns) == ["x"]


def test_read_sample_requires_sample_or_full(contract: DatasetContract) -> None:
    with pytest.raises(ValueError):
        read_sample(contract)


def test_read_sample_full_returns_all(contract: DatasetContract) -> None:
    df = read_sample(contract, full=True)
    assert len(df) == 1000
