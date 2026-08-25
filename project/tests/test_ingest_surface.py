"""Ingest surface: FX-A02 dedup inside select_and_clean_columns, with ledger accounting."""

from __future__ import annotations

import pandas as pd

from broadway.config.schema import ColumnRole, ColumnSchema, DatasetContract, TaskType
from project.etl.process import select_and_clean_columns


def _contract() -> DatasetContract:
    return DatasetContract(
        name="dedup-probe",
        path="unused.parquet",
        target="y",
        task=TaskType.REGRESSION,
        datetime_column=None,
        columns={
            "x": ColumnSchema(dtype="float64", null_count=0, role=ColumnRole.FEATURE),
            "y": ColumnSchema(dtype="float64", null_count=0, role=ColumnRole.FEATURE),
        },
        lookup_tables={},
    )


def test_select_and_clean_columns_dedups() -> None:
    contract = _contract()
    df = pd.DataFrame({"x": [1.0, 1.0], "y": [2.0, 2.0], "junk": ["a", "b"]})
    ledger: list[tuple[str, int]] = []
    out = select_and_clean_columns(df, contract, ledger)
    assert len(out) == 1  # two identical post-clean rows collapse to one
    assert ledger == [("dropna", 2), ("duplicates", 1)]  # accounting shows -1


def test_select_and_clean_columns_nan_twins_die_at_dropna_not_dedup() -> None:
    contract = _contract()
    df = pd.DataFrame(
        {"x": [1.0, 1.0, 3.0], "y": [float("nan"), float("nan"), 4.0]}
    )
    ledger: list[tuple[str, int]] = []
    out = select_and_clean_columns(df, contract, ledger)
    assert len(out) == 1
    assert ledger == [("dropna", 1), ("duplicates", 1)]


def test_select_and_clean_columns_zero_duplicates_keeps_counts_equal() -> None:
    contract = _contract()
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    ledger: list[tuple[str, int]] = []
    out = select_and_clean_columns(df, contract, ledger)
    assert len(out) == 2
    assert ledger == [("dropna", 2), ("duplicates", 2)]
