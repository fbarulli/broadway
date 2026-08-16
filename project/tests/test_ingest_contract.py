from __future__ import annotations

import pandera.errors
import pandas as pd
import pytest

from broadway.config.schema import ColumnRole, ColumnSchema, DatasetContract
from project.etl.process import (
    select_and_clean_columns,
    validate_contract_schema,
)


def _contract(columns: dict[str, ColumnSchema]) -> DatasetContract:
    return DatasetContract(
        name="toy",
        path="toy.parquet",
        target="target",
        task="regression",
        datetime_column=None,
        columns=columns,
        lookup_tables={},
    )


def _feature(dtype: str) -> ColumnSchema:
    return ColumnSchema(dtype=dtype, null_count=0, role=ColumnRole.FEATURE)


def test_select_and_clean_columns_keeps_exactly_contract_columns() -> None:
    contract = _contract({"a": _feature("float64"), "target": _feature("float64")})
    df = pd.DataFrame({"a": [1.0, 2.0], "target": [3.0, 4.0], "extra": [5, 6]})
    result = select_and_clean_columns(df, contract)
    assert list(result.columns) == ["a", "target"]


def test_extra_column_not_in_contract_is_dropped_not_error() -> None:
    contract = _contract({"a": _feature("float64"), "target": _feature("float64")})
    df = pd.DataFrame({"a": [1.0, 2.0], "target": [3.0, 4.0], "extra": [5, 6]})
    result = select_and_clean_columns(df, contract)
    assert "extra" not in result.columns


def test_missing_contract_column_raises() -> None:
    contract = _contract({"a": _feature("float64"), "target": _feature("float64")})
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="target"):
        select_and_clean_columns(df, contract)


@pytest.mark.parametrize("dtype", ["", "   "])
def test_missing_dtype_raises(dtype: str) -> None:
    contract = _contract({"a": _feature(dtype)})
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="missing a dtype"):
        validate_contract_schema(df, contract)


def test_dtype_mismatch_raises() -> None:
    contract = _contract({"a": _feature("float64")})
    df = pd.DataFrame({"a": pd.Series([1, 2, 3], dtype="int64")})
    with pytest.raises(pandera.errors.SchemaError):
        validate_contract_schema(df, contract)
