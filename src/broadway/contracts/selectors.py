"""Role-based column selectors over a :class:`DatasetContract`."""

from __future__ import annotations

from broadway.config.schema import ColumnRole, DatasetContract
from broadway.contracts.pandera import is_numeric_dtype


def feature_columns(contract: DatasetContract) -> list[str]:
    return [
        name for name, col in contract.columns.items() if col.role == ColumnRole.FEATURE
    ]


def datetime_columns(contract: DatasetContract) -> list[str]:
    return [
        name for name, col in contract.columns.items() if col.role == ColumnRole.DATETIME
    ]


def target_columns(contract: DatasetContract) -> list[str]:
    return [
        name for name, col in contract.columns.items() if col.role == ColumnRole.TARGET
    ]


def numeric_columns(contract: DatasetContract) -> list[str]:
    return [name for name, col in contract.columns.items() if is_numeric_dtype(col.dtype)]
