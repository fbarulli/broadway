"""
Data contracts: runtime checks that a DataFrame matches the schema
expected at a given pipeline stage. These are the enforcement layer for
the definitions in schema.py - schema.py says what SHOULD be true,
contracts.py checks that it IS true.

Call the relevant validate_* function at the boundary of each stage
(end of process_data(), end of the feature pipeline, etc.) so a schema
mismatch fails loudly at the point it's introduced, not downstream.
"""
from datetime import datetime

import pandas as pd

from broadway.features.schema import (
    RAW_FEATURES,
    RAW_FEATURE_TYPES,
    TARGET,
)


class DataContractError(Exception):
    """Raised when a DataFrame doesn't match its stage's expected schema."""


def _check_columns(df: pd.DataFrame, expected: list[str], stage: str) -> None:
    actual = set(df.columns)
    expected_set = set(expected)

    missing = expected_set - actual
    extra = actual - expected_set

    errors = []
    if missing:
        errors.append(f"missing columns: {sorted(missing)}")
    if extra:
        errors.append(f"unexpected extra columns: {sorted(extra)}")

    if errors:
        raise DataContractError(f"[{stage}] schema mismatch - {'; '.join(errors)}")


def _check_no_nulls(df: pd.DataFrame, stage: str) -> None:
    null_counts = df.isnull().sum()
    bad_cols = null_counts[null_counts > 0]
    if not bad_cols.empty:
        raise DataContractError(f"[{stage}] found nulls: {bad_cols.to_dict()}")


def validate_raw_schema(df: pd.DataFrame) -> None:
    """
    Contract for the output of etl/process.py.
    Must contain exactly RAW_FEATURES + TARGET, no nulls, correct dtypes.
    """
    stage = "raw_schema"
    expected = RAW_FEATURES + [TARGET]
    _check_columns(df, expected, stage)
    _check_no_nulls(df, stage)

    for col, expected_type in RAW_FEATURE_TYPES.items():
        if expected_type is float and not pd.api.types.is_float_dtype(df[col]):
            raise DataContractError(f"[{stage}] '{col}' expected float dtype, got {df[col].dtype}")
        if expected_type is int and not pd.api.types.is_integer_dtype(df[col]):
            raise DataContractError(f"[{stage}] '{col}' expected int dtype, got {df[col].dtype}")
        if issubclass(expected_type, datetime) and not pd.api.types.is_datetime64_any_dtype(df[col]):
            raise DataContractError(f"[{stage}] '{col}' expected datetime dtype, got {df[col].dtype}")
