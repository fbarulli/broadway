"""Column presence, dtype, and null-rate checks against the contract."""

from __future__ import annotations

import logging

import pandas as pd

from broadway.config.schema import DatasetContract

logger = logging.getLogger(__name__)


def check_columns(df: pd.DataFrame, contract: DatasetContract) -> list[str]:
    actual = set(df.columns)
    expected = set(contract.columns.keys())
    issues = []
    missing = expected - actual
    extra = actual - expected
    if missing:
        issues.append(f"missing columns: {sorted(missing)}")
    if extra:
        logger.warning("extra columns (allowed): %s", sorted(extra))
    if issues:
        logger.warning("column check failed — %s", "; ".join(issues))
    return issues


def check_nulls(df: pd.DataFrame, contract: DatasetContract, threshold: float) -> list[str]:
    issues = []
    for col in df.columns:
        null_rate = df[col].isna().mean()
        if null_rate > threshold:
            msg = f"{col}: null rate {null_rate:.1%} exceeds threshold {threshold:.1%}"
            issues.append(msg)
            logger.warning(msg)
    return issues


def check_dtypes(df: pd.DataFrame, contract: DatasetContract) -> list[str]:
    issues = []
    for col in set(df.columns) & set(contract.columns):
        expected = contract.columns[col].dtype
        actual = str(df[col].dtype)
        if expected != actual:
            msg = f"{col}: dtype mismatch — expected {expected}, got {actual}"
            issues.append(msg)
            logger.warning(msg)
    return issues
