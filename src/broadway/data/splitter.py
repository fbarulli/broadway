"""Time-based or stratified random train/test/val split, plus chronological helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from broadway.config.schema import DatasetContract, SplitConfig, TaskType


def _random_split(df: pd.DataFrame, val_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, val = train_test_split(df, test_size=val_size, random_state=random_state)
    return train, val


def _time_partition(
    df: pd.DataFrame, dt_col: str, fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological partition: the last ``fraction`` of rows form the second part."""
    df = df.sort_values(dt_col)
    cutoff = int(len(df) * (1 - fraction))
    return df.iloc[:cutoff], df.iloc[cutoff:]


def _stratified_split(
    df: pd.DataFrame, target: str, val_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, val = train_test_split(df, test_size=val_size, stratify=df[target], random_state=random_state)
    return train, val


def _split_with_fraction(
    df: pd.DataFrame,
    dataset: DatasetContract,
    split_cfg: SplitConfig,
    fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the split kind to ``df`` with ``fraction`` as the second part."""
    if split_cfg.type == "time":
        if not dataset.datetime_column:
            raise ValueError("time split requires a datetime_column in dataset config")
        return _time_partition(df, dataset.datetime_column, fraction)
    if split_cfg.type == "stratified":
        if dataset.task != TaskType.CLASSIFICATION:
            raise ValueError("stratified split requires a classification task")
        return _stratified_split(df, dataset.target, fraction, random_state)
    return _random_split(df, fraction, random_state)


def split(
    df: pd.DataFrame, dataset: DatasetContract, split_cfg: SplitConfig, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _split_with_fraction(
        df, dataset, split_cfg, split_cfg.validation_size, random_state
    )


def chronological_split(
    df: pd.DataFrame, datetime_column: str, test_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df chronologically so the last test_fraction of rows form the test set."""
    return _time_partition(df, datetime_column, test_fraction)


def calibration_split(
    df: pd.DataFrame,
    dataset: DatasetContract,
    split_cfg: SplitConfig,
    calibration_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into (fit, calibration); calibration is held out from the fit.

    ``time`` keeps chronology: the last ``calibration_size`` fraction is the
    calibration slice. ``stratified``/``random`` use the canonical split
    helpers with ``calibration_size`` as the second part.
    """
    return _split_with_fraction(df, dataset, split_cfg, calibration_size, random_state)
