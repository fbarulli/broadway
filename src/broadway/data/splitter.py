"""Time-based or stratified random train/test/val split."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from broadway.config.schema import DatasetContract, SplitConfig


def _random_split(df: pd.DataFrame, val_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, val = train_test_split(df, test_size=val_size, random_state=random_state)
    return train, val


def _time_split(df: pd.DataFrame, dt_col: str, val_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(dt_col)
    cutoff = int(len(df) * (1 - val_size))
    return df.iloc[:cutoff], df.iloc[cutoff:]


def _stratified_split(
    df: pd.DataFrame, target: str, val_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, val = train_test_split(df, test_size=val_size, stratify=df[target], random_state=random_state)
    return train, val


def split(
    df: pd.DataFrame, dataset: DatasetContract, split_cfg: SplitConfig, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split_cfg.type == "time":
        if not dataset.datetime_column:
            raise ValueError("time split requires a datetime_column in dataset config")
        return _time_split(df, dataset.datetime_column, split_cfg.validation_size)
    if split_cfg.type == "stratified":
        return _stratified_split(df, dataset.target, split_cfg.validation_size, random_state)
    return _random_split(df, split_cfg.validation_size, random_state)
