from __future__ import annotations

import pandas as pd
import pytest

from broadway.config.loader import load_config
from broadway.config.schema import DatasetContract, PipelineConfig
from broadway.contracts.checks import check_columns, check_nulls
from broadway.features.contracts import DataContractError


@pytest.fixture
def cfg() -> PipelineConfig:
    return load_config("contracts", dataset="taxi", experiment="taxi")


@pytest.fixture
def contract(cfg: PipelineConfig) -> DatasetContract:
    assert cfg.dataset is not None
    return cfg.dataset


@pytest.fixture
def null_threshold(cfg: PipelineConfig) -> float:
    assert cfg.contracts is not None
    return cfg.contracts.null_threshold


@pytest.fixture
def real_df() -> pd.DataFrame:
    return pd.read_parquet("data/processed/training_data.parquet").head(1000)


def test_valid_dataframe_passes_all_checks(
    real_df: pd.DataFrame, contract: DatasetContract, null_threshold: float
) -> None:
    assert check_columns(real_df, contract) == []
    assert check_nulls(real_df, contract, null_threshold) == []


def test_missing_required_column_raises(
    real_df: pd.DataFrame, contract: DatasetContract, null_threshold: float
) -> None:
    df = real_df.drop(columns=["trip_distance"])
    issues = check_columns(df, contract)
    assert len(issues) > 0
    assert any("trip_distance" in issue for issue in issues)
    with pytest.raises(DataContractError):
        raise DataContractError("; ".join(issues))


def test_wrong_dtype_not_checked_at_raw_boundary(
    real_df: pd.DataFrame, contract: DatasetContract, null_threshold: float
) -> None:
    df = real_df.copy()
    df["pickup_location_id"] = df["pickup_location_id"].astype("float64")
    assert check_columns(df, contract) == []
    assert check_nulls(df, contract, null_threshold) == []


def test_nulls_above_threshold_raises(
    real_df: pd.DataFrame, contract: DatasetContract, null_threshold: float
) -> None:
    df = real_df.copy()
    df.loc[df.sample(frac=0.1, random_state=42).index, "trip_distance"] = None
    issues = check_nulls(df, contract, null_threshold)
    assert len(issues) > 0
    assert any("trip_distance" in issue for issue in issues)
    with pytest.raises(DataContractError):
        raise DataContractError("; ".join(issues))
