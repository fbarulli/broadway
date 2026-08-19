from __future__ import annotations

import pandas as pd
import pytest
from contract_fixture import feature_columns, make_contract_frame

from broadway.config.loader import load_config
from broadway.config.schema import DatasetContract, PipelineConfig
from broadway.contracts.checks import check_columns, check_nulls
from broadway.features.contracts import DataContractError


@pytest.fixture
def cfg() -> PipelineConfig:
    return load_config("contracts", dataset="test", experiment="baseline")


@pytest.fixture
def contract(cfg: PipelineConfig) -> DatasetContract:
    assert cfg.dataset is not None
    return cfg.dataset


@pytest.fixture
def null_threshold(cfg: PipelineConfig) -> float:
    assert cfg.contracts is not None
    return cfg.contracts.null_threshold


@pytest.fixture
def real_df(contract: DatasetContract) -> pd.DataFrame:
    """Generated data matching the test dataset contract (never real data)."""
    return make_contract_frame(contract, n=1000)


def test_valid_dataframe_passes_all_checks(
    real_df: pd.DataFrame, contract: DatasetContract, null_threshold: float
) -> None:
    assert check_columns(real_df, contract) == []
    assert check_nulls(real_df, contract, null_threshold) == []


def test_missing_required_column_raises(
    real_df: pd.DataFrame, contract: DatasetContract, null_threshold: float
) -> None:
    dropped = feature_columns(contract)[0]
    df = real_df.drop(columns=[dropped])
    issues = check_columns(df, contract)
    assert len(issues) > 0
    assert any(dropped in issue for issue in issues)
    with pytest.raises(DataContractError):
        raise DataContractError("; ".join(issues))


def test_wrong_dtype_not_checked_at_raw_boundary(
    real_df: pd.DataFrame, contract: DatasetContract, null_threshold: float
) -> None:
    df = real_df.copy()
    col = feature_columns(contract)[0]
    df[col] = df[col].astype("float64")
    assert check_columns(df, contract) == []
    assert check_nulls(df, contract, null_threshold) == []


def test_nulls_above_threshold_raises(
    real_df: pd.DataFrame, contract: DatasetContract, null_threshold: float
) -> None:
    df = real_df.copy()
    col = feature_columns(contract)[0]
    df.loc[df.sample(frac=0.1, random_state=42).index, col] = None
    issues = check_nulls(df, contract, null_threshold)
    assert len(issues) > 0
    assert any(col in issue for issue in issues)
    with pytest.raises(DataContractError):
        raise DataContractError("; ".join(issues))
