from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from broadway.config.loader import load_config
from broadway.config.schema import DatasetContract, PipelineConfig
from broadway.contracts.checks import check_columns, check_dtypes, check_nulls
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
def real_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 1000
    return pd.DataFrame(
        {
            "rooms": rng.integers(1, 7, n),
            "area": rng.integers(30, 200, n),
            "neighborhood": rng.choice(["A", "B", "C", "D"], n),
            "price": rng.integers(100, 1000, n),
        }
    )


def test_valid_dataframe_passes_all_checks(
    real_df: pd.DataFrame, contract: DatasetContract, null_threshold: float
) -> None:
    assert check_columns(real_df, contract) == []
    assert check_dtypes(real_df, contract) == []
    assert check_nulls(real_df, contract, null_threshold) == []


def test_missing_required_column_raises(
    real_df: pd.DataFrame, contract: DatasetContract, null_threshold: float
) -> None:
    df = real_df.drop(columns=["area"])
    issues = check_columns(df, contract)
    assert len(issues) > 0
    assert any("area" in issue for issue in issues)
    with pytest.raises(DataContractError):
        raise DataContractError("; ".join(issues))


def test_wrong_dtype_raises(
    real_df: pd.DataFrame, contract: DatasetContract, null_threshold: float
) -> None:
    df = real_df.copy()
    df["rooms"] = df["rooms"].astype("float64")
    issues = check_dtypes(df, contract)
    assert len(issues) > 0
    assert any("rooms" in issue for issue in issues)
    with pytest.raises(DataContractError):
        raise DataContractError("; ".join(issues))


def test_nulls_above_threshold_raises(
    real_df: pd.DataFrame, contract: DatasetContract, null_threshold: float
) -> None:
    df = real_df.copy()
    df.loc[df.sample(frac=0.1, random_state=42).index, "rooms"] = None
    issues = check_nulls(df, contract, null_threshold)
    assert len(issues) > 0
    assert any("rooms" in issue for issue in issues)
    with pytest.raises(DataContractError):
        raise DataContractError("; ".join(issues))
