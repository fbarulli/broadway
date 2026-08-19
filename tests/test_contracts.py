from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
def real_df() -> pd.DataFrame:
    """Generated data matching the test dataset contract (never real data)."""
    rng = np.random.default_rng(42)
    n = 1000
    return pd.DataFrame(
        {
            "feature_1": rng.integers(1, 7, n),
            "feature_2": rng.integers(30, 200, n),
            "feature_3": rng.choice(["A", "B", "C", "D"], n),
            "target": rng.integers(100, 1000, n),
        }
    )


def test_valid_dataframe_passes_all_checks(
    real_df: pd.DataFrame, contract: DatasetContract, null_threshold: float
) -> None:
    assert check_columns(real_df, contract) == []
    assert check_nulls(real_df, contract, null_threshold) == []


def test_missing_required_column_raises(
    real_df: pd.DataFrame, contract: DatasetContract, null_threshold: float
) -> None:
    df = real_df.drop(columns=["feature_2"])
    issues = check_columns(df, contract)
    assert len(issues) > 0
    assert any("feature_2" in issue for issue in issues)
    with pytest.raises(DataContractError):
        raise DataContractError("; ".join(issues))


def test_wrong_dtype_not_checked_at_raw_boundary(
    real_df: pd.DataFrame, contract: DatasetContract, null_threshold: float
) -> None:
    df = real_df.copy()
    df["feature_1"] = df["feature_1"].astype("float64")
    assert check_columns(df, contract) == []
    assert check_nulls(df, contract, null_threshold) == []


def test_nulls_above_threshold_raises(
    real_df: pd.DataFrame, contract: DatasetContract, null_threshold: float
) -> None:
    df = real_df.copy()
    df.loc[df.sample(frac=0.1, random_state=42).index, "feature_2"] = None
    issues = check_nulls(df, contract, null_threshold)
    assert len(issues) > 0
    assert any("feature_2" in issue for issue in issues)
    with pytest.raises(DataContractError):
        raise DataContractError("; ".join(issues))
