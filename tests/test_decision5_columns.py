"""Decision 5 column-selection tests — synthetic data only.

Pins the schema-derived eligible-column boundary against the retired
dtype-driven selector: identical output (frame order included), the exact
fail-loud message for an unclaimed categorical, the recipe-explicit bypass,
and the numeric-category semantics over observed runtime dtypes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.linear_model import LinearRegression

from broadway.config.loader import load_config
from broadway.config.schema import (
    FeatureConfig,
    PreprocessingStepConfig,
    PipelineConfig,
)
from broadway.evaluate.metrics import compute_metrics
from broadway.schemas import schema_columns
from broadway.utils import eligible_feature_columns


def _retired_selector(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """The retired dtype-driven selection, inlined (it no longer exists)."""
    return df.select_dtypes(include="number").drop(columns=[target], errors="ignore")


def _cfg(
    schema_contract: str = "engineered",
    include: list[str] | None = None,
    preprocessing: list[dict[str, object]] | None = None,
) -> PipelineConfig:
    cfg = load_config("train", dataset="test", experiment="baseline")
    updates: dict[str, object] = {
        "data_source": cfg.experiment.data_source.model_copy(
            update={"schema_contract": schema_contract}
        ),
        "preprocessing": [
            PreprocessingStepConfig(**step) for step in (preprocessing or [])
        ],
    }
    if include is not None:
        updates["features"] = FeatureConfig(
            include=include, exclude=[], derived=[], encodings=[]
        )
    return cfg.model_copy(update={"experiment": cfg.experiment.model_copy(update=updates)})


def _mixed_frame(n: int = 40, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "feature_1": rng.integers(0, 100, n).astype("int32"),
            "feature_2": rng.integers(0, 100, n) * 1.5,
            "extra_num": rng.integers(0, 50, n),
            "feature_3": rng.choice(["A", "B"], n),
            "event_at": pd.date_range("2024-01-01", periods=n, freq="h"),
            "target": rng.integers(0, 100, n),
        }
    )


def _equality_frame(n: int = 40, seed: int = 42) -> pd.DataFrame:
    """Frame whose numeric columns are ALL declared — schema-derived and the
    retired selector then see identical columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "feature_1": rng.integers(0, 100, n).astype("int32"),
            "feature_2": rng.integers(0, 100, n) * 1.5,
            "feature_3": rng.choice(["A", "B"], n),
            "event_at": pd.date_range("2024-01-01", periods=n, freq="h"),
            "target": rng.integers(0, 100, n),
        }
    )


def _mixed_frame(n: int = 40, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "feature_1": rng.integers(0, 100, n).astype("int32"),
            "feature_2": rng.integers(0, 100, n) * 1.5,
            "extra_num": rng.integers(0, 50, n),
            "feature_3": rng.choice(["A", "B"], n),
            "event_at": pd.date_range("2024-01-01", periods=n, freq="h"),
            "target": rng.integers(0, 100, n),
        }
    )


def test_schema_derived_matches_retired_selector_including_frame_order() -> None:
    df = _equality_frame()
    cfg = _cfg(include=["feature_1", "feature_2"])
    eligible = eligible_feature_columns(df, cfg)
    assert_frame_equal(eligible, _retired_selector(df, "target"))


def test_golden_metric_equality_with_retired_selection() -> None:
    df = _equality_frame()
    cfg = _cfg(include=["feature_1", "feature_2"])
    X_new = eligible_feature_columns(df, cfg)
    X_old = _retired_selector(df, "target")
    y = df["target"]
    metrics_old = compute_metrics(y.to_numpy(), LinearRegression().fit(X_old, y).predict(X_old))
    metrics_new = compute_metrics(y.to_numpy(), LinearRegression().fit(X_new, y).predict(X_new))
    assert metrics_new == metrics_old


def test_undeclared_numeric_column_is_excluded() -> None:
    """The deliberate difference from the retired dtype selector: a numeric
    runtime column OUTSIDE the declared surface does not leak into X."""
    df = _mixed_frame()
    cfg = _cfg(include=["feature_1", "feature_2"])
    eligible = eligible_feature_columns(df, cfg)
    assert list(eligible.columns) == ["feature_1", "feature_2"]


def test_unclaimed_categorical_raises_exact_message() -> None:
    df = _mixed_frame()
    cfg = _cfg(include=["feature_1", "feature_2", "feature_3"])
    with pytest.raises(ValueError) as exc_info:
        eligible_feature_columns(df, cfg)
    assert str(exc_info.value) == (
        "categorical column 'feature_3' has no preprocessing step and no "
        "numeric-selector fallback applies — add a preprocessing step claiming "
        "it, or repoint the schema contract so it is not eligible"
    )


def test_recipe_explicit_column_bypasses_assertion() -> None:
    df = _mixed_frame()
    cfg = _cfg(
        include=["feature_1", "feature_2", "feature_3"],
        preprocessing=[
            {"type": "target_encoding", "columns": ["feature_3"], "params": {"smoothing": 20}}
        ],
    )
    eligible = eligible_feature_columns(df, cfg)
    assert "feature_3" in eligible.columns


@pytest.mark.parametrize("dtype", ["int32", "int64", "float64"])
def test_numeric_category_dtypes_pass_unclaimed(dtype: str) -> None:
    df = pd.DataFrame({"feature_1": np.arange(4).astype(dtype), "target": [1, 2, 3, 4]})
    eligible = eligible_feature_columns(df, _cfg(include=["feature_1"]))
    assert list(eligible.columns) == ["feature_1"]


def test_datetime64_fires_the_assertion() -> None:
    df = pd.DataFrame(
        {
            "feature_1": pd.date_range("2024-01-01", periods=4, freq="h"),
            "target": [1, 2, 3, 4],
        }
    )
    with pytest.raises(ValueError, match="categorical column 'feature_1'"):
        eligible_feature_columns(df, _cfg(include=["feature_1"]))


def test_engineered_requires_features_config() -> None:
    cfg = load_config("train", dataset="test", experiment="baseline")
    assert cfg.dataset is not None
    with pytest.raises(ValueError, match="'engineered' requires"):
        schema_columns("engineered", cfg.dataset)


def test_raw_and_joined_ignore_features_argument() -> None:
    cfg = load_config("train", dataset="test", experiment="baseline")
    assert cfg.dataset is not None
    features = FeatureConfig(include=["feature_1"], exclude=[], derived=[], encodings=[])
    assert schema_columns("raw", cfg.dataset, features=features) == frozenset(
        cfg.dataset.columns
    )
