"""Decision 5 tests: schema-derived feature columns replace the dtype selector.

The dtype-driven selector is retired: feature selection now resolves the
experiment's ``schema_contract`` to a declared column set and intersects it
with the frame, asserting every eligible column is numeric unless a
preprocessing step claims it. Synthetic contracts + generated frames only
(platform hygiene: no project coupling, no real data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from contract_fixture import make_contract_frame

from broadway.config.loader import load_config
from broadway.config.schema import (
    ColumnRole,
    ColumnSchema,
    DatasetContract,
    DataSourceRef,
    EnvironmentConfig,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    PipelineConfig,
    PreprocessingStepConfig,
    SplitConfig,
    TaskType,
)
from broadway.evaluate.metrics import compute_metrics
from broadway.training.trainer import train
from broadway.utils import eligible_feature_columns


def _environment() -> EnvironmentConfig:
    return EnvironmentConfig(
        log_level="INFO",
        data_dir="data",
        raw_subdir="raw",
        processed_subdir="processed",
        download_chunk_size=8192,
        mlflow_tracking_uri="http://localhost:5000",
        database_user="postgres",
        database_password="postgres",
        database_name="broadway",
        database_host="localhost",
        database_port=5432,
        sample_size_ci=1000,
        sample_size_stats=10000,
        api_replicas_min=1,
        api_replicas_max=3,
        api_hpa_cpu_threshold=80,
        monitoring_schedule="0 * * * *",
    )


def _synthetic_dataset(
    extra: dict[str, ColumnSchema] | None = None,
    datetime_column: str | None = None,
) -> DatasetContract:
    columns = {
        "rooms": ColumnSchema(dtype="int32", null_count=0, role=ColumnRole.FEATURE),
        "area": ColumnSchema(dtype="int64", null_count=0, role=ColumnRole.FEATURE),
        "price": ColumnSchema(dtype="float64", null_count=0, role=ColumnRole.TARGET),
    }
    if extra:
        columns.update(extra)
    return DatasetContract(
        name="synthetic",
        path="synthetic.parquet",
        target="price",
        task=TaskType.REGRESSION,
        datetime_column=datetime_column,
        columns=columns,
        lookup_tables={},
    )


def _make_cfg(
    dataset: DatasetContract,
    include: list[str],
    preprocessing: list[PreprocessingStepConfig] | None = None,
) -> PipelineConfig:
    experiment = ExperimentConfig(
        data_source=DataSourceRef(loader="canonical", schema_contract="raw"),
        features=FeatureConfig(include=include, exclude=[], derived=[], encodings=[]),
        model=ModelConfig(type="linear", params={}),
        split=SplitConfig(type="random", validation_size=0.2),
        random_state=42,
        target_metric="rmse",
        preprocessing=preprocessing or [],
    )
    return PipelineConfig(environment=_environment(), dataset=dataset, experiment=experiment)


def test_schema_derived_matches_retired_selector() -> None:
    """All-numeric frame: the schema-derived column set equals the retired
    dtype selector's output, and the fitted metrics are identical."""
    dataset = _synthetic_dataset()
    cfg = _make_cfg(dataset, include=["rooms", "area"])
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "rooms": rng.integers(1, 6, 40).astype("int32"),
            "area": rng.integers(20, 80, 40),
            "price": rng.normal(100.0, 10.0, 40),
        }
    )
    y = df["price"]
    retired = list(df.select_dtypes(include="number").drop(columns=["price"]).columns)
    assert eligible_feature_columns(df, cfg) == retired
    X_new = df[eligible_feature_columns(df, cfg)]
    X_old = df[retired]
    pd.testing.assert_frame_equal(X_new, X_old)
    metrics_new = compute_metrics(y.to_numpy(), train(cfg, X_new, y)[0].predict(X_new))
    metrics_old = compute_metrics(y.to_numpy(), train(cfg, X_old, y)[0].predict(X_old))
    assert metrics_new == metrics_old


def test_assertion_names_unclaimed_categorical() -> None:
    """An object column in the schema with no claiming preprocessing step
    fails loud — the retired selector silently dropped it instead."""
    dataset = _synthetic_dataset(
        {"hood": ColumnSchema(dtype="object", null_count=0, role=ColumnRole.FEATURE)}
    )
    cfg = _make_cfg(dataset, include=["rooms", "area", "hood"])
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "rooms": rng.integers(1, 6, 20).astype("int32"),
            "area": rng.integers(20, 80, 20),
            "hood": rng.choice(["north", "south"], 20),
            "price": rng.normal(100.0, 10.0, 20),
        }
    )
    with pytest.raises(
        ValueError,
        match="categorical column 'hood' has no preprocessing step and no "
        "numeric-selector fallback applies",
    ):
        eligible_feature_columns(df, cfg)


def test_recipe_claimed_categorical_bypasses() -> None:
    """A preprocessing step claiming the categorical column keeps it eligible."""
    dataset = _synthetic_dataset(
        {"hood": ColumnSchema(dtype="object", null_count=0, role=ColumnRole.FEATURE)}
    )
    cfg = _make_cfg(
        dataset,
        include=["rooms", "area", "hood"],
        preprocessing=[
            PreprocessingStepConfig(
                type="target_encoding", columns=["hood"], params={"smoothing": 20}
            )
        ],
    )
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "rooms": rng.integers(1, 6, 20).astype("int32"),
            "area": rng.integers(20, 80, 20),
            "hood": rng.choice(["north", "south"], 20),
            "price": rng.normal(100.0, 10.0, 20),
        }
    )
    assert "hood" in eligible_feature_columns(df, cfg)


def test_numeric_category_semantics() -> None:
    """int32/int64/float64 all pass; a datetime64 column is not numeric and
    fails the same unclaimed-categorical assertion."""
    dataset = _synthetic_dataset()
    cfg = _make_cfg(dataset, include=["rooms", "area"])
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "rooms": rng.integers(1, 6, 20).astype("int32"),
            "area": rng.integers(20, 80, 20),
            "price": rng.normal(100.0, 10.0, 20),
        }
    )
    assert set(eligible_feature_columns(df, cfg)) == {"rooms", "area"}

    dt_dataset = _synthetic_dataset(
        {"ts": ColumnSchema(dtype="datetime64", null_count=0, role=ColumnRole.DATETIME)},
        datetime_column="ts",
    )
    dt_cfg = _make_cfg(dt_dataset, include=["rooms", "area"])
    df["ts"] = pd.to_datetime(rng.integers(0, 1_700_000_000, 20), unit="s")
    with pytest.raises(ValueError, match="categorical column 'ts'"):
        eligible_feature_columns(df, dt_cfg)


def test_shipped_experiments_pass_assertion() -> None:
    """Each repointed shipped config accepts a generated test-contract frame
    (plus its post-engineering columns) and selects exactly what the retired
    dtype selector would have."""
    shipped = {
        "baseline": {},
        "engineered": {"engineered_feature_1": "float64", "feature_3_target_enc": "float64"},
        "hyperopt": {"feature_3_target_enc": "float64"},
    }
    for name, extra_cols in shipped.items():
        cfg = load_config("full", dataset="test", experiment=name, analysis="test")
        assert cfg.dataset is not None
        df = make_contract_frame(cfg.dataset, n=50)
        for col, dtype in extra_cols.items():
            df[col] = np.zeros(len(df), dtype=dtype)
        cols = eligible_feature_columns(df, cfg)
        retired = list(
            df.select_dtypes(include="number").drop(columns=[cfg.dataset.target]).columns
        )
        assert cols == retired
