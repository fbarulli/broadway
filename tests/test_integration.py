from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from contract_fixture import make_contract_frame, numeric_feature_columns, target_column

from broadway.config.loader import load_config
from broadway.config.schema import PreprocessingStepConfig
from broadway.data.cleaner import clean
from broadway.data.splitter import split
from broadway.evaluate.metrics import compute_metrics
from broadway.training.trainer import train


@pytest.fixture
def tmp_dataset(tmp_path: Path) -> Path:
    cfg = load_config("full", dataset="test", experiment="baseline", analysis="test")
    assert cfg.dataset is not None
    cols = numeric_feature_columns(cfg.dataset) + [target_column(cfg.dataset)]
    df = make_contract_frame(cfg.dataset, n=100)[cols]
    f = tmp_path / "training_data.parquet"
    df.to_parquet(f)
    return f


def _cfg():
    return load_config("full", dataset="test", experiment="baseline", analysis="test")


def test_pipeline_on_synthetic_data(tmp_dataset: Path) -> None:
    cfg = _cfg()
    df, _ = clean(pd.read_parquet(tmp_dataset), cfg.dataset)
    target = cfg.dataset.target
    assert not df[target].isna().any()
    split_cfg = cfg.experiment.split
    train_df, val_df = split(df, cfg.dataset, split_cfg, random_state=cfg.experiment.random_state)
    assert len(train_df) > 0 and len(val_df) > 0
    assert len(train_df) + len(val_df) == len(df)
    dt_col = cfg.dataset.datetime_column
    drop_cols = [target] + ([dt_col] if dt_col else [])
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[target]
    model, result = train(cfg, X_train, y_train)
    assert result.train_time_seconds >= 0
    X_val = val_df.drop(columns=drop_cols)
    y_val = val_df[target]
    y_pred = model.predict(X_val)
    metrics = compute_metrics(y_val.values, y_pred)
    assert metrics["rmse"] >= 0 and metrics["r2"] >= -1 and metrics["mae"] >= 0
    assert metrics["rmse"] < 100, f"RMSE too high on synthetic data: {metrics['rmse']}"


def test_train_with_passthrough_recipe_identical_to_no_block(tmp_dataset: Path) -> None:
    """Slice 3 integration: a preprocessing block of passthrough steps must
    produce metrics identical to the no-block baseline — the Pipeline wrapper
    changes nothing numerically (passthrough identity end-to-end)."""
    cfg = _cfg()
    df, _ = clean(pd.read_parquet(tmp_dataset), cfg.dataset)
    target = cfg.dataset.target
    split_cfg = cfg.experiment.split
    train_df, val_df = split(df, cfg.dataset, split_cfg, random_state=cfg.experiment.random_state)
    dt_col = cfg.dataset.datetime_column
    drop_cols = [target] + ([dt_col] if dt_col else [])
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[target]
    X_val = val_df.drop(columns=drop_cols)
    y_val = val_df[target]

    def _metrics(cfg) -> dict:
        model, _ = train(cfg, X_train, y_train)
        return compute_metrics(y_val.values, model.predict(X_val))

    baseline_metrics = _metrics(cfg)
    recipe_cfg = cfg.model_copy(
        update={
            "experiment": cfg.experiment.model_copy(
                update={
                    "preprocessing": [
                        PreprocessingStepConfig(
                            type="passthrough", columns=list(X_train.columns)
                        )
                    ]
                }
            )
        }
    )
    assert _metrics(recipe_cfg) == baseline_metrics


def test_linear_model_coefficients(tmp_dataset: Path) -> None:
    cfg = _cfg()
    df, _ = clean(pd.read_parquet(tmp_dataset), cfg.dataset)
    target = cfg.dataset.target
    dt_col = cfg.dataset.datetime_column
    drop_cols = [target] + ([dt_col] if dt_col else [])
    X = df.drop(columns=drop_cols)
    y = df[target]
    model, _ = train(cfg, X, y)
    assert hasattr(model.named_steps["model"], "coef_")
    assert len(model.named_steps["model"].coef_) == len(X.columns)
