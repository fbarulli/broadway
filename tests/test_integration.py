from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from broadway.config.loader import load_config
from broadway.data.cleaner import clean
from broadway.data.splitter import split
from broadway.evaluate.metrics import compute_metrics
from broadway.training.trainer import train


@pytest.fixture
def tmp_dataset(tmp_path: Path) -> Path:
    rows = 100
    df = pd.DataFrame(
        {
            "feature_1": [1 + i % 6 for i in range(rows)],
            "feature_2": [30.0 + i * 1.5 for i in range(rows)],
            "target": [100.0 + i * 8.0 for i in range(rows)],
        }
    )
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
    model, result = train(cfg.experiment.model.type, X_train, y_train)
    assert result.train_time_seconds >= 0
    X_val = val_df.drop(columns=drop_cols)
    y_val = val_df[target]
    y_pred = model.predict(X_val)
    metrics = compute_metrics(y_val.values, y_pred)
    assert metrics["rmse"] >= 0 and metrics["r2"] >= -1 and metrics["mae"] >= 0
    assert metrics["rmse"] < 100, f"RMSE too high on synthetic data: {metrics['rmse']}"


def test_linear_model_coefficients(tmp_dataset: Path) -> None:
    cfg = _cfg()
    df, _ = clean(pd.read_parquet(tmp_dataset), cfg.dataset)
    target = cfg.dataset.target
    dt_col = cfg.dataset.datetime_column
    drop_cols = [target] + ([dt_col] if dt_col else [])
    X = df.drop(columns=drop_cols)
    y = df[target]
    model, _ = train(cfg.experiment.model.type, X, y)
    assert hasattr(model, "coef_")
    assert len(model.coef_) == len(X.columns)
