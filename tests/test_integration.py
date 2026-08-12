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
            "trip_distance": [1.0 + i * 0.3 for i in range(rows)],
            "trip_duration_minutes": [5.0 + i * 0.2 for i in range(rows)],
            "passenger_count": [1.0] * rows,
            "pickup_location_id": [i % 10 for i in range(rows)],
            "dropoff_location_id": [i % 8 for i in range(rows)],
            "pickup_datetime": pd.date_range("2024-01-01", periods=rows, freq="h"),
        }
    )
    f = tmp_path / "training_data.parquet"
    df.to_parquet(f)
    return f


def test_pipeline_on_synthetic_data(tmp_dataset: Path) -> None:
    cfg = load_config("full", dataset="taxi", experiment="taxi")
    df = clean(pd.read_parquet(tmp_dataset), cfg.dataset)
    target = cfg.dataset.target
    assert not df[target].isna().any()
    split_cfg = cfg.experiment.split
    train_df, val_df = split(df, cfg.dataset, split_cfg, random_state=cfg.experiment.random_state)
    assert len(train_df) > 0 and len(val_df) > 0
    assert len(train_df) + len(val_df) == len(df)
    dt_col = cfg.dataset.datetime_column
    X_train = train_df.drop(columns=[target, dt_col])
    y_train = train_df[target]
    model, elapsed = train(cfg.experiment.model.type, X_train, y_train)
    assert elapsed > 0
    X_val = val_df.drop(columns=[target, dt_col])
    y_val = val_df[target]
    y_pred = model.predict(X_val)
    metrics = compute_metrics(y_val.values, y_pred)
    assert metrics["rmse"] >= 0 and metrics["r2"] >= -1 and metrics["mae"] >= 0
    assert metrics["rmse"] < 10, f"RMSE too high on synthetic data: {metrics['rmse']}"


def test_linear_model_coefficients(tmp_dataset: Path) -> None:
    cfg = load_config("full", dataset="taxi", experiment="taxi")
    df = clean(pd.read_parquet(tmp_dataset), cfg.dataset)
    target = cfg.dataset.target
    dt_col = cfg.dataset.datetime_column
    X = df.drop(columns=[target, dt_col])
    y = df[target]
    model, _ = train(cfg.experiment.model.type, X, y)
    assert hasattr(model, "coef_")
    assert len(model.coef_) == len(X.columns)
