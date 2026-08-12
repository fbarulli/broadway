"""End-to-end lifecycle test for the contract-driven flow.

Exercises, on a tiny synthetic dataset, the full chain:
contract validation -> feature creation -> training -> evaluation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandera.errors
import pytest
import yaml

from broadway.config.loader import load_config
from broadway.config.schema import DatasetContract
from broadway.contracts.pandera import build_raw_schema
from broadway.evaluate.contracts import EvaluationResult
from broadway.evaluate.metrics import compute_metrics
from broadway.evaluate.promotion import should_promote
from broadway.evaluate.validation import cross_validate, residual_summary
from broadway.features.schema import TARGET
from broadway.training.contracts import TrainingResult
from broadway.training.trainer import train

import project.data as data
from project.features import ENGINEERED_FEATURES, ENGINEERED_SCHEMA
from project.ml_pipeline import FeaturePipeline

_DATASET_YAML = Path("configs/dataset/taxi.yaml")


@pytest.fixture
def contract() -> DatasetContract:
    return DatasetContract(**yaml.safe_load(_DATASET_YAML.read_text()))


@pytest.fixture
def raw_frame(contract: DatasetContract) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 60
    frames: dict[str, pd.Series] = {}
    for name, col in contract.columns.items():
        if col.dtype.startswith("datetime"):
            frames[name] = pd.Series(pd.date_range("2024-01-01", periods=n, freq="h")).astype(
                col.dtype
            )
        elif col.dtype.startswith("int"):
            frames[name] = pd.Series(rng.integers(1, 4, n), dtype=col.dtype)
        else:
            frames[name] = pd.Series(rng.uniform(0.5, 60.0, n), dtype=col.dtype)
    return pd.DataFrame(frames)


def test_contract_driven_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract: DatasetContract,
    raw_frame: pd.DataFrame,
) -> None:
    # 1. Contract validation
    data_path = tmp_path / "training_data.parquet"
    raw_frame.to_parquet(data_path)

    zones_path = tmp_path / "taxi_zone_lookup.csv"
    pd.DataFrame(
        {
            data.ZONE_ID_COL: [1, 2, 3],
            data.ZONE_BOROUGH_COL: data.BOROUGHS[:3],
        }
    ).to_csv(zones_path, index=False)

    monkeypatch.setattr(data, "DATA_PATH", str(data_path))
    monkeypatch.setattr(data, "LOOKUP_PATH", str(zones_path))

    raw_schema = build_raw_schema(contract)
    raw_schema.validate(raw_frame)
    with pytest.raises(pandera.errors.SchemaError):
        raw_schema.validate(raw_frame.drop(columns=[TARGET]))

    # 2. Feature creation
    pipeline = FeaturePipeline(
        lookup_path=str(zones_path),
        encoding_smoothing=data.FEATURE_ENCODING_SMOOTHING,
        frequency_fill=data.FEATURE_FREQUENCY_FILL,
        rush_hour_morning_start=data.FEATURE_RUSH_HOUR_MORNING_START,
        rush_hour_morning_end=data.FEATURE_RUSH_HOUR_MORNING_END,
        rush_hour_evening_start=data.FEATURE_RUSH_HOUR_EVENING_START,
        rush_hour_evening_end=data.FEATURE_RUSH_HOUR_EVENING_END,
        night_start=data.FEATURE_NIGHT_START,
        night_end=data.FEATURE_NIGHT_END,
        passenger_count_min=data.FEATURE_PASSENGER_COUNT_MIN,
        passenger_count_max=data.FEATURE_PASSENGER_COUNT_MAX,
    )
    engineered = pipeline.fit_transform(raw_frame)
    assert list(engineered.columns) == list(ENGINEERED_FEATURES)
    ENGINEERED_SCHEMA.validate(engineered)

    # 3. Training
    cfg = load_config("train", dataset="taxi", experiment="taxi")
    assert cfg.experiment is not None
    assert cfg.train is not None
    model_type = cfg.experiment.model.type
    model_params = cfg.experiment.model.params

    model, training_result = train(model_type, engineered, raw_frame[TARGET], **model_params)
    assert isinstance(training_result, TrainingResult)
    assert training_result.model_type == model_type
    assert training_result.params == model_params
    assert training_result.train_time_seconds >= 0

    # 4. Evaluation
    y_true = raw_frame[TARGET].to_numpy()
    y_pred = model.predict(engineered)
    metrics = compute_metrics(y_true, y_pred)

    eval_cfg = load_config("evaluate", dataset="taxi", experiment="taxi")
    assert eval_cfg.evaluate is not None
    threshold = eval_cfg.evaluate.promotion_threshold

    promote, reason = should_promote(metrics["rmse"], None, threshold)
    cv_metrics = cross_validate(
        model,
        engineered.to_numpy(),
        y_true,
        cv_folds=cfg.train.cv_folds,
        random_state=cfg.train.random_state,
    )
    residuals = residual_summary(y_true, y_pred)

    result = EvaluationResult(
        metrics=metrics,
        promote=promote,
        reason=reason,
        cv_metrics=cv_metrics,
        residuals=residuals,
    )

    assert set(result.model_dump()) >= {"metrics", "promote", "reason"}
    assert set(result.metrics) == {"mae", "rmse", "r2"}
    assert result.promote is True
    assert result.reason
    assert result.cv_metrics is not None
    assert result.residuals is not None
