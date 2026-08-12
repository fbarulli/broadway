from __future__ import annotations

import mlflow
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from broadway.training.contracts import TrainingResult
from broadway.training.mlflow_utils import log_metrics, log_model, setup_mlflow
from broadway.training.models.base import BaseModel
from broadway.training.optuna import run_study


def test_training_result_json_round_trip() -> None:
    result = TrainingResult(
        model_type="linear",
        params={"fit_intercept": 1, "alpha": 0.5},
        train_time_seconds=1.5,
        artifact_path="runs:/abc123/linear",
    )
    loaded = TrainingResult.model_validate_json(result.model_dump_json())
    assert loaded == result


def test_training_result_artifact_path_defaults_to_none() -> None:
    result = TrainingResult(
        model_type="lgbm",
        params={"n_estimators": 100},
        train_time_seconds=2.0,
    )
    assert result.artifact_path is None


def test_run_study_finds_quadratic_optimum() -> None:
    def objective(params: dict) -> float:
        return (params["x"] - 3.0) ** 2

    best = run_study(objective, n_trials=50, random_state=42)
    assert best["x"] == pytest.approx(3.0, abs=1.0)


def test_setup_mlflow_logs_without_server(tmp_path) -> None:
    setup_mlflow(str(tmp_path), "test_experiment")
    model = LinearRegression().fit(
        np.array([[1.0], [2.0], [3.0]]), np.array([2.0, 4.0, 6.0])
    )
    with mlflow.start_run():
        log_metrics({"rmse": 0.1, "r2": 0.99})
        uri = log_model(model, "linear")
    assert isinstance(uri, str)
    assert uri


def test_base_model_is_importable_and_abstract() -> None:
    assert callable(BaseModel)
    with pytest.raises(TypeError):
        BaseModel()
