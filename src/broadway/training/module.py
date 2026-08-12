"""Train model — optional HPO — log to MLflow — write TrainingResult."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import mlflow
import pandas as pd

from broadway.config.schema import PipelineConfig
from broadway.data.splitter import split
from broadway.evaluate.metrics import compute_metrics
from broadway.training.contracts import TrainingResult
from broadway.training.mlflow_utils import log_metrics, log_model, log_params, setup_mlflow
from broadway.training.optuna import run_study
from broadway.training.trainer import train
from broadway.utils import feature_columns

logger = logging.getLogger(__name__)


def _processed_dir(cfg: PipelineConfig) -> Path:
    return Path(cfg.environment.data_dir) / cfg.environment.processed_subdir


def _load_features(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    out_dir = _processed_dir(cfg)
    train_df = pd.read_parquet(out_dir / cfg.etl.train_features_file)
    val_path = out_dir / cfg.etl.val_features_file
    val_df = pd.read_parquet(val_path) if val_path.exists() else None
    return train_df, val_df


def _xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    return feature_columns(df, target), df[target]


def _hpo_objective(
    model_type: str,
    target_metric: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Callable[[dict[str, float | int]], float]:
    def objective(params: dict[str, float | int]) -> float:
        model, _ = train(model_type, X_train, y_train, **params)
        metrics = compute_metrics(y_val.to_numpy(), model.predict(X_val))
        return metrics[target_metric]

    return objective


def _resolve_params(
    cfg: PipelineConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, float | int | str]:
    if cfg.experiment.hpo is None:
        return cfg.experiment.model.params
    objective = _hpo_objective(
        cfg.experiment.model.type,
        cfg.experiment.target_metric,
        X_train,
        y_train,
        X_val,
        y_val,
    )
    return run_study(
        objective,
        search_space=cfg.experiment.hpo.search_space,
        n_trials=cfg.experiment.hpo.trials,
        direction="minimize",
        random_state=cfg.experiment.random_state,
    )


def run(cfg: PipelineConfig) -> None:
    if not cfg.dataset or not cfg.experiment or not cfg.train or not cfg.etl:
        raise ValueError("training step requires dataset, experiment, train, and etl config")

    train_df, val_df = _load_features(cfg)
    target = cfg.dataset.target

    if val_df is None:
        train_df, val_df = split(train_df, cfg.dataset, cfg.experiment.split, cfg.experiment.random_state)

    X_train, y_train = _xy(train_df, target)
    X_val, y_val = _xy(val_df, target)

    params = _resolve_params(cfg, X_train, y_train, X_val, y_val)

    model, result = train(cfg.experiment.model.type, X_train, y_train, **params)

    setup_mlflow(cfg.environment.mlflow_tracking_uri, cfg.dataset.name)
    with mlflow.start_run():
        log_params(params)
        metrics = compute_metrics(y_val.to_numpy(), model.predict(X_val))
        log_metrics(metrics)
        artifact_path = log_model(model, "model")

    result = result.model_copy(update={"artifact_path": artifact_path})

    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / cfg.train.output_file).write_text(result.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"training result written to {out_dir / cfg.train.output_file}")
