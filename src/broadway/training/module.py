"""Train model — optional HPO — log to MLflow — write TrainingResult."""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import pandas as pd

from broadway.analysis.contracts import AnalysisMode, require_mode
from broadway.baseline.improvement import improvement_vs_baseline
from broadway.baseline.module import load_persisted
from broadway.config.schema import PipelineConfig
from broadway.data.splitter import split
from broadway.evaluate.metrics import compute_metrics
from broadway.lineage.ids import node_id
from broadway.lineage.records import write_record
from broadway.training.hpo import run_hpo
from broadway.training.mlflow_utils import (
    log_metrics,
    log_model,
    log_params,
    setup_mlflow,
)
from broadway.training.trainer import train
from broadway.utils import feature_columns

logger = logging.getLogger(__name__)


def _processed_dir(cfg: PipelineConfig) -> Path:
    return Path(cfg.environment.data_dir) / cfg.environment.processed_subdir


def _load_features(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    assert cfg.etl is not None
    out_dir = _processed_dir(cfg)
    train_df = pd.read_parquet(out_dir / cfg.etl.train_features_file)
    val_path = out_dir / cfg.etl.val_features_file
    val_df = pd.read_parquet(val_path) if val_path.exists() else None
    return train_df, val_df


def _xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    return feature_columns(df, target), df[target]


def _resolve_params(
    cfg: PipelineConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, float | int | str]:
    assert cfg.experiment is not None
    if cfg.experiment.hpo is None:
        return cfg.experiment.model.params
    result = run_hpo(
        cfg.experiment.hpo,
        X_train,
        y_train,
        X_val,
        y_val,
        random_state=cfg.experiment.random_state,
    )
    # The pipeline trains cfg.experiment.model.type: when hpo.models holds
    # several entries, the others only feed the leaderboard and bandit
    # allocation — the study target is the entry matching model.type.
    best = result["models"].get(cfg.experiment.model.type)
    if best is None:
        raise ValueError(
            f"hpo produced no valid trial for model '{cfg.experiment.model.type}'"
        )
    return dict(best["best_params"])


def run(cfg: PipelineConfig) -> None:
    if not cfg.dataset or not cfg.experiment or not cfg.train or not cfg.etl:
        raise ValueError("training step requires dataset, experiment, train, and etl config")
    require_mode(cfg.analysis, AnalysisMode.PREDICTION)
    assert cfg.analysis is not None and cfg.analysis.name is not None

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
        baseline_result = load_persisted(cfg)
        if baseline_result is not None and baseline_result.metric in metrics:
            improvement = improvement_vs_baseline(metrics[baseline_result.metric], baseline_result, cfg.dataset.task)
            if improvement is not None:
                mlflow.log_metric("baseline_improvement", improvement)
                logger.info(f"train: improvement over {baseline_result.strategy} baseline = {improvement:.1%}")
        artifact_path = log_model(model, "model")

    result = result.model_copy(update={"artifact_path": artifact_path})

    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / cfg.train.output_file).write_text(result.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"training result written to {out_dir / cfg.train.output_file}")
    write_record(
        node_id("training", cfg.dataset.name),
        "training",
        str(out_dir / cfg.train.output_file),
        [node_id("baseline", cfg.analysis.name), node_id("analysis", cfg.analysis.name)],
    )
