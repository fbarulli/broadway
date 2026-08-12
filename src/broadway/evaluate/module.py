"""Load model → evaluate on holdout → cross-validate → check promotion.

A validation set is required: the evaluate step measures generalization on a
held-out split, so it raises if ``val_features_file`` is missing rather than
silently falling back to the training set.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from broadway.config.schema import PipelineConfig
from broadway.evaluate.comparison import compare_models
from broadway.evaluate.contracts import EvaluationResult
from broadway.evaluate.metrics import compute_metrics
from broadway.evaluate.promotion import should_promote
from broadway.evaluate.validation import cross_validate, residual_summary
from broadway.training.contracts import TrainingResult
from broadway.training.mlflow_utils import get_champion, promote_candidate, setup_mlflow
from broadway.training.models.registry import get_model
from broadway.utils import feature_columns

logger = logging.getLogger(__name__)


def _processed_dir(cfg: PipelineConfig) -> Path:
    return Path(cfg.environment.data_dir) / cfg.environment.processed_subdir


def _load_val_features(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.Series]:
    out_dir = _processed_dir(cfg)
    val_path = out_dir / cfg.etl.val_features_file
    if not val_path.exists():
        raise FileNotFoundError(
            f"validation features not found: {val_path} — evaluate requires a held-out set"
        )
    val_df = pd.read_parquet(val_path)
    target = cfg.dataset.target
    return feature_columns(val_df, target), val_df[target]


def _load_train_features(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.Series]:
    out_dir = _processed_dir(cfg)
    train_df = pd.read_parquet(out_dir / cfg.etl.train_features_file)
    target = cfg.dataset.target
    return feature_columns(train_df, target), train_df[target]


def _load_training_result(cfg: PipelineConfig) -> TrainingResult:
    path = Path(cfg.train.output_dir) / cfg.train.output_file
    if not path.exists():
        raise FileNotFoundError(f"training result not found: {path} — run the train step first")
    return TrainingResult.model_validate_json(path.read_text(encoding="utf-8"))


def _load_candidate(result: TrainingResult) -> Any:
    if not result.artifact_path:
        raise ValueError("training result has no artifact_path — model was not logged to MLflow")
    return mlflow.pyfunc.load_model(result.artifact_path)


def _load_champion(model_uri: str) -> Any:
    return mlflow.pyfunc.load_model(model_uri)


def run(cfg: PipelineConfig) -> None:
    if not cfg.dataset or not cfg.experiment or not cfg.evaluate or not cfg.etl or not cfg.train:
        raise ValueError("evaluate step requires dataset, experiment, evaluate, etl, and train config")

    setup_mlflow(cfg.environment.mlflow_tracking_uri, cfg.dataset.name)

    X_val, y_val = _load_val_features(cfg)
    result = _load_training_result(cfg)
    candidate = _load_candidate(result)

    y_true = y_val.to_numpy()
    y_pred = candidate.predict(X_val)
    candidate_metrics = compute_metrics(y_true, y_pred)

    champion_uri = get_champion(cfg.dataset.name)
    champion_metrics: dict[str, float] | None = None
    if champion_uri is not None:
        champion = _load_champion(champion_uri)
        champion_metrics = compute_metrics(y_true, champion.predict(X_val))

    comparison = compare_models(candidate_metrics, champion_metrics)
    logger.debug("evaluate: champion comparison=%s", comparison)

    target_metric = cfg.evaluate.target_metric
    champion_score = champion_metrics[target_metric] if champion_metrics is not None else None
    promote, reason = should_promote(
        candidate_metrics[target_metric],
        champion_score,
        cfg.evaluate.promotion_threshold,
    )

    X_train, y_train = _load_train_features(cfg)
    cv_model = get_model(result.model_type, **result.params)
    cv_metrics = cross_validate(
        cv_model,
        X_train.to_numpy(),
        y_train.to_numpy(),
        cfg.train.cv_folds,
        cfg.experiment.random_state,
    )
    residuals = residual_summary(y_true, y_pred)

    evaluation = EvaluationResult(
        metrics=candidate_metrics,
        promote=promote,
        reason=reason,
        cv_metrics=cv_metrics,
        residuals=residuals,
        # persist the candidate-vs-champion comparison even when champion is None,
        # so downstream consumers can trace champion-None (delta=None) values explicitly.
        comparison=comparison,
    )

    eval_dir = Path(cfg.evaluate.output_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / cfg.evaluate.output_file).write_text(evaluation.model_dump_json(indent=2), encoding="utf-8")

    if promote:
        try:
            promote_candidate(cfg.dataset.name, result.artifact_path)
        except mlflow.exceptions.MlflowException as exc:
            logger.warning("promotion skipped — model registry unavailable: %s", exc)

    logger.info(f"evaluate: {target_metric}={candidate_metrics[target_metric]:.4f}, promote={promote}")
