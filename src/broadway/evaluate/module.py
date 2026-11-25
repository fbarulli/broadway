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
import numpy as np
import pandas as pd

from broadway.analysis.contracts import AnalysisMode, require_mode
from broadway.baseline.improvement import improvement_vs_baseline
from broadway.baseline.module import load_persisted
from broadway.calibration import CALIBRATION_FILE
from broadway.calibration.conformal import SplitConformal
from broadway.calibration.contracts import (
    CalibrationEvaluation,
    CalibrationSummary,
    PeriodCoverage,
)
from broadway.calibration.metrics import coverage, coverage_by_period, mean_width
from broadway.config.schema import PipelineConfig
from broadway.evaluate.comparison import compare_models
from broadway.evaluate.contracts import BaselineComparison, EvaluationResult
from broadway.evaluate.metrics import compute_metrics
from broadway.evaluate.promotion import should_promote
from broadway.evaluate.validation import cross_validate, residual_summary
from broadway.features.generic import validate_engineered_frame
from broadway.features.plan import ColumnPlan
from broadway.lineage.ids import node_id
from broadway.lineage.records import write_record
from broadway.training.contracts import TrainingResult
from broadway.training.mlflow_utils import (
    get_champion,
    log_metrics,
    promote_candidate,
    setup_mlflow,
)
from broadway.training.trainer import build_model_pipeline

logger = logging.getLogger(__name__)


def _feature_frame(cfg: PipelineConfig, frame: pd.DataFrame) -> pd.DataFrame:
    """Project a persisted feature frame through the experiment's column plan."""
    assert cfg.dataset is not None and cfg.experiment is not None
    return ColumnPlan.from_config(cfg.dataset, cfg.experiment).apply(frame)[0]


def _processed_dir(cfg: PipelineConfig) -> Path:
    return Path(cfg.environment.data_dir) / cfg.environment.processed_subdir


def _load_val_features(
    cfg: PipelineConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    assert cfg.etl is not None and cfg.dataset is not None
    out_dir = _processed_dir(cfg)
    val_path = out_dir / cfg.etl.val_features_file
    if not val_path.exists():
        raise FileNotFoundError(
            f"validation features not found: {val_path} — evaluate requires a held-out set"
        )
    val_df = pd.read_parquet(val_path)
    validate_engineered_frame(cfg, val_df)
    assert cfg.dataset is not None
    return _feature_frame(cfg, val_df), val_df[cfg.dataset.target], val_df


def _load_train_features(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.Series]:
    assert cfg.etl is not None and cfg.dataset is not None
    out_dir = _processed_dir(cfg)
    train_df = pd.read_parquet(out_dir / cfg.etl.train_features_file)
    validate_engineered_frame(cfg, train_df)
    assert cfg.dataset is not None
    return _feature_frame(cfg, train_df), train_df[cfg.dataset.target]


def _load_training_result(cfg: PipelineConfig) -> TrainingResult:
    assert cfg.train is not None
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


def _period_coverage(
    cfg: PipelineConfig,
    val_df: pd.DataFrame,
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> list[PeriodCoverage]:
    """Per-ISO-week holdout coverage; empty when the dataset declares no datetime."""
    assert cfg.dataset is not None and cfg.evaluate is not None
    column = cfg.dataset.datetime_column
    if column is None or column not in val_df.columns:
        return []
    min_rows = cfg.evaluate.min_period_rows if cfg.evaluate.min_period_rows is not None else 1
    return coverage_by_period(
        pd.DatetimeIndex(val_df[column]), y_true, lower, upper, min_rows=min_rows
    )


def _calibration_evaluation(
    cfg: PipelineConfig,
    summary: CalibrationSummary,
    val_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> CalibrationEvaluation:
    """Evaluate the calibrated interval on the untouched holdout."""
    assert cfg.evaluate is not None
    lower, upper = SplitConformal().interval(y_pred, summary.q)
    observed = coverage(y_true, lower, upper)
    nominal = 1.0 - summary.alpha
    tolerance = cfg.evaluate.coverage_tolerance
    calibrated = None if tolerance is None else abs(observed - nominal) <= tolerance
    return CalibrationEvaluation(
        method=summary.method,
        alpha=summary.alpha,
        q=summary.q,
        nominal_coverage=nominal,
        coverage=observed,
        mean_width=mean_width(lower, upper),
        periods=_period_coverage(cfg, val_df, y_true, lower, upper),
        coverage_tolerance=tolerance,
        calibrated=calibrated,
    )


def _calibration_warning(evaluation: CalibrationEvaluation) -> str:
    """Name the observed-vs-nominal gap when the interval misses tolerance."""
    assert evaluation.coverage_tolerance is not None
    gap = evaluation.coverage - evaluation.nominal_coverage
    return (
        f"calibration coverage {evaluation.coverage:.3f} deviates from nominal "
        f"{evaluation.nominal_coverage:.3f} by {gap:+.3f} "
        f"(tolerance {evaluation.coverage_tolerance:.3f})"
    )


def _persist_calibration(
    eval_dir: Path, evaluation: CalibrationEvaluation
) -> None:
    """Write the calibration artifact and log coverage/width to MLflow."""
    (eval_dir / CALIBRATION_FILE).write_text(
        evaluation.model_dump_json(indent=2), encoding="utf-8"
    )
    with mlflow.start_run():
        log_metrics(
            {
                "calibration.coverage": evaluation.coverage,
                "calibration.mean_width": evaluation.mean_width,
            }
        )
    logger.info(
        f"evaluate: calibration coverage={evaluation.coverage:.4f}, "
        f"mean_width={evaluation.mean_width:.4f}, calibrated={evaluation.calibrated}"
    )


def run(cfg: PipelineConfig) -> None:
    if not cfg.dataset or not cfg.experiment or not cfg.evaluate or not cfg.etl or not cfg.train:
        raise ValueError("evaluate step requires dataset, experiment, evaluate, etl, and train config")
    require_mode(cfg.analysis, AnalysisMode.PREDICTION)
    warnings: list[str] = []

    setup_mlflow(cfg.environment.mlflow_tracking_uri, cfg.dataset.name)

    X_val, y_val, val_df = _load_val_features(cfg)
    result = _load_training_result(cfg)
    candidate = _load_candidate(result)

    y_true = y_val.to_numpy()
    y_pred = candidate.predict(X_val)
    candidate_metrics = compute_metrics(y_true, y_pred)

    calibration_evaluation: CalibrationEvaluation | None = None
    if result.calibration is not None:
        calibration_evaluation = _calibration_evaluation(
            cfg, result.calibration, val_df, y_true, y_pred
        )
        if calibration_evaluation.calibrated is False:
            warnings.append(_calibration_warning(calibration_evaluation))

    baseline_result = load_persisted(cfg)
    baseline_comparison: BaselineComparison | None = None
    if baseline_result is not None and baseline_result.metric in candidate_metrics:
        candidate_value = candidate_metrics[baseline_result.metric]
        baseline_comparison = BaselineComparison(
            metric=baseline_result.metric,
            baseline_value=baseline_result.value,
            candidate_value=candidate_value,
            delta=candidate_value - baseline_result.value,
            improvement=improvement_vs_baseline(candidate_value, baseline_result, cfg.dataset.task),
        )

    champion_uri = get_champion(cfg.dataset.name)
    champion_metrics: dict[str, float] | None = None
    if champion_uri is not None:
        champion = _load_champion(champion_uri)
        champion_metrics = compute_metrics(y_true, champion.predict(X_val))

    comparison = compare_models(candidate_metrics, champion_metrics)
    logger.debug("evaluate: champion comparison=%s", comparison)

    target_metric = cfg.evaluate.target_metric
    champion_score = champion_metrics[target_metric] if champion_metrics is not None else None
    # Reading 1 (T-BUG-2): the promotion DECISION keeps the pre-existing
    # champion-vs-holdout semantics of should_promote — it is made here,
    # before CV, and CV results are recorded but never gate promotion. The
    # reorder changes only WHERE promotion EXECUTES: after CV and after the
    # EvaluationResult is persisted (see the terminal-stage block below).
    promote, reason = should_promote(
        candidate_metrics[target_metric],
        champion_score,
        cfg.evaluate.promotion_threshold,
    )

    if champion_uri is None:
        warnings.append("no champion model found — candidate compared against none")

    X_train, y_train = _load_train_features(cfg)
    # Score the same composed Pipeline shape train/HPO fit (preprocessing +
    # registry model), so CV reflects the deployed pipeline rather than a bare
    # estimator; build_model_pipeline re-seeds from cfg.experiment.random_state.
    cv_model = build_model_pipeline(cfg, result.model_type, result.params)
    cv_metrics = cross_validate(
        cv_model,
        X_train.to_numpy(),
        y_train.to_numpy(),
        cfg.train.cv_folds,
        cfg.experiment.random_state,
        cfg.train.cv_kind,
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
        baseline=baseline_comparison,
        calibration=calibration_evaluation,
        warnings=warnings,
    )

    eval_dir = Path(cfg.evaluate.output_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / cfg.evaluate.output_file).write_text(evaluation.model_dump_json(indent=2), encoding="utf-8")
    if calibration_evaluation is not None:
        _persist_calibration(eval_dir, calibration_evaluation)
    write_record(
        node_id("evaluation", cfg.dataset.name),
        "evaluation",
        str(eval_dir / cfg.evaluate.output_file),
        [node_id("training", cfg.dataset.name)],
    )

    # Promotion is the terminal stage (T-BUG-2): it executes only after CV
    # and the EvaluationResult are persisted, so a raising stage — NaN or
    # object-dtype inputs, missing artifacts — can never leave a promoted
    # model with no evaluation artifact behind. A registry-unavailable
    # MlflowException stays a soft failure: the skip is logged AND the
    # already-persisted artifact is re-written with the warning appended, so
    # the persisted EvaluationResult still carries the promotion-skip signal
    # (belt and braces; the artifact existed before promotion in any case).
    if promote:
        dataset_name = cfg.dataset.name
        artifact_path = result.artifact_path
        try:
            assert dataset_name is not None and artifact_path is not None
            promote_candidate(dataset_name, artifact_path)
        except mlflow.exceptions.MlflowException as exc:
            warning = f"promotion skipped — model registry unavailable: {exc}"
            warnings.append(warning)
            logger.warning(warning)
            (eval_dir / cfg.evaluate.output_file).write_text(
                evaluation.model_copy(update={"warnings": warnings}).model_dump_json(indent=2),
                encoding="utf-8",
            )

    logger.info(f"evaluate: {target_metric}={candidate_metrics[target_metric]:.4f}, promote={promote}")
