"""Train model — optional HPO — log to MLflow — write TrainingResult."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from broadway.analysis.contracts import AnalysisMode, require_mode
from broadway.baseline.improvement import improvement_vs_baseline
from broadway.baseline.module import load_persisted
from broadway.calibration import CALIBRATION_FILE
from broadway.calibration.conformal import SplitConformal
from broadway.calibration.contracts import CalibrationConfig, CalibrationSummary
from broadway.config.schema import PipelineConfig, TaskType
from broadway.data.splitter import calibration_split, split
from broadway.evaluate.metrics import compute_metrics
from broadway.features.generic import validate_engineered_frame
from broadway.lineage.ids import node_id
from broadway.lineage.records import write_record
from broadway.training.hpo import run_hpo
from broadway.training.mlflow_utils import (
    log_datasets,
    log_metrics,
    log_model,
    log_params,
    setup_mlflow,
)
from broadway.training.trainer import train
from broadway.utils import eligible_feature_columns

logger = logging.getLogger(__name__)


def _processed_dir(cfg: PipelineConfig) -> Path:
    return Path(cfg.environment.data_dir) / cfg.environment.processed_subdir


def _load_features(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    assert cfg.etl is not None
    out_dir = _processed_dir(cfg)
    train_df = pd.read_parquet(out_dir / cfg.etl.train_features_file)
    val_path = out_dir / cfg.etl.val_features_file
    val_df = pd.read_parquet(val_path) if val_path.exists() else None
    # Re-enforce the engineered-feature contract on read (same SSOT the
    # features step writes with): a dtype, column-order, or target-dtype drift
    # in the persisted file must fail loud here, not surface mid-training.
    validate_engineered_frame(cfg, train_df)
    if val_df is not None:
        validate_engineered_frame(cfg, val_df)
    return train_df, val_df


def _xy(cfg: PipelineConfig, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    assert cfg.dataset is not None
    return eligible_feature_columns(df, cfg), df[cfg.dataset.target]


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
        cfg,
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


def _calibration_config(cfg: PipelineConfig) -> CalibrationConfig | None:
    """Return the experiment's calibration request, refusing unsupported combos.

    HPO selects params on a validation slice while split-conformal calibration
    needs its own exchangeable calibration slice; combining them in one step
    would leak the calibration slice into model selection, so it fails loud.
    """
    assert cfg.experiment is not None
    calibration = cfg.experiment.calibration
    if calibration is not None:
        assert cfg.dataset is not None
        if cfg.dataset.task is not TaskType.REGRESSION:
            raise ValueError(
                "calibration is regression-only: split-conformal intervals are "
                "fitted on absolute residuals, but the dataset task is "
                f"'{cfg.dataset.task.value}'"
            )
    if calibration is not None and cfg.experiment.hpo is not None:
        raise ValueError(
            "calibration with hpo is unsupported in this slice: calibration "
            "fits q on a held-out calibration slice while HPO selects params on "
            "its validation slice; configure hpo or calibration, not both"
        )
    return calibration


def _fit_calibration(
    cfg: PipelineConfig,
    calibration_df: pd.DataFrame,
    model: Pipeline,
    calibration: CalibrationConfig,
) -> CalibrationSummary:
    """Fit the residual quantile on the calibration slice (never on the holdout)."""
    assert cfg.dataset is not None
    X_cal, y_cal = _xy(cfg, calibration_df)
    residuals = np.abs(y_cal.to_numpy() - model.predict(X_cal))
    q = SplitConformal().quantile(residuals, calibration.alpha)
    return CalibrationSummary(
        method=calibration.method,
        alpha=calibration.alpha,
        q=q,
        n_calibration=len(calibration_df),
        calibration_size=calibration.calibration_size,
    )


def _log_calibration(summary: CalibrationSummary) -> None:
    log_params(
        {"calibration.method": summary.method, "calibration.alpha": summary.alpha}
    )
    log_metrics({"calibration.q": summary.q})


def run(cfg: PipelineConfig) -> None:
    if not cfg.dataset or not cfg.experiment or not cfg.train or not cfg.etl:
        raise ValueError("training step requires dataset, experiment, train, and etl config")
    require_mode(cfg.analysis, AnalysisMode.PREDICTION)
    assert cfg.analysis is not None and cfg.analysis.name is not None

    calibration = _calibration_config(cfg)
    train_df, val_df = _load_features(cfg)

    if val_df is None:
        train_df, val_df = split(train_df, cfg.dataset, cfg.experiment.split, cfg.experiment.random_state)

    fit_df = train_df
    calibration_df: pd.DataFrame | None = None
    if calibration is not None:
        # The fit/calibration split is the only slice q is measured on; val_df
        # stays the untouched holdout and never feeds the calibration quantile.
        fit_df, calibration_df = calibration_split(
            train_df,
            cfg.dataset,
            cfg.experiment.split,
            calibration.calibration_size,
            cfg.experiment.random_state,
        )

    X_train, y_train = _xy(cfg, fit_df)
    X_val, y_val = _xy(cfg, val_df)

    params = _resolve_params(cfg, X_train, y_train, X_val, y_val)

    model, result = train(cfg, X_train, y_train, **params)
    summary: CalibrationSummary | None = None
    if calibration is not None:
        assert calibration_df is not None
        summary = _fit_calibration(cfg, calibration_df, model, calibration)
        result = result.model_copy(update={"calibration": summary})

    setup_mlflow(cfg.environment.mlflow_tracking_uri, cfg.dataset.name)
    with mlflow.start_run():
        log_params(params)
        if summary is not None:
            _log_calibration(summary)
        log_params(
            {
                f"data_source.{key}": value
                for key, value in cfg.experiment.data_source.model_dump().items()
                if value is not None
            }
        )
        # Dataset lineage from the already-loaded in-memory frames (D9): no
        # parquet re-read, source paths are lineage strings only. A missing
        # source (e.g. val built by in-memory split) warns and continues.
        processed_dir = _processed_dir(cfg)
        assert cfg.dataset is not None and cfg.etl is not None
        log_datasets(
            cfg.dataset.name,
            train_df,
            str(processed_dir / cfg.etl.train_features_file),
            val_df,
            str(processed_dir / cfg.etl.val_features_file),
        )
        # Val metrics are used for model selection — reported numbers carry
        # selection bias (optimistic); no held-out third split exists.
        metrics = compute_metrics(y_val.to_numpy(), model.predict(X_val))
        log_metrics(metrics)
        baseline_result = load_persisted(cfg)
        if baseline_result is not None and baseline_result.metric in metrics:
            improvement = improvement_vs_baseline(metrics[baseline_result.metric], baseline_result, cfg.dataset.task)
            if improvement is not None:
                mlflow.log_metric("baseline_improvement", improvement)
                logger.info(f"train: improvement over {baseline_result.strategy} baseline = {improvement:.1%}")
        # Every integer column reaching the signature is null-free by enforced
        # pipeline construction: ETL keeps an int dtype only for null-free
        # numeric columns (parse_numeric, and build_raw_schema rejects a
        # float-with-NaN column against an int declaration); the datetime
        # builders crash on nulls (astype(int)); the boolean-derived builders
        # (is_weekend, same_group, ...) coerce nulls to 0. Declaring them
        # float64 instead would break pyfunc schema enforcement for the
        # pipeline's own int64 inputs, so the int signature is the accurate
        # contract — hence the hint is a false alarm for this matrix, and the
        # suppression is scoped (module mlflow.types.utils, message
        # prefix-anchored) so it cannot swallow unrelated warnings.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Hint: Inferred schema contains integer column",
                module=r"mlflow\.types\.utils",
            )
            signature = infer_signature(X_train, y_train)
        artifact_path = log_model(model, "model", signature=signature)

    result = result.model_copy(update={"artifact_path": artifact_path})

    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / cfg.train.output_file).write_text(result.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"training result written to {out_dir / cfg.train.output_file}")
    if summary is not None:
        calibration_path = out_dir / CALIBRATION_FILE
        calibration_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"calibration written to {calibration_path}")
    write_record(
        node_id("training", cfg.dataset.name),
        "training",
        str(out_dir / cfg.train.output_file),
        [node_id("baseline", cfg.analysis.name), node_id("analysis", cfg.analysis.name)],
    )
