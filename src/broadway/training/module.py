"""Train model — optional HPO — log to MLflow — write TrainingResult."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
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
from broadway.data.loader import ArtifactStore, ProcessedStore
from broadway.data.splitter import calibration_split, split
from broadway.evaluate.contracts import BootstrapConfig
from broadway.evaluate.metrics import MetricBootstrap, compute_metrics
from broadway.features.generic import validate_engineered_frame
from broadway.features.plan import ColumnPlan
from broadway.lineage.ids import node_id
from broadway.lineage.records import write_record
from broadway.observability.profiling import phase
from broadway.reports.decision import Decision, DecisionRule, Evidence, SegmentEvidence
from broadway.reports.step_output import StepOutput
from broadway.tracking.bundle import TrackingBundle
from broadway.tracking.manifest import MANIFEST_FILENAME, tracking_dir
from broadway.training.contracts import TrainingResult
from broadway.training.hpo import run_hpo
from broadway.training.learning_curve import LearningCurve, LearningCurveResult
from broadway.training.mlflow_utils import (
    log_datasets,
    log_metrics,
    log_model,
    log_params,
    setup_mlflow,
)
from broadway.training.trainer import build_model_pipeline, train

logger = logging.getLogger(__name__)


def _processed_dir(cfg: PipelineConfig) -> Path:
    return ProcessedStore.from_config(cfg).directory


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
    assert cfg.dataset is not None and cfg.experiment is not None
    X, _ = ColumnPlan.from_config(cfg.dataset, cfg.experiment).apply(df)
    return X, df[cfg.dataset.target]


def _resolve_params(
    cfg: PipelineConfig, fit_df: pd.DataFrame, val_df: pd.DataFrame
) -> dict[str, float | int | str]:
    """Resolve final params (HPO-best when configured) on the deployed fit slice."""
    assert cfg.experiment is not None
    if cfg.experiment.hpo is None:
        return cfg.experiment.model.params
    X_train, y_train = _xy(cfg, fit_df)
    X_val, y_val = _xy(cfg, val_df)
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
    decision = TrainingDecision.measure(
        cfg,
        cfg.experiment.model.type,
        dict(best["best_params"]),
        cfg.experiment.model.params,
        X_train,
        y_train,
        X_val,
        y_val,
    )
    if decision is not None:
        published = decision.publish()
        logger.info(
            "training decision: %s — improvement over defaults %.4f",
            published.verdict,
            published.value,
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


class TrainingDecision:
    """Publish the HPO verdict: does the best trial beat the declared defaults?

    The metric is the relative improvement of the HPO-best params over the
    experiment's default params on the validation slice. The interval is a
    paired bootstrap over those rows; the candidate is adopted only when the
    interval's lower bound clears the evaluate step's promotion threshold (the
    one shipping bar the config declares).
    """

    def __init__(
        self,
        *,
        cfg: PipelineConfig,
        target_metric: str,
        best_params: dict[str, float | int | str],
        default_params: dict[str, float | int | str],
        best_value: float,
        default_value: float,
        y_val: np.ndarray,
        pred_best: np.ndarray,
        pred_default: np.ndarray,
        interval: tuple[float, float],
    ) -> None:
        assert cfg.dataset is not None and cfg.evaluate is not None
        if best_value <= 0.0:
            raise ValueError(
                f"relative HPO improvement needs a positive default metric, got {best_value}"
            )
        self._cfg = cfg
        self._target_metric = target_metric
        self._best_params = best_params
        self._default_params = default_params
        self._best_value = best_value
        self._default_value = default_value
        self._truth = y_val
        self._pred_best = pred_best
        self._pred_default = pred_default
        self._absolute = interval

    @classmethod
    def measure(
        cls,
        cfg: PipelineConfig,
        model_type: str,
        best_params: dict[str, float | int | str],
        default_params: dict[str, float | int | str],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> TrainingDecision | None:
        """Fit both param sets once and pair-bootstrap the improvement; None when unsupported."""
        if cfg.evaluate is None or cfg.experiment is None or cfg.experiment.hpo is None:
            return None
        target = cfg.experiment.hpo.target_metric
        if cfg.experiment.hpo.direction != "minimize" or target not in {"rmse", "mae", "mape"}:
            return None
        best_pipeline = build_model_pipeline(cfg, model_type, best_params).fit(X_train, y_train)
        default_pipeline = build_model_pipeline(cfg, model_type, default_params).fit(
            X_train, y_train
        )
        pred_best = np.asarray(best_pipeline.predict(X_val), dtype=float)
        pred_default = np.asarray(default_pipeline.predict(X_val), dtype=float)
        truth = y_val.to_numpy(dtype=float)
        best_value = float(compute_metrics(truth, pred_best, metrics=(target,))[target])
        default_value = float(
            compute_metrics(truth, pred_default, metrics=(target,))[target]
        )
        bootstrap = MetricBootstrap(
            BootstrapConfig(
                iterations=cfg.evaluate.bootstrap.iterations,
                seed=cfg.evaluate.bootstrap.seed,
                level=cfg.evaluate.bootstrap.level,
            )
        )
        interval = bootstrap.improvement_interval(
            truth, pred_best, pred_default, metric=target
        )
        return cls(
            cfg=cfg,
            target_metric=target,
            best_params=best_params,
            default_params=default_params,
            best_value=best_value,
            default_value=default_value,
            y_val=truth,
            pred_best=pred_best,
            pred_default=pred_default,
            interval=(interval.lower, interval.upper),
        )

    @property
    def relative_value(self) -> float:
        return (self._default_value - self._best_value) / self._default_value

    @property
    def relative_interval(self) -> tuple[float, float]:
        low, high = self._absolute
        return low / self._default_value, high / self._default_value

    def decision(self) -> Decision:
        assert self._cfg.dataset is not None and self._cfg.evaluate is not None
        low, high = self.relative_interval
        midpoint = len(self._truth) // 2

        def half_value(half: slice) -> float:
            best = float(
                compute_metrics(
                    self._truth[half], self._pred_best[half], metrics=(self._target_metric,)
                )[self._target_metric]
            )
            default = float(
                compute_metrics(
                    self._truth[half], self._pred_default[half], metrics=(self._target_metric,)
                )[self._target_metric]
            )
            return (default - best) / default if default else 0.0

        evidence = Evidence(
            baseline=(
                f"experiment defaults {self._default_params} on the validation slice"
            ),
            delta=self.relative_value,
            segments=[
                SegmentEvidence(
                    segment="first_half", value=half_value(slice(0, midpoint)), sample_size=midpoint
                ),
                SegmentEvidence(
                    segment="second_half",
                    value=half_value(slice(midpoint, len(self._truth))),
                    sample_size=len(self._truth) - midpoint,
                ),
            ],
            sensitivity={
                f"best_value_n{len(self._truth)}": self._best_value,
                f"default_value_n{len(self._truth)}": self._default_value,
            },
            falsifier=(
                "the improvement interval's lower bound falls below the evaluate "
                "step's promotion threshold"
            ),
            provenance={
                "dataset": self._cfg.dataset.name,
                "best_params": str(self._best_params),
                "default_params": str(self._default_params),
                "seed": str(self._cfg.evaluate.bootstrap.seed),
                "rows": str(len(self._truth)),
                "resampling": "paired row bootstrap on the validation slice",
            },
        )
        threshold = self._cfg.evaluate.promotion_threshold
        return Decision.evaluate(
            rule=DecisionRule(
                metric=f"improvement_over_default_{self._target_metric}",
                direction="max",
                min_effect=threshold,
                min_sample_size=100,
                action="adopt the HPO-best params",
                owner=self._cfg.dataset.name,
            ),
            value=self.relative_value,
            interval=(min(low, self.relative_value), max(high, self.relative_value)),
            sample_size=len(self._truth),
            question="Do the HPO-best params beat the declared defaults on validation?",
            evidence=evidence,
            unit="relative_improvement",
        )

    def publish(self) -> Decision:
        """Write training.md/.png and the decision record beside the training result."""
        assert self._cfg.train is not None
        decision = self.decision()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
        labels = ["default params", "HPO best"]
        values = [self._default_value, self._best_value]
        bars = axis.bar(labels, values, color=["#9ecae1", "#08519c"])
        for bar in bars:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():,.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axis.set_ylabel(self._target_metric)
        axis.set_title("HPO best vs declared defaults (validation)")
        output = StepOutput(
            results_dir=Path(self._cfg.train.output_dir), stem="training", decision=decision
        )
        output.save_figure(figure)
        plt.close("all")
        output.write()
        return decision


@dataclass(frozen=True)
class _Fitted:
    """The fitted model and every value the logging/persistence phases need."""

    model: Pipeline
    result: TrainingResult
    params: dict[str, float | int | str]
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    summary: CalibrationSummary | None


def _load_frames(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the persisted feature frames, splitting in memory when no val file exists."""
    assert cfg.dataset is not None and cfg.experiment is not None
    with phase("load_features"):
        train_df, val_df = _load_features(cfg)
    if val_df is None:
        with phase("split"):
            train_df, val_df = split(
                train_df, cfg.dataset, cfg.experiment.split, cfg.experiment.random_state
            )
    assert val_df is not None
    return train_df, val_df


def _tracking_bundle(cfg: PipelineConfig) -> TrackingBundle | None:
    """Resolve the optional Data Tracking bundle for this run."""
    assert cfg.dataset is not None and cfg.train is not None
    tracking = cfg.train.tracking
    if tracking is None:
        return None
    processed_dir = _processed_dir(cfg)
    return TrackingBundle.from_config(
        dataset=cfg.dataset.name,
        config=tracking,
        clean_result_path=processed_dir / f"{cfg.dataset.name}_clean.json",
        machine_dir=ArtifactStore.from_contract(cfg.dataset, tracking_dir()).directory,
    )


def _record_tracking(cfg: PipelineConfig, train_df: pd.DataFrame) -> TrackingBundle | None:
    """Write the training baseline; the returned bundle feeds artifact logging."""
    assert cfg.train is not None
    bundle = _tracking_bundle(cfg)
    if bundle is None:
        return None
    with phase("tracking_baseline"):
        bundle.record_training_truth(train_df)
    return bundle


def _calibration_frames(
    cfg: PipelineConfig,
    train_df: pd.DataFrame,
    calibration: CalibrationConfig | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Split the calibration slice off the fit frame; no calibration keeps one frame.

    The fit/calibration split is the only slice q is measured on; val_df stays
    the untouched holdout and never feeds the calibration quantile.
    """
    if calibration is None:
        return train_df, None
    assert cfg.dataset is not None and cfg.experiment is not None
    with phase("split"):
        return calibration_split(
            train_df,
            cfg.dataset,
            cfg.experiment.split,
            calibration.calibration_size,
            cfg.experiment.random_state,
        )


def _run_learning_curve(
    cfg: PipelineConfig,
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    params: dict[str, float | int | str],
) -> LearningCurveResult | None:
    """Run the training-size curve on the deployed fit slice before the full fit.

    The curve refits the same pipeline the final model trains: the slice after
    the calibration split and the resolved (HPO-best when configured) params.
    It works on fresh frames, so the final training and the persisted frames
    are untouched.
    """
    assert cfg.train is not None and cfg.experiment is not None
    if cfg.train.learning_curve is None:
        return None
    model_type = cfg.experiment.model.type
    return LearningCurve(
        cfg,
        cfg.train.learning_curve,
        lambda: build_model_pipeline(cfg, model_type, params),
    ).run(fit_df, val_df)


def _fit_model(
    cfg: PipelineConfig,
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    calibration: CalibrationConfig | None,
    calibration_df: pd.DataFrame | None,
    params: dict[str, float | int | str],
) -> _Fitted:
    """Fit the model with the resolved params and the calibration slice."""
    X_train, y_train = _xy(cfg, fit_df)
    X_val, y_val = _xy(cfg, val_df)
    model, result = train(cfg, X_train, y_train, **params)
    summary: CalibrationSummary | None = None
    if calibration is not None:
        assert calibration_df is not None
        with phase("calibration_fit"):
            summary = _fit_calibration(cfg, calibration_df, model, calibration)
        result = result.model_copy(update={"calibration": summary})
    return _Fitted(
        model=model,
        result=result,
        params=params,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        summary=summary,
    )


def _log_inputs(
    cfg: PipelineConfig, fitted: _Fitted, train_df: pd.DataFrame, val_df: pd.DataFrame
) -> None:
    """Log params, calibration, data source, and dataset lineage."""
    assert cfg.dataset is not None and cfg.experiment is not None and cfg.etl is not None
    log_params(fitted.params)
    if fitted.summary is not None:
        _log_calibration(fitted.summary)
    log_params(
        {
            f"data_source.{key}": value
            for key, value in cfg.experiment.data_source.model_dump().items()
            if value is not None
        }
    )
    # Dataset lineage from the already-loaded in-memory frames (D9): no parquet
    # re-read, source paths are lineage strings only. A missing source (e.g. val
    # built by in-memory split) warns and continues.
    processed_dir = _processed_dir(cfg)
    log_datasets(
        cfg.dataset.name,
        train_df,
        str(processed_dir / cfg.etl.train_features_file),
        val_df,
        str(processed_dir / cfg.etl.val_features_file),
    )


def _log_baseline_improvement(cfg: PipelineConfig, metrics: dict[str, float]) -> None:
    """Log the delta over the persisted baseline when both sides carry its metric."""
    assert cfg.dataset is not None
    baseline_result = load_persisted(cfg)
    if baseline_result is None or baseline_result.metric not in metrics:
        return
    improvement = improvement_vs_baseline(
        metrics[baseline_result.metric], baseline_result, cfg.dataset.task
    )
    if improvement is not None:
        mlflow.log_metric("baseline_improvement", improvement)
        logger.info(
            f"train: improvement over {baseline_result.strategy} baseline = {improvement:.1%}"
        )


def _log_evaluation(cfg: PipelineConfig, fitted: _Fitted) -> str:
    """Log val metrics, the baseline delta, the signature, and the model artifact."""
    # Val metrics are used for model selection — reported numbers carry
    # selection bias (optimistic); no held-out third split exists.
    metrics = compute_metrics(
        fitted.y_val.to_numpy(), fitted.model.predict(fitted.X_val)
    )
    log_metrics(metrics)
    _log_baseline_improvement(cfg, metrics)
    # Every integer column reaching the signature is null-free by enforced
    # pipeline construction: ETL keeps an int dtype only for null-free numeric
    # columns (parse_numeric, and build_raw_schema rejects a float-with-NaN
    # column against an int declaration); the datetime builders crash on nulls
    # (astype(int)); the boolean-derived builders (is_weekend, same_group, ...)
    # coerce nulls to 0. Declaring them float64 instead would break pyfunc
    # schema enforcement for the pipeline's own int64 inputs, so the int
    # signature is the accurate contract — hence the hint is a false alarm for
    # this matrix, and the suppression is scoped (module mlflow.types.utils,
    # message prefix-anchored) so it cannot swallow unrelated warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Hint: Inferred schema contains integer column",
            module=r"mlflow\.types\.utils",
        )
        signature = infer_signature(fitted.X_train, fitted.y_train)
    return log_model(fitted.model, "model", signature=signature)


def _log_learning_curve(cfg: PipelineConfig, curve: LearningCurveResult) -> None:
    """Log the curve's artifacts plus the reference and minimum-data answer."""
    assert cfg.train is not None and cfg.train.learning_curve is not None
    config = cfg.train.learning_curve
    for path in (config.output_path, config.csv_path, config.png_path):
        mlflow.log_artifact(str(path))
    log_params(
        {
            "learning_curve.split": curve.split,
            "learning_curve.direction": curve.direction,
            "learning_curve.target_metric": curve.target_metric,
        }
    )
    metrics: dict[str, float] = {
        "learning_curve.tolerance": curve.tolerance,
        "learning_curve.absolute_tolerance": curve.absolute_tolerance,
    }
    if curve.metric_threshold is not None:
        metrics["learning_curve.metric_threshold"] = curve.metric_threshold
    if curve.reference.mean is not None:
        metrics["learning_curve.reference"] = curve.reference.mean
    if curve.min_fraction is not None:
        metrics["learning_curve.min_fraction"] = curve.min_fraction
        if curve.min_rows is not None:
            metrics["learning_curve.min_rows"] = float(curve.min_rows)
    log_metrics(metrics)


def _log_tracking_artifacts(bundle: TrackingBundle | None) -> None:
    """Log the tracking bundle copy as MLflow run artifacts when it exists."""
    if bundle is None:
        return
    for path in (
        bundle.machine_dir / MANIFEST_FILENAME,
        bundle.baseline_path,
        bundle.data_card_path,
    ):
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path="tracking")


def _log_training(
    cfg: PipelineConfig,
    fitted: _Fitted,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    curve: LearningCurveResult | None,
    bundle: TrackingBundle | None,
) -> str:
    """Open the MLflow run and log its inputs and model artifact."""
    assert cfg.dataset is not None
    with phase("mlflow_log"):
        setup_mlflow(cfg.environment.mlflow_tracking_uri, cfg.dataset.name)
        with mlflow.start_run():
            _log_inputs(cfg, fitted, train_df, val_df)
            _log_tracking_artifacts(bundle)
            if curve is not None:
                _log_learning_curve(cfg, curve)
            return _log_evaluation(cfg, fitted)


def _persist(
    cfg: PipelineConfig, result: TrainingResult, summary: CalibrationSummary | None
) -> None:
    """Write the training result, optional calibration, and the lineage record."""
    assert cfg.train is not None and cfg.dataset is not None
    assert cfg.analysis is not None and cfg.analysis.name is not None
    with phase("persist"):
        out_dir = Path(cfg.train.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / cfg.train.output_file).write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
        logger.info(f"training result written to {out_dir / cfg.train.output_file}")
        if summary is not None:
            calibration_path = out_dir / CALIBRATION_FILE
            calibration_path.write_text(
                summary.model_dump_json(indent=2), encoding="utf-8"
            )
            logger.info(f"calibration written to {calibration_path}")
        write_record(
            node_id("training", cfg.dataset.name),
            "training",
            str(out_dir / cfg.train.output_file),
            [node_id("baseline", cfg.analysis.name), node_id("analysis", cfg.analysis.name)],
        )


def run(cfg: PipelineConfig) -> None:
    if not cfg.dataset or not cfg.experiment or not cfg.train or not cfg.etl:
        raise ValueError("training step requires dataset, experiment, train, and etl config")
    require_mode(cfg.analysis, AnalysisMode.PREDICTION)
    assert cfg.analysis is not None and cfg.analysis.name is not None

    calibration = _calibration_config(cfg)
    train_df, val_df = _load_frames(cfg)
    bundle = _record_tracking(cfg, train_df)
    fit_df, calibration_df = _calibration_frames(cfg, train_df, calibration)
    # Params first (HPO when configured), then the curve on the deployed fit
    # slice with those params: the 100% point matches the final model's data.
    params = _resolve_params(cfg, fit_df, val_df)
    curve = _run_learning_curve(cfg, fit_df, val_df, params)
    fitted = _fit_model(cfg, fit_df, val_df, calibration, calibration_df, params)
    artifact_path = _log_training(cfg, fitted, train_df, val_df, curve, bundle)
    result = fitted.result.model_copy(update={"artifact_path": artifact_path})
    _persist(cfg, result, fitted.summary)
