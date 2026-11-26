"""Load model → evaluate on holdout → cross-validate → check promotion.

A validation set is required: the evaluate step measures generalization on a
held-out split, so it raises if ``val_features_file`` is missing rather than
silently falling back to the training set.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

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
from broadway.calibration.metrics import (
    coverage,
    coverage_by_period,
    coverage_tolerance,
    mean_width,
)
from broadway.calibration.quantile import QuantileInterval
from broadway.config.schema import PipelineConfig
from broadway.evaluate.comparison import compare_models
from broadway.evaluate.contracts import (
    LEDGER_RESIDUAL_COLUMN,
    TIME_CV_KINDS,
    BaselineComparison,
    DecisionConfig,
    DecisionReport,
    ErrorAnalysisConfig,
    EvaluationResult,
    TimeSeriesCvResult,
)
from broadway.evaluate.metrics import MetricBootstrap, compute_metrics, tune_threshold
from broadway.evaluate.promotion import should_promote
from broadway.evaluate.validation import (
    ResidualDiagnosticsPass,
    TimeSeriesCV,
    cross_validate,
    residual_summary,
)
from broadway.features.generic import validate_engineered_frame
from broadway.features.plan import ColumnPlan
from broadway.lineage.ids import node_id
from broadway.lineage.records import write_record
from broadway.observability.profiling import timed
from broadway.reports.decision import Decision, DecisionRule, Evidence, SegmentEvidence
from broadway.reports.step_output import StepOutput
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


def _load_train_features(
    cfg: PipelineConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    assert cfg.etl is not None and cfg.dataset is not None
    out_dir = _processed_dir(cfg)
    train_df = pd.read_parquet(out_dir / cfg.etl.train_features_file)
    validate_engineered_frame(cfg, train_df)
    column = cfg.dataset.datetime_column
    dates = train_df[column] if column is not None else None
    return _feature_frame(cfg, train_df), train_df[cfg.dataset.target], dates


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


def _quantile_bounds(
    candidate: Any, X_val: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Read the candidate's native quantile pair, or fail loud naming the contract."""
    predict_interval = getattr(candidate, "predict_interval", None)
    if predict_interval is None:
        raise ValueError(
            "evaluate.interval_source='quantile' requires a candidate artifact that "
            "exposes predict_interval(features) -> (lower, mid, upper); the loaded "
            "pyfunc exposes predict only"
        )
    lower, mid, upper = predict_interval(X_val)
    return QuantileInterval().interval(lower, mid, upper)


def _quantile_alpha(params: dict[str, float | int | str]) -> float:
    """The interval miscoverage the quantile candidate was built at."""
    # PipelineConfig refuses a quantile interval source without this param.
    assert "alpha" in params
    return float(params["alpha"])


def _calibration_evaluation(
    cfg: PipelineConfig,
    summary: CalibrationSummary | None,
    candidate: Any,
    X_val: pd.DataFrame,
    val_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> CalibrationEvaluation:
    """Evaluate the configured interval source on the untouched holdout."""
    assert cfg.evaluate is not None
    method: Literal["split_conformal", "quantile"]
    if cfg.evaluate.interval_source == "conformal":
        assert summary is not None
        lower, upper = SplitConformal().interval(y_pred, summary.q)
        method, alpha, q = summary.method, summary.alpha, summary.q
    else:
        assert cfg.experiment is not None
        lower, upper = _quantile_bounds(candidate, X_val)
        method, alpha, q = (
            "quantile",
            _quantile_alpha(cfg.experiment.model.params),
            None,
        )
    observed = coverage(y_true, lower, upper)
    nominal = 1.0 - alpha
    configured = cfg.evaluate.coverage_tolerance
    tolerance = (
        None
        if configured is None
        else coverage_tolerance(nominal, len(y_true), configured)
    )
    calibrated = None if tolerance is None else abs(observed - nominal) <= tolerance
    return CalibrationEvaluation(
        method=method,
        alpha=alpha,
        q=q,
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


class PromotionDecision:
    """Interval-based promotion: the whole improvement interval must clear the bar.

    A point estimate above the threshold is not evidence: the paired bootstrap
    interval says how much of the lead survives holdout sampling noise. The
    candidate is promoted only when the interval's LOWER bound clears the
    declared relative threshold; otherwise the record states why it was held.
    The record (``promotion_decision.json``) is the verdict's evidence trail.
    """

    def __init__(
        self,
        *,
        cfg: PipelineConfig,
        target_metric: str,
        candidate_value: float,
        champion_value: float,
        champion_uri: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        champion_pred: np.ndarray,
        interval: tuple[float, float],
        level: float,
        sensitivity_interval: tuple[float, float],
    ) -> None:
        assert cfg.dataset is not None and cfg.evaluate is not None
        if champion_value <= 0.0:
            raise ValueError(
                f"promotion compares relative improvement, but the champion "
                f"{target_metric}={champion_value} is not positive"
            )
        self._cfg = cfg
        self._metric = target_metric
        self._candidate = candidate_value
        self._champion = champion_value
        self._champion_uri = champion_uri
        self._truth = np.asarray(y_true, dtype=float)
        self._pred = np.asarray(y_pred, dtype=float)
        self._champion_pred = np.asarray(champion_pred, dtype=float)
        self._absolute = interval
        self._level = level
        self._sensitivity = sensitivity_interval

    @classmethod
    def measure(
        cls,
        cfg: PipelineConfig,
        target_metric: str,
        candidate_value: float,
        champion_value: float,
        champion_uri: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        champion_pred: np.ndarray,
    ) -> PromotionDecision:
        """Pair-bootstrap the improvement, then the same interval at a second level."""
        assert cfg.evaluate is not None
        bootstrap = MetricBootstrap(cfg.evaluate.bootstrap)
        interval = bootstrap.improvement_interval(
            y_true, y_pred, champion_pred, metric=target_metric
        )
        sensitivity = MetricBootstrap(
            cfg.evaluate.bootstrap.model_copy(update={"level": 0.90})
        ).improvement_interval(y_true, y_pred, champion_pred, metric=target_metric)
        return cls(
            cfg=cfg,
            target_metric=target_metric,
            candidate_value=candidate_value,
            champion_value=champion_value,
            champion_uri=champion_uri,
            y_true=y_true,
            y_pred=y_pred,
            champion_pred=champion_pred,
            interval=(interval.lower, interval.upper),
            level=cfg.evaluate.bootstrap.level,
            sensitivity_interval=(sensitivity.lower, sensitivity.upper),
        )

    @property
    def relative_value(self) -> float:
        return (self._champion - self._candidate) / self._champion

    @property
    def relative_interval(self) -> tuple[float, float]:
        low, high = self._absolute
        return low / self._champion, high / self._champion

    @property
    def promote(self) -> bool:
        """Promotion is the computed verdict, not a bare interval comparison."""
        return self.decision().verdict == "act"

    def _segment_values(self, half: slice) -> tuple[float, float]:
        """Candidate and champion metric on one holdout half."""
        truth = self._truth[half]
        candidate = float(
            compute_metrics(truth, self._pred[half], metrics=(self._metric,))[self._metric]
        )
        champion = float(
            compute_metrics(truth, self._champion_pred[half], metrics=(self._metric,))[self._metric]
        )
        return candidate, champion

    def _segments(self) -> list[SegmentEvidence]:
        """Re-measure the relative improvement on the two holdout halves."""
        midpoint = len(self._truth) // 2
        segments: list[SegmentEvidence] = []
        for label, half in (
            ("first_half", slice(0, midpoint)),
            ("second_half", slice(midpoint, len(self._truth))),
        ):
            size = half.stop - half.start
            if size < 2:
                continue
            candidate, champion = self._segment_values(half)
            segments.append(
                SegmentEvidence(
                    segment=label,
                    value=(champion - candidate) / champion if champion else 0.0,
                    sample_size=size,
                )
            )
        return segments

    def decision(self) -> Decision:
        assert self._cfg.dataset is not None and self._cfg.evaluate is not None
        low, high = self.relative_interval
        threshold = self._cfg.evaluate.promotion_threshold
        evidence = Evidence(
            baseline=f"champion {self._champion_uri} on the same holdout",
            delta=self.relative_value,
            segments=self._segments(),
            sensitivity={
                "level_90pct_low": self._sensitivity[0] / self._champion,
                "level_90pct_high": self._sensitivity[1] / self._champion,
                "absolute_improvement_low": self._absolute[0],
                "absolute_improvement_high": self._absolute[1],
            },
            falsifier=(
                f"the {self._level:.0%} improvement interval's lower bound falls "
                f"below the {threshold:.1%} threshold"
            ),
            provenance={
                "dataset": self._cfg.dataset.name,
                "champion": self._champion_uri,
                "seed": str(self._cfg.evaluate.bootstrap.seed),
                "rows": str(len(self._truth)),
                "resampling": "paired row bootstrap on the holdout",
            },
        )
        return Decision.evaluate(
            rule=DecisionRule(
                metric=f"improvement_{self._metric}",
                direction="max",
                min_effect=threshold,
                min_sample_size=100,
                action="promote the candidate to champion",
                owner=self._cfg.dataset.name,
            ),
            value=self.relative_value,
            interval=(min(low, self.relative_value), max(high, self.relative_value)),
            sample_size=len(self._truth),
            question=(
                f"Does the candidate beat the champion on {self._metric} by more "
                f"than {threshold:.1%} on the holdout?"
            ),
            evidence=evidence,
            unit="relative_improvement",
        )

    def publish(self, eval_dir: Path) -> Decision:
        """Write promotion.md/.png and ``promotion_decision.json``; return the record."""
        decision = self.decision()
        output = StepOutput(results_dir=eval_dir, stem="promotion", decision=decision)
        output.save_figure(self._figure(decision))
        output.write()
        return decision

    def _figure(self, decision: Decision) -> object:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(6.0, 3.6), layout="constrained")
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.bar(
            ["relative improvement"],
            [decision.value],
            yerr=[
                [decision.value - decision.interval_low],
                [decision.interval_high - decision.value],
            ],
            color="#0072B2",
            capsize=6,
        )
        axis.axhline(
            decision.threshold, color="#D55E00", linestyle="--", label="promotion threshold"
        )
        axis.axhline(0.0, color="#999999", linewidth=0.8)
        axis.set_ylabel("relative improvement")
        axis.set_title("candidate vs champion — interval vs threshold")
        axis.legend(fontsize=8)
        return figure


class HoldoutErrorAnalysis:

    """Per-row holdout error ledger, plus the declared binary-decision report.

    The ledger always carries ``y_true``, ``y_pred``, ``residual`` (prediction
    minus truth, this codebase's residual sign), ``abs_error`` and the segment
    columns. Segment columns are the configured names validated against the
    experiment's ColumnPlan, or the whole plan when none are configured — the
    code never carries a column-name literal. ``thresholds.<json>`` is written
    only for a configured ``decision`` block; without one no cutoff is invented.
    """

    def __init__(
        self,
        config: ErrorAnalysisConfig,
        decision: DecisionConfig | None,
        plan: ColumnPlan,
        target: str,
    ) -> None:
        if decision is not None and decision.column is not None and decision.column != target:
            raise ValueError(
                f"decision column '{decision.column}' is not the scored target '{target}'"
            )
        self._config = config
        self._decision = decision
        self._plan = plan
        self._target = target

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> HoldoutErrorAnalysis:
        if cfg.dataset is None or cfg.experiment is None or cfg.evaluate is None:
            raise ValueError("error analysis requires dataset, experiment, and evaluate config")
        return cls(
            cfg.evaluate.error_analysis,
            cfg.evaluate.decision,
            ColumnPlan.from_config(cfg.dataset, cfg.experiment),
            cfg.dataset.target,
        )

    def write(
        self,
        eval_dir: Path,
        features: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        with timed() as elapsed:
            try:
                ledger_path = self._write_ledger(eval_dir, features, y_true, y_pred)
                report = self._write_thresholds(eval_dir, y_true, y_pred)
            except Exception:
                logger.exception("evaluate: error analysis failed")
                raise
        tuned = (
            f", tuned threshold {report.chosen_threshold} -> {self._config.thresholds_file}"
            if report is not None
            else ""
        )
        logger.info(
            "evaluate: error analysis wrote %d holdout rows -> %s in %.3fs%s",
            len(y_true),
            ledger_path.name,
            elapsed(),
            tuned,
        )

    def _segment_columns(self) -> tuple[str, ...]:
        """Configured segment columns, or the whole plan when none are given."""
        configured = self._config.segment_columns
        unknown = sorted(set(configured) - set(self._plan.names))
        if unknown:
            raise ValueError(
                f"error analysis segment column(s) not in the experiment column plan: {unknown}"
            )
        return tuple(dict.fromkeys(configured)) if configured else self._plan.names

    def _write_ledger(
        self,
        eval_dir: Path,
        features: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Path:
        truth = np.asarray(y_true, dtype=float)
        prediction = np.asarray(y_pred, dtype=float)
        ledger = pd.DataFrame(
            {
                "y_true": truth,
                "y_pred": prediction,
                LEDGER_RESIDUAL_COLUMN: prediction - truth,
                "abs_error": np.abs(prediction - truth),
            }
        )
        # A segment colliding with a ledger column raises in the join rather
        # than silently overwriting the measured values.
        segments = features[list(self._segment_columns())].reset_index(drop=True)
        ledger = ledger.join(segments)
        path = eval_dir / self._config.ledger_file
        ledger.to_parquet(path, index=False)
        return path

    def _write_thresholds(
        self, eval_dir: Path, y_true: np.ndarray, y_pred: np.ndarray
    ) -> DecisionReport | None:
        decision = self._decision
        if decision is None:
            return None
        labels = (np.asarray(y_true, dtype=float) >= decision.threshold).astype(int)
        grid = np.linspace(decision.grid.min, decision.grid.max, decision.grid.steps)
        chosen, rows = tune_threshold(labels, y_pred, grid, decision.objective)
        report = DecisionReport(
            column=decision.column or self._target,
            objective=decision.objective,
            label_threshold=decision.threshold,
            chosen_threshold=chosen,
            rows=rows,
        )
        (eval_dir / self._config.thresholds_file).write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )
        return report


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
    assert cfg.evaluate is not None
    with timed() as elapsed:
        metric_intervals = MetricBootstrap(cfg.evaluate.bootstrap).intervals(y_true, y_pred)
    logger.info(
        "evaluate: bootstrap %d draws at %.0f%% level, %d rows: rmse [%.4f, %.4f] in %.3fs",
        cfg.evaluate.bootstrap.iterations,
        cfg.evaluate.bootstrap.level * 100.0,
        len(y_true),
        metric_intervals["rmse"].lower,
        metric_intervals["rmse"].upper,
        elapsed(),
    )

    calibration_evaluation: CalibrationEvaluation | None = None
    if result.calibration is not None or cfg.evaluate.interval_source == "quantile":
        calibration_evaluation = _calibration_evaluation(
            cfg,
            result.calibration,
            candidate,
            X_val,
            val_df,
            y_true,
            y_pred,
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
    champion_pred: np.ndarray | None = None
    if champion_uri is not None:
        champion = _load_champion(champion_uri)
        champion_pred = np.asarray(champion.predict(X_val), dtype=float)
        champion_metrics = compute_metrics(y_true, champion_pred)

    comparison = compare_models(candidate_metrics, champion_metrics)
    logger.debug("evaluate: champion comparison=%s", comparison)

    target_metric = cfg.evaluate.target_metric
    champion_score = champion_metrics[target_metric] if champion_metrics is not None else None
    # Reading 1 (T-BUG-2): the promotion DECISION keeps the pre-existing
    # champion-vs-holdout semantics — it is made here, before CV, and CV
    # results are recorded but never gate promotion. The reorder changes only
    # WHERE promotion EXECUTES: after CV and after the EvaluationResult is
    # persisted (see the terminal-stage block below).
    #
    # The verdict is interval-based: the paired bootstrap's LOWER bound must
    # clear the declared relative threshold, so a point-estimate lead that
    # noise could erase does not ship. The record is published beside the
    # evaluation result as promotion_decision.json.
    promotion: PromotionDecision | None = None
    if champion_uri is not None and champion_score is not None and champion_pred is not None:
        try:
            promotion = PromotionDecision.measure(
                cfg,
                target_metric,
                candidate_metrics[target_metric],
                champion_score,
                champion_uri,
                y_true,
                y_pred,
                champion_pred,
            )
            decision = promotion.decision()
            promote = decision.verdict == "act"
            reason = (
                f"{decision.verdict}: {decision.metric}="
                f"{decision.value:.4f} [{decision.interval_low:.4f}, "
                f"{decision.interval_high:.4f}] vs threshold "
                f"{cfg.evaluate.promotion_threshold:.1%}"
            )
        except ValueError as error:
            promotion = None
            promote, reason = should_promote(
                candidate_metrics[target_metric],
                champion_score,
                cfg.evaluate.promotion_threshold,
            )
            warnings.append(f"interval-based promotion unavailable: {error}")
    else:
        promote, reason = should_promote(
            candidate_metrics[target_metric],
            champion_score,
            cfg.evaluate.promotion_threshold,
        )
        if champion_uri is None:
            warnings.append("no champion model found — candidate compared against none")

    X_train, y_train, train_dates = _load_train_features(cfg)
    # Score the same composed Pipeline shape train/HPO fit (preprocessing +
    # registry model), so CV reflects the deployed pipeline rather than a bare
    # estimator; build_model_pipeline re-seeds from cfg.experiment.random_state.
    cv_model = build_model_pipeline(cfg, result.model_type, result.params)
    cv_detail: TimeSeriesCvResult | None = None
    cv_decision: Decision | None = None
    if cfg.train.cv_kind in TIME_CV_KINDS:
        with timed() as elapsed:
            cv_detail, cv_decision = TimeSeriesCV.from_config(cfg).run(
                cv_model,
                X_train.to_numpy(),
                y_train.to_numpy(),
                dates=train_dates,
            )
        cv_metrics = cv_detail.metrics
        logger.info(
            "evaluate: %s cv folds=%d gap=%d horizon=%d seasonal=%s time_indexed=%s "
            "rmse=%.4f mae=%.4f in %.3fs",
            cv_detail.kind,
            cv_detail.folds,
            cv_detail.gap,
            cv_detail.horizon,
            cv_detail.seasonal_period,
            cv_detail.time_indexed,
            cv_detail.metrics["rmse"],
            cv_detail.metrics["mae"],
            elapsed(),
        )
        if cv_decision is not None:
            logger.info(
                "evaluate: cv decision verdict=%s improvement=%.4f [%.4f, %.4f] "
                "vs %s (owner=%s)",
                cv_decision.verdict,
                cv_decision.value,
                cv_decision.interval_low,
                cv_decision.interval_high,
                cv_decision.evidence.baseline,
                cv_decision.owner,
            )
    else:
        cv_metrics = cross_validate(
            cv_model,
            X_train.to_numpy(),
            y_train.to_numpy(),
            cfg.train.cv_folds,
            cfg.experiment.random_state,
            cfg.train.cv_kind,
            gap=cfg.train.cv_gap,
        )
    residuals = residual_summary(y_true, y_pred)

    eval_dir = Path(cfg.evaluate.output_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    if promotion is not None:
        published = promotion.publish(eval_dir)
        logger.info(
            "evaluate: promotion decision %s — %s %.4f [%.4f, %.4f]",
            published.verdict,
            published.metric,
            published.value,
            published.interval_low,
            published.interval_high,
        )
    HoldoutErrorAnalysis.from_config(cfg).write(eval_dir, X_val, y_true, y_pred)

    with timed() as elapsed:
        residual_diagnostics = ResidualDiagnosticsPass.from_config(cfg).compute(
            eval_dir / cfg.evaluate.error_analysis.ledger_file
        )
    logger.info(
        "evaluate: residual diagnostics on %d rows: acf=%s, ljung-box p=%.4f, "
        "jarque-bera p=%.4f, qq=%.4f in %.3fs",
        residual_diagnostics.n_obs,
        {lag: round(value, 4) for lag, value in residual_diagnostics.acf_lags.items()},
        residual_diagnostics.ljung_box_p_value,
        residual_diagnostics.jarque_bera_p_value,
        residual_diagnostics.qq_correlation,
        elapsed(),
    )

    evaluation = EvaluationResult(
        metrics=candidate_metrics,
        promote=promote,
        reason=reason,
        cv_metrics=cv_metrics,
        cv_detail=cv_detail,
        cv_decision=cv_decision,
        residuals=residuals,
        residual_diagnostics=residual_diagnostics,
        metric_intervals=metric_intervals,
        # persist the candidate-vs-champion comparison even when champion is None,
        # so downstream consumers can trace champion-None (delta=None) values explicitly.
        comparison=comparison,
        baseline=baseline_comparison,
        calibration=calibration_evaluation,
        warnings=warnings,
    )

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
