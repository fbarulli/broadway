from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from broadway.analysis.contracts import AnalysisMode
from broadway.baseline import causal, classification, hypothesis, prediction
from broadway.baseline.contracts import BaselineResult, load_result, save_result
from broadway.config.schema import TARGET_NOTE, PipelineConfig, TaskType
from broadway.data.loader import ProcessedStore, load
from broadway.evaluate.contracts import BootstrapConfig
from broadway.evaluate.metrics import MetricBootstrap, compute_metrics
from broadway.features.generic import validate_engineered_frame
from broadway.features.plan import ColumnPlan
from broadway.lineage.ids import node_id
from broadway.lineage.records import write_record
from broadway.reports.decision import Decision, DecisionRule, Evidence, SegmentEvidence
from broadway.reports.decision_log import DecisionLog
from broadway.reports.measurement import MeasurementEstimator, MeasurementSpec
from broadway.reports.step_output import StepOutput
from broadway.trace import ArtifactTrace
from broadway.training.trainer import build_model_pipeline

logger = logging.getLogger(__name__)


def _git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return proc.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _build_trace(cfg: PipelineConfig) -> ArtifactTrace:
    # DOCUMENTED SILENCE (determinism ledger d): wall-clock bytes — created_at
    # stamps run time into persisted artifacts by design; byte-stability
    # awaits a freeze flag (pinned-timestamp config/env).
    return ArtifactTrace(
        created_at=datetime.now(UTC),
        commit=_git_commit(),
        dataset=cfg.dataset.name if cfg.dataset else None,
        analysis_goal=cfg.analysis.goal if cfg.analysis else None,
    )


def load_persisted(cfg: PipelineConfig) -> BaselineResult | None:
    if cfg.baseline is None:
        return None
    path = Path(cfg.baseline.output_dir) / cfg.baseline.output_file
    if not path.exists():
        return None
    return load_result(path)


def _compute_baseline(
    cfg: PipelineConfig, frame: pd.DataFrame | None = None
) -> BaselineResult:
    assert cfg.analysis is not None
    mode = cfg.analysis.mode
    if mode == AnalysisMode.PREDICTION:
        if not cfg.dataset:
            raise ValueError("prediction baseline requires a dataset config")
        df = load(cfg.dataset) if frame is None else frame
        return prediction.run(df, cfg.dataset.target, cfg.dataset.task)
    if mode == AnalysisMode.HYPOTHESIS:
        if not cfg.dataset:
            raise ValueError("hypothesis baseline requires a dataset config")
        if cfg.analysis is None or cfg.analysis.hypothesis is None:
            raise ValueError("hypothesis baseline requires an analysis contract with a hypothesis group")
        df = load(cfg.dataset)
        return hypothesis.run(
            df,
            cfg.dataset.target,
            cfg.analysis.hypothesis.group_column,
            cfg.analysis.hypothesis.group_values,
        )
    if mode == AnalysisMode.CAUSAL:
        if not cfg.causal:
            raise ValueError("causal baseline requires causal config")
        return causal.run(cfg.causal)
    raise ValueError(f"unsupported analysis mode: {mode}")


class BaselineDecision:
    """Publish which naive baseline the candidate must beat (mean vs median).

    The metric is ``mean_mae / median_mae``: above 1 the mean predictor is
    worse, so the median is the honest naive reference. Measured (and
    day-blocked when a datetime column exists) on the same rows the baseline
    scores, with the two MAE values as the evidence slices.
    """

    def __init__(self, cfg: PipelineConfig, result: BaselineResult, df: pd.DataFrame) -> None:
        assert cfg.analysis is not None and cfg.dataset is not None
        self._cfg = cfg
        self._result = result
        self._df = df

    def rule(self) -> DecisionRule:
        assert self._cfg.analysis is not None
        return DecisionRule(
            metric="mean_over_median_mae",
            direction="max",
            min_effect=1.0,
            min_sample_size=100,
            action="adopt the median as the naive baseline",
            owner=self._cfg.analysis.name or "baseline",
        )

    def _frame(self) -> pd.DataFrame:
        assert self._cfg.dataset is not None
        target = self._cfg.dataset.target
        frame = pd.DataFrame({"y": self._df[target].to_numpy(dtype=float)})
        column = self._cfg.dataset.datetime_column
        if column is not None and column in self._df.columns:
            frame["day"] = pd.to_datetime(self._df[column]).dt.floor("D")
        return frame

    @staticmethod
    def _ratio(frame: pd.DataFrame) -> float:
        mean_mae = float((frame["y"] - frame["y"].mean()).abs().mean())
        median_mae = float((frame["y"] - frame["y"].median()).abs().mean())
        return mean_mae / median_mae if median_mae else 0.0

    def decision(self) -> Decision:
        frame = self._frame()
        block = "day" if "day" in frame.columns else None
        spec = MeasurementSpec(
            bootstrap="block" if block else "iid",
            block_column=block,
            iterations=200,
            seed=7,
        )
        measurement = MeasurementEstimator().estimate(
            frame, self._ratio, spec, name=self.rule().metric, unit="ratio"
        )
        details = self._result.details
        mean_mae = float(details.get("mean_mae", 0.0))
        median_mae = float(details.get("median_mae", 0.0))
        midpoint = len(frame) // 2
        evidence = Evidence(
            baseline="mean and median predict the same value (ratio 1)",
            segments=[
                SegmentEvidence(
                    segment="first_half",
                    value=self._ratio(frame.iloc[:midpoint]),
                    sample_size=midpoint,
                ),
                SegmentEvidence(
                    segment="second_half",
                    value=self._ratio(frame.iloc[midpoint:]),
                    sample_size=len(frame) - midpoint,
                ),
            ],
            sensitivity={
                f"mean_mae_n{len(frame)}": mean_mae,
                f"median_mae_n{len(frame)}": median_mae,
            },
            falsifier="the mean/median MAE ratio interval falls to 1 (either ties)",
            provenance={
                "artifact": self._cfg.baseline.output_file if self._cfg.baseline else "baseline",
                "dataset": self._cfg.dataset.name if self._cfg.dataset else "",
                "seed": str(measurement.seed),
                "analysis_sample": str(measurement.sample_size),
                "block": block or "iid",
            },
        )
        return measurement.to_decision(
            self.rule(),
            question="Is the mean predictor worse than the median, so the candidate must beat the median?",
            evidence=evidence,
        )

    def publish(self, out_dir: Path) -> Decision:
        decision = self.decision()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        details = self._result.details
        figure, axis = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
        labels = ["mean predictor", "median predictor"]
        values = [float(details.get("mean_mae", 0.0)), float(details.get("median_mae", 0.0))]
        bars = axis.bar(labels, values, color=["#9ecae1", "#08519c"])
        for bar in bars:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():,.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axis.set_ylabel("holdout MAE")
        axis.set_title("naive baselines: mean vs median")
        output = StepOutput(
            results_dir=out_dir,
            stem="baseline",
            decision=decision,
            log=DecisionLog(tracking_uri=self._cfg.environment.mlflow_tracking_uri),
        )
        output.save_figure(figure)
        plt.close("all")
        output.write()
        return decision


@dataclass(frozen=True)
class HoldoutReference:
    """The holdout rows the naive and linear references are scored on."""

    truth: np.ndarray
    naive_prediction: np.ndarray
    linear_prediction: np.ndarray


def _load_feature_splits(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Read the persisted engineered splits, or None when the step has no features.

    The baseline is runnable without a feature plan (the platform's synthetic
    config); the linear reference exists only once the features step has
    published the model surface.
    """
    if cfg.etl is None or cfg.dataset is None or cfg.experiment is None:
        return None
    directory = ProcessedStore.from_config(cfg).directory
    train_path = directory / cfg.etl.train_features_file
    val_path = directory / cfg.etl.val_features_file
    if not train_path.is_file() or not val_path.is_file():
        return None
    train = pd.read_parquet(train_path)
    val = pd.read_parquet(val_path)
    validate_engineered_frame(cfg, train)
    validate_engineered_frame(cfg, val)
    return train, val


def _holdout_reference(
    cfg: PipelineConfig, train: pd.DataFrame, val: pd.DataFrame
) -> HoldoutReference:
    """Score the naive mean and the linear reference on the untouched holdout."""
    assert cfg.dataset is not None and cfg.experiment is not None and cfg.baseline is not None
    plan = ColumnPlan.from_config(cfg.dataset, cfg.experiment)
    X_train, _ = plan.apply(train)
    X_val, _ = plan.apply(val)
    y_train = train[cfg.dataset.target]
    truth = val[cfg.dataset.target].to_numpy(dtype=float)
    naive = np.full(truth.shape, float(y_train.mean()))
    if cfg.baseline.linear_reference_model is None:
        raise ValueError("baseline.linear_reference_model is not configured")
    model = build_model_pipeline(
        cfg, cfg.baseline.linear_reference_model, cfg.experiment.model.params
    )
    model.fit(X_train, y_train)
    linear = np.asarray(model.predict(X_val), dtype=float)
    return HoldoutReference(truth=truth, naive_prediction=naive, linear_prediction=linear)


def _bootstrap(cfg: PipelineConfig) -> MetricBootstrap:
    assert cfg.baseline is not None and cfg.baseline.bootstrap is not None
    return MetricBootstrap(cfg.baseline.bootstrap)


class LinearReferenceDecision:
    """Publish the linear reference's paired improvement over the naive mean.

    The metric is the relative MAE improvement of the fitted linear reference
    over a constant training-mean forecast on the same holdout rows. The
    interval is the paired row bootstrap; segments re-measure both holdout
    halves; the sensitivity repeats the interval at the 90% level.
    """

    def __init__(
        self, cfg: PipelineConfig, result: BaselineResult, reference: HoldoutReference
    ) -> None:
        assert cfg.dataset is not None and cfg.baseline is not None
        baseline = cfg.baseline
        if (
            baseline.linear_reference_model is None
            or baseline.bootstrap is None
            or baseline.min_effect is None
            or baseline.min_sample_size is None
            or baseline.action is None
            or baseline.owner is None
        ):
            raise ValueError(
                "the linear-reference baseline needs linear_reference_model, bootstrap, "
                "min_effect, min_sample_size, action and owner in step/baseline.yaml"
            )
        self._cfg = cfg
        self._result = result
        self._reference = reference
        self._bootstrap_spec: BootstrapConfig = baseline.bootstrap
        self._model = baseline.linear_reference_model
        self._output_file = baseline.output_file
        self._rule = DecisionRule(
            metric="improvement_over_naive_mae",
            direction="max",
            min_effect=baseline.min_effect,
            min_sample_size=baseline.min_sample_size,
            action=baseline.action,
            owner=baseline.owner,
        )

    def _mae(self, prediction: np.ndarray, rows: slice | np.ndarray) -> float:
        truth = self._reference.truth[rows]
        return float(compute_metrics(truth, prediction[rows], metrics=("mae",))["mae"])

    def decision(self) -> Decision:
        assert self._cfg.dataset is not None and self._cfg.baseline is not None
        reference = self._reference
        naive_mae = self._mae(reference.naive_prediction, slice(None))
        linear_mae = self._mae(reference.linear_prediction, slice(None))
        value = (naive_mae - linear_mae) / naive_mae
        measured = _bootstrap(self._cfg).improvement_interval(
            reference.truth, reference.linear_prediction, reference.naive_prediction, metric="mae"
        )
        low, high = measured.lower / naive_mae, measured.upper / naive_mae
        sensitivity = MetricBootstrap(
            self._bootstrap_spec.model_copy(update={"level": 0.90})
        ).improvement_interval(
            reference.truth, reference.linear_prediction, reference.naive_prediction, metric="mae"
        )
        midpoint = len(reference.truth) // 2
        segments = [
            SegmentEvidence(
                segment=label,
                value=self._segment(half),
                sample_size=half.stop - half.start,
            )
            for label, half in (
                ("first_half", slice(0, midpoint)),
                ("second_half", slice(midpoint, len(reference.truth))),
            )
        ]
        evidence = Evidence(
            baseline="naive training-mean predictor on the holdout rows",
            delta=value,
            segments=segments,
            sensitivity={
                "level_90pct_low": sensitivity.lower,
                "level_90pct_high": sensitivity.upper,
                f"naive_mae_n{len(reference.truth)}": naive_mae,
                f"linear_mae_n{len(reference.truth)}": linear_mae,
            },
            falsifier=(
                "the linear reference's paired MAE improvement interval falls at or "
                "below the configured effect floor"
            ),
            provenance={
                "dataset": self._cfg.dataset.name,
                "target": TARGET_NOTE,
                "artifact": self._output_file,
                "seed": str(self._bootstrap_spec.seed),
                "rows": str(len(reference.truth)),
                "resampling": "paired row bootstrap on the holdout",
            },
        )
        return Decision.evaluate(
            rule=self._rule,
            value=value,
            interval=(low, high),
            sample_size=len(reference.truth),
            question="Does the linear reference beat the naive mean on the holdout?",
            evidence=evidence,
            unit="relative_improvement",
        )

    def _segment(self, half: slice) -> float:
        reference = self._reference
        naive = self._mae(reference.naive_prediction, half)
        linear = self._mae(reference.linear_prediction, half)
        return (naive - linear) / naive if naive else 0.0

    def publish(self) -> Decision:
        assert self._cfg.baseline is not None
        decision = self.decision()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        reference = self._reference
        figure, axis = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
        labels = ["naive mean", "linear reference"]
        values = [
            self._mae(reference.naive_prediction, slice(None)),
            self._mae(reference.linear_prediction, slice(None)),
        ]
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
        axis.set_ylabel("holdout MAE")
        axis.set_title("naive mean vs linear reference (holdout)")
        output = StepOutput(
            results_dir=Path(self._cfg.baseline.output_dir),
            stem="baseline",
            decision=decision,
            log=DecisionLog(tracking_uri=self._cfg.environment.mlflow_tracking_uri),
        )
        output.save_figure(figure)
        plt.close("all")
        output.write()
        return decision


def _persisted_prediction_result(
    cfg: PipelineConfig, result: BaselineResult, reference: HoldoutReference
) -> BaselineResult:
    """Carry the holdout naive MAE as the persisted reference value.

    ``evaluate`` compares the candidate's holdout MAE with this value, so it
    must be measured on the same rows; the training-split naive metrics stay in
    ``details`` as evidence.
    """
    holdout_mae = float(
        compute_metrics(reference.truth, reference.naive_prediction, metrics=("mae",))["mae"]
    )
    holdout_rmse = float(
        compute_metrics(reference.truth, reference.naive_prediction, metrics=("rmse",))["rmse"]
    )
    details = dict(result.details)
    details.update(
        {
            "holdout_mean_mae": holdout_mae,
            "holdout_mean_rmse": holdout_rmse,
            "holdout_linear_mae": float(
                compute_metrics(
                    reference.truth, reference.linear_prediction, metrics=("mae",)
                )["mae"]
            ),
            "holdout_rows": len(reference.truth),
        }
    )
    return result.model_copy(update={"value": holdout_mae, "details": details})


def _run_classification(
    cfg: PipelineConfig,
    splits: tuple[pd.DataFrame, pd.DataFrame] | None,
    out_path: Path,
) -> BaselineResult:
    """Persist and publish the no-skill prevalence baseline for a binary target."""
    assert cfg.baseline is not None and cfg.dataset is not None
    if splits is None:
        raise ValueError(
            "the classification baseline requires the features step's train/val "
            "splits (run features first)"
        )
    train, val = splits
    reference = classification.build_reference(cfg, train, val)
    result = _compute_baseline(cfg, train)
    result = classification.persist(cfg, result, reference)
    result = result.model_copy(update={"trace": _build_trace(cfg)})
    save_result(result, out_path)
    decision = classification.ClassificationBaselineDecision(
        cfg, reference, train, val
    ).publish()
    logger.info(
        "baseline decision: %s — linear %s vs prevalence %.4f = %.4f [%.4f, %.4f]",
        decision.verdict,
        decision.metric,
        reference.prevalence_value,
        decision.value,
        decision.interval_low,
        decision.interval_high,
    )
    logger.info(
        "baseline: %s %s=%.4f -> %s", result.strategy, result.metric, result.value, out_path
    )
    return result


def run(cfg: PipelineConfig) -> None:
    if not cfg.analysis:
        raise ValueError("baseline step requires an analysis contract (--analysis)")
    if not cfg.baseline:
        raise ValueError("baseline step requires baseline config")
    splits = _load_feature_splits(cfg)
    out_dir = Path(cfg.baseline.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg.baseline.output_file
    if cfg.dataset is not None and cfg.dataset.task is TaskType.CLASSIFICATION:
        result = _run_classification(cfg, splits, out_path)
    else:
        reference = _holdout_reference(cfg, *splits) if splits is not None else None
        result = _compute_baseline(cfg, splits[0] if splits is not None else None)
        if reference is not None:
            result = _persisted_prediction_result(cfg, result, reference)
        result = result.model_copy(update={"trace": _build_trace(cfg)})
        save_result(result, out_path)
        if result.mode == AnalysisMode.PREDICTION and cfg.dataset is not None:
            if reference is not None:
                decision = LinearReferenceDecision(cfg, result, reference).publish()
                logger.info(
                    "baseline decision: %s — linear vs naive improvement %.4f",
                    decision.verdict,
                    decision.value,
                )
            else:
                decision = BaselineDecision(cfg, result, load(cfg.dataset)).publish(out_dir)
                logger.info(
                    "baseline decision: %s — mean/median MAE ratio %.4f",
                    decision.verdict,
                    decision.value,
                )
        logger.info(f"baseline: {result.strategy} {result.metric}={result.value:.4f} -> {out_path}")
    parents = (
        [node_id("analysis", cfg.analysis.name), node_id("profile", cfg.dataset.name)]
        if cfg.dataset is not None
        else [node_id("analysis", cfg.analysis.name)]
    )
    write_record(node_id("baseline", cfg.analysis.name), "baseline", str(out_path), parents)
