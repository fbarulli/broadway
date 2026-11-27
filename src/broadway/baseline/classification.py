"""Classification baseline: the fitted linear reference versus the no-skill prevalence.

The persisted baseline value is the naive prevalence's PR-AUC on the holdout (a
constant score has PR-AUC equal to the prevalence). The decision asks the only
question the baseline can answer honestly: does a fitted linear reference beat
that no-skill prevalence by more than the configured effect floor, with the
label-permutation null at this n as the noise band?
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from broadway.baseline.contracts import BaselineResult
from broadway.config.schema import BINARY_TARGET_NOTE, BootstrapConfig, PipelineConfig
from broadway.data.target import TargetCaster
from broadway.evaluate.metrics import MetricBootstrap, compute_metrics
from broadway.features.plan import ColumnPlan
from broadway.reports.decision import Decision, DecisionRule, Evidence, SegmentEvidence
from broadway.reports.decision_log import DecisionLog
from broadway.reports.step_output import StepOutput
from broadway.training.trainer import build_model_pipeline


@dataclass(frozen=True)
class ClassificationReference:
    """The holdout rows and the two references scored on them."""

    truth: np.ndarray
    prevalence: np.ndarray
    linear: np.ndarray
    prevalence_value: float
    linear_value: float


def _pr_auc(truth: np.ndarray, score: np.ndarray) -> float:
    return float(compute_metrics(truth, score, metrics=("pr_auc",))["pr_auc"])


def _linear_prediction(
    cfg: PipelineConfig, X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: np.ndarray
) -> np.ndarray:
    assert cfg.baseline is not None and cfg.experiment is not None
    if cfg.baseline.linear_reference_model is None:
        raise ValueError("baseline.linear_reference_model is not configured")
    model = build_model_pipeline(
        cfg, cfg.baseline.linear_reference_model, cfg.experiment.model.params
    )
    model.fit(X_train, y_train)
    return np.asarray(model.predict(X_val), dtype=float)


def build_reference(
    cfg: PipelineConfig, train: pd.DataFrame, val: pd.DataFrame
) -> ClassificationReference:
    """Fit the linear reference on train and score it against the prevalence on val."""
    assert cfg.dataset is not None and cfg.experiment is not None
    caster = TargetCaster.from_config(cfg)
    train = caster.cast(train)
    val = caster.cast(val)
    plan = ColumnPlan.from_config(cfg.dataset, cfg.experiment)
    X_train, _ = plan.apply(train)
    X_val, _ = plan.apply(val)
    target = cfg.dataset.target
    y_train = train[target].to_numpy(dtype=float)
    truth = val[target].to_numpy(dtype=float)
    prevalence_value = float(y_train.mean())
    prevalence = np.full(truth.shape, prevalence_value)
    linear = _linear_prediction(cfg, X_train, X_val, y_train)
    return ClassificationReference(
        truth=truth,
        prevalence=prevalence,
        linear=linear,
        prevalence_value=_pr_auc(truth, prevalence),
        linear_value=_pr_auc(truth, linear),
    )


class ClassificationBaselineDecision:
    """Publish the linear reference's paired improvement over the prevalence."""

    def __init__(
        self,
        cfg: PipelineConfig,
        reference: ClassificationReference,
        train: pd.DataFrame,
        val: pd.DataFrame,
    ) -> None:
        assert (
            cfg.dataset is not None
            and cfg.experiment is not None
            and cfg.baseline is not None
        )
        baseline = cfg.baseline
        if (
            baseline.bootstrap is None
            or baseline.min_effect is None
            or baseline.min_sample_size is None
            or baseline.action is None
            or baseline.owner is None
            or baseline.null_iterations is None
        ):
            raise ValueError(
                "the classification baseline needs bootstrap, min_effect, "
                "min_sample_size, null_iterations, action and owner in step/baseline.yaml"
            )
        self._cfg = cfg
        self._reference = reference
        self._train = train
        self._val = val
        self._bootstrap: BootstrapConfig = baseline.bootstrap
        self._null_iterations = baseline.null_iterations
        self._metric = cfg.experiment.target_metric
        self._rule = DecisionRule(
            metric=f"improvement_over_naive_{self._metric}",
            direction="max",
            min_effect=baseline.min_effect,
            min_sample_size=baseline.min_sample_size,
            action=baseline.action,
            owner=baseline.owner,
        )

    def _relative(self, candidate: np.ndarray, rows: slice | None = None) -> float:
        """Relative improvement of ``candidate`` over the prevalence on ``rows``."""
        reference = self._reference
        if rows is None:
            truth, prevalence = reference.truth, reference.prevalence
            reference_value = reference.prevalence_value
        else:
            truth = reference.truth[rows]
            prevalence = reference.prevalence[rows]
            candidate = candidate[rows]
            reference_value = _pr_auc(truth, prevalence)
        if reference_value <= 0.0:
            raise ValueError(
                f"the prevalence reference {self._metric} must be positive for a "
                "relative improvement"
            )
        return (_pr_auc(truth, candidate) - reference_value) / reference_value

    def _permutation_null(self) -> list[float]:
        """Relative improvement a no-signal fit produces on shuffled train labels."""
        assert self._cfg.dataset is not None and self._cfg.experiment is not None
        plan = ColumnPlan.from_config(self._cfg.dataset, self._cfg.experiment)
        X_train, _ = plan.apply(self._train)
        X_val, _ = plan.apply(self._val)
        generator = np.random.default_rng(self._bootstrap.seed)
        y_train = self._train[self._cfg.dataset.target].to_numpy(dtype=float)
        null: list[float] = []
        for _ in range(self._null_iterations):
            shuffled = y_train[generator.permutation(y_train.size)]
            prediction = _linear_prediction(self._cfg, X_train, X_val, shuffled)
            null.append(self._relative(prediction))
        return null

    def decision(self) -> Decision:
        assert self._cfg.dataset is not None and self._cfg.baseline is not None
        reference = self._reference
        value = self._relative(reference.linear)
        # The paired bootstrap returns ``reference - candidate`` (error-metric
        # orientation); a higher-is-better score metric flips the sign.
        interval = MetricBootstrap(self._bootstrap).improvement_interval(
            reference.truth,
            reference.linear,
            reference.prevalence,
            metric="pr_auc",
        )
        low, high = (
            -interval.upper / reference.prevalence_value,
            -interval.lower / reference.prevalence_value,
        )
        sensitivity = MetricBootstrap(
            self._bootstrap.model_copy(update={"level": 0.90})
        ).improvement_interval(
            reference.truth, reference.linear, reference.prevalence, metric="pr_auc"
        )
        biased = not (low <= value <= high)
        evidence = Evidence(
            baseline=(
                "naive training prevalence on the holdout rows (a constant score's "
                "PR-AUC equals the prevalence)"
            ),
            delta=value,
            segments=self._segments(),
            sensitivity={
                "level_90pct_low": -sensitivity.upper / reference.prevalence_value,
                "level_90pct_high": -sensitivity.lower / reference.prevalence_value,
                f"prevalence_n{len(reference.truth)}": reference.prevalence_value,
                f"linear_{self._metric}_n{len(reference.truth)}": reference.linear_value,
            },
            falsifier=(
                "the paired improvement interval's lower bound falls at or below the "
                "permutation null / configured effect floor"
            ),
            provenance={
                "dataset": self._cfg.dataset.name,
                "target": BINARY_TARGET_NOTE,
                "artifact": self._cfg.baseline.output_file,
                "seed": str(self._bootstrap.seed),
                "rows": str(len(reference.truth)),
                "prevalence": f"{reference.prevalence_value:.6f}",
                "model": str(self._cfg.baseline.linear_reference_model),
                "null_iterations": str(self._null_iterations),
                "resampling": "paired row bootstrap on the holdout; label permutation null",
            },
        )
        return Decision.evaluate(
            rule=self._rule,
            value=value,
            interval=(low, high),
            sample_size=len(reference.truth),
            question=(
                "Does the fitted linear reference beat the no-skill bad-review "
                f"prevalence on {self._metric}?"
            ),
            evidence=evidence,
            null_values=self._permutation_null(),
            unit="relative_improvement",
            bootstrap_biased=biased,
        )

    def _segments(self) -> list[SegmentEvidence]:
        """Re-measure the relative improvement on the two holdout halves."""
        reference = self._reference
        midpoint = len(reference.truth) // 2
        segments: list[SegmentEvidence] = []
        for label, half in (
            ("first_half", slice(0, midpoint)),
            ("second_half", slice(midpoint, len(reference.truth))),
        ):
            size = half.stop - half.start
            if size < 2:
                continue
            segments.append(
                SegmentEvidence(
                    segment=label,
                    value=self._relative(reference.linear, half),
                    sample_size=size,
                )
            )
        return segments

    def publish(self) -> Decision:
        assert self._cfg.baseline is not None
        decision = self.decision()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        reference = self._reference
        figure, axis = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
        labels = ["naive prevalence", "linear reference"]
        values = [reference.prevalence_value, reference.linear_value]
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
        axis.set_ylabel(self._metric)
        axis.set_title("no-skill prevalence vs linear reference (holdout)")
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


def persist(
    cfg: PipelineConfig, result: BaselineResult, reference: ClassificationReference
) -> BaselineResult:
    """Carry the holdout prevalence as the value ``evaluate`` must beat."""
    details = dict(result.details)
    details.update(
        {
            "holdout_prevalence": reference.prevalence_value,
            "holdout_linear_pr_auc": reference.linear_value,
            "holdout_rows": len(reference.truth),
        }
    )
    return result.model_copy(
        update={
            "strategy": "naive_prevalence",
            "metric": "pr_auc",
            "value": reference.prevalence_value,
            "details": details,
        }
    )
