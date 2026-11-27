"""Model selection — repeated shuffled CV, paired improvement, one champion.

Cross-sectional survey rows are exchangeable, so the folds are a repeated
shuffled KFold (never the forward-chaining time geometry). Every configured
candidate and the naive mean reference score on the identical folds; the
champion is chosen by the configured rule (point-estimate leaderboard or the
simplest candidate within one standard error) and the published decision is its
paired row-level improvement over the naive reference, with a 95% interval from
the canonical ``MetricBootstrap``. The champion-vs-linear interval is evidence
in the same record, and the champion manifest is the HPO handoff.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from broadway.config.schema import (
    NAIVE_PREVALENCE_REFERENCE,
    TARGET_NOTE,
    BootstrapConfig,
    PipelineConfig,
    SelectionStep,
    TaskType,
)
from broadway.data.loader import ProcessedStore
from broadway.data.target import TargetCaster
from broadway.evaluate.leaderboard import ChampionDecision, MetricDirection, MetricLeaderboard
from broadway.evaluate.metrics import MetricBootstrap, compute_metrics
from broadway.evaluate.repeated_cv import CVResult, RepeatedCV
from broadway.features.generic import validate_engineered_frame
from broadway.features.plan import ColumnPlan
from broadway.observability.profiling import timed
from broadway.reports.decision import Decision, DecisionRule, Evidence, SegmentEvidence
from broadway.reports.decision_log import DecisionLog
from broadway.reports.step_output import StepOutput
from broadway.selection.candidates import CandidateSet

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectionFrame:
    """The persisted train features, their target, and the grouping column."""

    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series | None

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> SelectionFrame:
        """Read the persisted train features through the experiment's column plan."""
        assert cfg.dataset is not None and cfg.experiment is not None and cfg.etl is not None
        out_dir = ProcessedStore.from_config(cfg).directory
        frame = pd.read_parquet(out_dir / cfg.etl.train_features_file)
        validate_engineered_frame(cfg, frame)
        frame = TargetCaster.from_config(cfg).cast(frame)
        plan = ColumnPlan.from_config(cfg.dataset, cfg.experiment)
        X, _ = plan.apply(frame)
        groups = (
            None
            if cfg.dataset.group_column is None
            else frame[cfg.dataset.group_column].rename(cfg.dataset.group_column)
        )
        return cls(X=X, y=frame[cfg.dataset.target], groups=groups)


class ChampionRecord(BaseModel):
    """The selected candidate and the folds it was selected on (config handoff)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    champion: str
    model: str
    params: dict[str, float | int | str]
    metric: str
    cv_mean: float
    interval_low: float
    interval_high: float
    selection_rule: str
    rule_fired: str
    vs_linear_improvement: float


def _metric(truth: np.ndarray, prediction: np.ndarray, metric: str) -> float:
    return float(compute_metrics(truth, prediction, metrics=(metric,))[metric])


def _relative_improvement(
    truth: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    metric: str,
    bootstrap: BootstrapConfig,
) -> tuple[float, float, float]:
    """Paired row-level improvement of ``candidate`` over ``reference`` on the OOF rows."""
    candidate_value = _metric(truth, candidate, metric)
    reference_value = _metric(truth, reference, metric)
    if reference_value <= 0.0:
        raise ValueError(f"reference {metric} must be positive for a relative delta")
    interval = MetricBootstrap(bootstrap).improvement_interval(
        truth, candidate, reference, metric=metric
    )
    value = (reference_value - candidate_value) / reference_value
    return value, interval.lower / reference_value, interval.upper / reference_value


class SelectionDecision:
    """Publish the champion's paired row-level improvement over the references."""

    def __init__(
        self,
        cfg: PipelineConfig,
        step: SelectionStep,
        results: dict[str, CVResult],
        champion: str,
        rule_fired: str,
        one_se_alternative: str,
        truth: np.ndarray,
        sensitivity_improvement: float,
        grouping: str,
        features: pd.DataFrame | None = None,
    ) -> None:
        assert cfg.dataset is not None
        self._cfg = cfg
        self._step = step
        self._results = results
        self._champion = champion
        self._rule_fired = rule_fired
        self._one_se_alternative = one_se_alternative
        self._truth = truth
        self._sensitivity = sensitivity_improvement
        self._grouping = grouping
        self._features = features
        self._task = cfg.dataset.task

    def _reference_label(self) -> str:
        """Name the no-skill reference the paired improvement is measured against."""
        model = self._results[self._step.baseline_candidate].model
        if model == NAIVE_PREVALENCE_REFERENCE or self._task is TaskType.CLASSIFICATION:
            return f"naive training-prevalence predictor '{model}'"
        return f"naive training-mean predictor '{model}'"

    def _improvement(self, candidate: str, reference: str) -> tuple[float, float, float]:
        step = self._step
        return _relative_improvement(
            self._truth,
            self._results[candidate].out_of_fold_prediction(),
            self._results[reference].out_of_fold_prediction(),
            step.metric,
            step.bootstrap,
        )

    def decision(self) -> Decision:
        assert self._cfg.dataset is not None
        step = self._step
        champion = self._results[self._champion]
        reference = self._results[step.baseline_candidate]
        value, low, high = self._improvement(self._champion, step.baseline_candidate)
        if self._champion == step.linear_reference:
            linear_value, linear_low, linear_high = 0.0, 0.0, 0.0
        else:
            linear_value, linear_low, linear_high = self._improvement(
                self._champion, step.linear_reference
            )
        evidence = Evidence(
            baseline=(
                f"{self._reference_label()} on the identical "
                f"{step.cv.repeats}x{step.cv.folds} folds ({self._grouping})"
            ),
            delta=value,
            segments=self._segments(),
            sensitivity={
                f"champion_vs_linear_{step.metric}": linear_value,
                f"champion_vs_linear_{step.metric}_low": linear_low,
                f"champion_vs_linear_{step.metric}_high": linear_high,
                f"second_seed_improvement_seed{step.sensitivity_random_state}": self._sensitivity,
                f"champion_cv_{step.metric}_n{len(self._truth)}": champion.mean,
                f"naive_cv_{step.metric}_n{len(self._truth)}": reference.mean,
            },
            falsifier=(
                "the paired improvement interval's lower bound falls at or below the "
                "configured effect floor"
            ),
            provenance={
                "dataset": self._cfg.dataset.name,
                "target": TARGET_NOTE,
                "seed": str(step.cv.random_state),
                "folds": f"{step.cv.repeats}x{step.cv.folds} — {self._grouping}",
                "group_column": self._cfg.dataset.group_column or "none",
                "selection_rule": step.selection_rule,
                "rule_fired": self._rule_fired,
                "one_se_would_pick": self._one_se_alternative,
                "candidates": ", ".join(candidate.name for candidate in step.candidates),
                "champion_params": str(self._champion_params()),
                "resampling": "paired row bootstrap on out-of-fold predictions",
            },
        )
        return Decision.evaluate(
            rule=DecisionRule(
                metric=f"improvement_over_naive_{step.metric}",
                direction="max",
                min_effect=step.min_effect,
                min_sample_size=step.min_sample_size,
                action=step.action,
                owner=step.owner,
            ),
            value=value,
            interval=(low, high),
            sample_size=len(self._truth),
            question=(
                f"Does the selected {step.metric} champion beat the naive reference on "
                "identical repeated folds?"
            ),
            evidence=evidence,
            unit="relative_improvement",
        )

    def _champion_params(self) -> dict[str, float | int | str]:
        for candidate in self._step.candidates:
            if candidate.name == self._champion:
                return candidate.params
        raise ValueError(f"champion '{self._champion}' is not a configured candidate")

    def _segments(self) -> list[SegmentEvidence]:
        """Re-measure the improvement on two slices of the scored rows.

        A regression target splits at its median; a binary target has only two
        values, so the slice is the first plan feature's median (a dimension
        that could hide a failure) and is named from that column.
        """
        step = self._step
        prediction = self._results[self._champion].out_of_fold_prediction()
        reference_prediction = self._results[step.baseline_candidate].out_of_fold_prediction()
        if self._task is TaskType.CLASSIFICATION:
            slices = self._feature_slices()
        else:
            split = float(np.median(self._truth))
            slices = (
                ("low_target_half", self._truth <= split),
                ("high_target_half", self._truth > split),
            )
        segments: list[SegmentEvidence] = []
        for label, mask in slices:
            if int(mask.sum()) < 2:
                continue
            champion_error = _metric(self._truth[mask], prediction[mask], step.metric)
            reference_error = _metric(
                self._truth[mask], reference_prediction[mask], step.metric
            )
            segments.append(
                SegmentEvidence(
                    segment=label,
                    value=(reference_error - champion_error) / reference_error,
                    sample_size=int(mask.sum()),
                )
            )
        return segments

    def _feature_slices(self) -> tuple[tuple[str, np.ndarray], ...]:
        """Two halves of the first plan feature (classification target is binary)."""
        if self._features is None or self._features.empty:
            raise ValueError(
                "classification selection segments need the feature frame (pass "
                "features= to SelectionDecision)"
            )
        column = str(self._features.columns[0])
        values = self._features[column].to_numpy(dtype=float)
        split = float(np.nanmedian(values))
        return (
            (f"low_{column}", values <= split),
            (f"high_{column}", values > split),
        )


class SelectionRun:
    """Run every candidate, rank them, and publish the selection evidence."""

    def __init__(self, cfg: PipelineConfig) -> None:
        if cfg.selection is None or cfg.experiment is None:
            raise ValueError("selection step requires the selection and experiment config")
        self._cfg = cfg
        self._step: SelectionStep = cfg.selection
        self._frame: SelectionFrame | None = None

    def results(self) -> tuple[dict[str, CVResult], np.ndarray]:
        frame = SelectionFrame.from_config(self._cfg)
        self._frame = frame
        truth = frame.y.to_numpy(dtype=float)
        factories = CandidateSet(self._cfg, self._step.candidates).factories()
        cv = RepeatedCV(self._step.cv, self._step.metric, self._step.secondary_metric)
        results: dict[str, CVResult] = {}
        for name, factory in factories.items():
            with timed() as elapsed:
                results[name] = cv.run(name, factory, frame.X, frame.y, frame.groups)
            logger.info(
                "selection: %s %s=%.4f in %.2fs (%s)",
                name,
                self._step.metric,
                results[name].mean,
                elapsed(),
                results[name].grouping,
            )
        return results, truth

    def _intervals(self, results: dict[str, CVResult]) -> dict[str, tuple[float, float]]:
        return {name: result.interval(self._step.bootstrap) for name, result in results.items()}

    def _leaderboard(self, results: dict[str, CVResult]) -> MetricLeaderboard:
        frame = pd.DataFrame(
            [
                {
                    "candidate": name,
                    self._step.metric: result.mean,
                    self._step.secondary_metric: result.secondary_mean,
                }
                for name, result in results.items()
                if name != self._step.baseline_candidate
            ]
        )
        return MetricLeaderboard(
            frame,
            id_column="candidate",
            metrics={
                self._step.metric: MetricDirection(self._step.metric_direction),
                self._step.secondary_metric: MetricDirection(
                    self._step.secondary_direction
                ),
            },
        )

    def _one_se(
        self, results: dict[str, CVResult], leaderboard: MetricLeaderboard
    ) -> tuple[ChampionDecision, str]:
        decision = leaderboard.champion(self._step.metric, self._intervals(results))
        best = results[decision.champion]
        ceiling = best.mean + best.standard_error
        intervals = self._intervals(results)
        n_runs = len(results) - 1
        for candidate in self._step.candidates:
            if candidate.name == self._step.baseline_candidate:
                continue
            value = results[candidate.name].mean
            if value <= ceiling:
                low, high = intervals[candidate.name]
                selected = ChampionDecision(
                    metric=self._step.metric,
                    direction=MetricDirection(self._step.metric_direction),
                    champion=candidate.name,
                    value=value,
                    interval_low=min(low, value),
                    interval_high=max(high, value),
                    runner_up=decision.champion,
                    margin=value - best.mean,
                    n_runs=n_runs,
                )
                return selected, (
                    "one_se (simplest within one standard error; "
                    f"best {decision.champion} {best.mean:.4f}, "
                    f"ceiling {ceiling:.4f}, chosen {candidate.name} {value:.4f})"
                )
        return decision, "one_se (no simpler candidate inside one standard error)"

    def select(
        self, results: dict[str, CVResult]
    ) -> tuple[ChampionDecision, str, str]:
        """Apply the configured rule; also report what the one-SE rule would pick."""
        leaderboard = self._leaderboard(results)
        one_se, one_se_note = self._one_se(results, leaderboard)
        if self._step.selection_rule == "one_se":
            return one_se, one_se_note, one_se.champion
        decision = leaderboard.champion(self._step.metric, self._intervals(results))
        return decision, "point (leaderboard top)", one_se.champion

    def champion(self, results: dict[str, CVResult]) -> ChampionDecision:
        return self.select(results)[0]

    def sensitivity(self, results: dict[str, CVResult], truth: np.ndarray) -> float:
        """Re-run champion vs reference under a second fold seed (row-level)."""
        frame = SelectionFrame.from_config(self._cfg)
        factories = CandidateSet(self._cfg, self._step.candidates)
        geometry = self._step.cv.model_copy(
            update={"random_state": self._step.sensitivity_random_state}
        )
        cv = RepeatedCV(geometry, self._step.metric, self._step.secondary_metric)
        champion = self.champion(results).champion
        candidate = cv.run(
            champion, factories.reference(champion), frame.X, frame.y, frame.groups
        )
        reference = cv.run(
            self._step.baseline_candidate,
            factories.reference(self._step.baseline_candidate),
            frame.X,
            frame.y,
            frame.groups,
        )
        value, _, _ = _relative_improvement(
            truth,
            candidate.out_of_fold_prediction(),
            reference.out_of_fold_prediction(),
            self._step.metric,
            self._step.bootstrap,
        )
        return value

    def publish(
        self, results: dict[str, CVResult], truth: np.ndarray, sensitivity: float
    ) -> Decision:
        step = self._step
        selected, rule_fired, one_se = self.select(results)
        intervals = self._intervals(results)
        table = pd.DataFrame(
            [
                {
                    "candidate": name,
                    step.metric: result.mean,
                    f"{step.metric}_low": intervals[name][0],
                    f"{step.metric}_high": intervals[name][1],
                    f"{step.metric}_se": result.standard_error,
                    step.secondary_metric: result.secondary_mean,
                    "role": _role(name, step),
                    "folds": len(result.folds),
                }
                for name, result in results.items()
            ]
        )
        decision = SelectionDecision(
            self._cfg,
            step,
            results,
            selected.champion,
            rule_fired,
            one_se,
            truth,
            sensitivity,
            grouping=results[selected.champion].grouping,
            features=None if self._frame is None else self._frame.X,
        ).decision()
        output = StepOutput(
            results_dir=Path(step.output_dir),
            stem="selection",
            decision=decision,
            log=DecisionLog(tracking_uri=self._cfg.environment.mlflow_tracking_uri),
        )
        figures = self._leaderboard(results).figures()
        output.save_figure(figures[step.metric])
        figures[step.secondary_metric].savefig(
            Path(step.output_dir) / f"selection_{step.secondary_metric}.png", dpi=150
        )
        output.add_table("candidates", table)
        output.write()
        self._write_manifest(selected, rule_fired, results, decision)
        return decision

    def _write_manifest(
        self,
        selected: ChampionDecision,
        rule_fired: str,
        results: dict[str, CVResult],
        decision: Decision,
    ) -> None:
        step = self._step
        spec = next(
            candidate for candidate in step.candidates if candidate.name == selected.champion
        )
        record = ChampionRecord(
            champion=selected.champion,
            model=spec.model,
            params=spec.params,
            metric=step.metric,
            cv_mean=results[selected.champion].mean,
            interval_low=selected.interval_low,
            interval_high=selected.interval_high,
            selection_rule=step.selection_rule,
            rule_fired=rule_fired,
            vs_linear_improvement=float(
                decision.evidence.sensitivity[f"champion_vs_linear_{step.metric}"]
            ),
        )
        Path(step.output_file).write_text(
            json.dumps(record.model_dump(), indent=2) + "\n", encoding="utf-8"
        )


def _role(name: str, step: SelectionStep) -> str:
    if name == step.baseline_candidate:
        return "naive_reference"
    if name == step.linear_reference:
        return "linear_reference"
    return "candidate"


def run(cfg: PipelineConfig) -> None:
    if cfg.selection is None or cfg.experiment is None:
        raise ValueError("selection step requires the selection and experiment config")
    selection = SelectionRun(cfg)
    results, truth = selection.results()
    sensitivity = selection.sensitivity(results, truth)
    decision = selection.publish(results, truth, sensitivity)
    logger.info(
        "selection decision: %s — improvement %.4f [%.4f, %.4f]",
        decision.verdict,
        decision.value,
        decision.interval_low,
        decision.interval_high,
    )
