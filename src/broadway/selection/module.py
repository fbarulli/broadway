"""Model selection — repeated shuffled CV, paired improvement, one champion.

Cross-sectional survey rows are exchangeable, so the folds are a repeated
shuffled KFold (never the forward-chaining time geometry). Every configured
candidate and the naive mean reference score on the identical folds; the
champion is the leaderboard's best on the primary metric and the published
decision is its paired improvement over the naive reference, with a 95%
interval and a verdict computed by ``Decision.evaluate`` against the
config-declared effect floor. The champion manifest is the handoff the HPO
step tunes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from broadway.config.schema import PipelineConfig, SelectionStep
from broadway.data.loader import ProcessedStore
from broadway.evaluate.leaderboard import ChampionDecision, MetricDirection, MetricLeaderboard
from broadway.evaluate.metrics import compute_metrics
from broadway.evaluate.repeated_cv import CVResult, RepeatedCV
from broadway.features.generic import validate_engineered_frame
from broadway.features.plan import ColumnPlan
from broadway.observability.profiling import timed
from broadway.reports.decision import Decision, DecisionRule, Evidence, SegmentEvidence
from broadway.reports.decision_log import DecisionLog
from broadway.reports.step_output import StepOutput
from broadway.selection.candidates import CandidateSet

logger = logging.getLogger(__name__)


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
    baseline: str
    baseline_cv_mean: float


def load_selection_frame(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.Series]:
    """Read the persisted train features through the experiment's column plan."""
    assert cfg.dataset is not None and cfg.experiment is not None and cfg.etl is not None
    out_dir = ProcessedStore.from_config(cfg).directory
    frame = pd.read_parquet(out_dir / cfg.etl.train_features_file)
    validate_engineered_frame(cfg, frame)
    plan = ColumnPlan.from_config(cfg.dataset, cfg.experiment)
    X, _ = plan.apply(frame)
    return X, frame[cfg.dataset.target]


class SelectionDecision:
    """Publish the champion's paired improvement over the naive reference.

    The interval is a bootstrap over the paired per-fold deltas of the SAME
    folds (positive = the champion's error is lower). Segments re-measure the
    improvement on the low/high halves of the observed target — a slice that
    would expose a champion that only wins for satisfied respondents — and the
    sensitivity re-runs the pair under a second fold seed.
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        step: SelectionStep,
        results: dict[str, CVResult],
        champion: str,
        truth: np.ndarray,
        sensitivity_improvement: float,
    ) -> None:
        assert cfg.dataset is not None
        self._cfg = cfg
        self._step = step
        self._results = results
        self._champion = champion
        self._truth = truth
        self._sensitivity = sensitivity_improvement

    def decision(self) -> Decision:
        assert self._cfg.dataset is not None
        step = self._step
        champion = self._results[self._champion]
        reference = self._results[step.baseline_candidate]
        value, low, high = champion.paired_improvement(reference, step.bootstrap)
        linear = self._results[step.linear_reference]
        linear_value, _, _ = linear.paired_improvement(reference, step.bootstrap)
        evidence = Evidence(
            baseline=(
                f"naive training-mean predictor '{reference.model}' on the identical "
                f"{step.cv.repeats}x{step.cv.folds} shuffled folds"
            ),
            delta=value,
            segments=self._segments(champion, reference),
            sensitivity={
                f"second_seed_improvement_seed{step.sensitivity_random_state}": self._sensitivity,
                f"linear_reference_improvement_{step.metric}": linear_value,
                f"champion_cv_{step.metric}_n{len(self._truth)}": champion.mean,
                f"naive_cv_{step.metric}_n{len(self._truth)}": reference.mean,
            },
            falsifier=(
                "the paired improvement interval's lower bound falls at or below the "
                "configured effect floor"
            ),
            provenance={
                "dataset": self._cfg.dataset.name,
                "seed": str(step.cv.random_state),
                "folds": f"{step.cv.repeats}x{step.cv.folds} shuffled KFold",
                "candidates": ", ".join(candidate.name for candidate in step.candidates),
                "champion_params": str(self._champion_params()),
                "resampling": "paired fold bootstrap",
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
            interval=(min(low, value), max(high, value)),
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

    def _segments(self, champion: CVResult, reference: CVResult) -> list[SegmentEvidence]:
        """Re-measure the improvement on the low and high target halves."""
        prediction, reference_prediction = champion.paired_rows(reference)
        split = float(np.median(self._truth))
        segments: list[SegmentEvidence] = []
        for label, mask in (
            ("low_target_half", self._truth <= split),
            ("high_target_half", self._truth > split),
        ):
            if int(mask.sum()) < 2:
                continue
            champion_error = _metric(self._truth[mask], prediction[mask], champion.metric)
            reference_error = _metric(
                self._truth[mask], reference_prediction[mask], champion.metric
            )
            segments.append(
                SegmentEvidence(
                    segment=label,
                    value=(reference_error - champion_error) / reference_error,
                    sample_size=int(mask.sum()),
                )
            )
        return segments


def _metric(truth: np.ndarray, prediction: np.ndarray, metric: str) -> float:
    return float(compute_metrics(truth, prediction, metrics=(metric,))[metric])


class SelectionRun:
    """Run every candidate, rank them, and publish the selection evidence."""

    def __init__(self, cfg: PipelineConfig) -> None:
        if cfg.selection is None or cfg.experiment is None:
            raise ValueError("selection step requires the selection and experiment config")
        self._cfg = cfg
        self._step: SelectionStep = cfg.selection

    def results(self) -> tuple[dict[str, CVResult], np.ndarray]:
        X, y = load_selection_frame(self._cfg)
        truth = y.to_numpy(dtype=float)
        factories = CandidateSet(self._cfg, self._step.candidates).factories()
        cv = RepeatedCV(self._step.cv, self._step.metric, self._step.secondary_metric)
        results: dict[str, CVResult] = {}
        for name, factory in factories.items():
            with timed() as elapsed:
                results[name] = cv.run(name, factory, X, y)
            logger.info(
                "selection: %s %s=%.4f in %.2fs",
                name,
                self._step.metric,
                results[name].mean,
                elapsed(),
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
                self._step.metric: MetricDirection.MINIMIZE,
                self._step.secondary_metric: MetricDirection.MINIMIZE,
            },
        )

    def champion(self, results: dict[str, CVResult]) -> ChampionDecision:
        return self._leaderboard(results).champion(self._step.metric, self._intervals(results))

    def sensitivity(self, results: dict[str, CVResult]) -> float:
        """Re-run champion vs reference under a second fold seed."""
        X, y = load_selection_frame(self._cfg)
        factories = CandidateSet(self._cfg, self._step.candidates)
        geometry = self._step.cv.model_copy(
            update={"random_state": self._step.sensitivity_random_state}
        )
        cv = RepeatedCV(geometry, self._step.metric, self._step.secondary_metric)
        champion = self.champion(results).champion
        candidate = cv.run(champion, factories.reference(champion), X, y)
        reference = cv.run(
            self._step.baseline_candidate,
            factories.reference(self._step.baseline_candidate),
            X,
            y,
        )
        improvement, _, _ = candidate.paired_improvement(reference, self._step.bootstrap)
        return improvement

    def publish(
        self, results: dict[str, CVResult], truth: np.ndarray, sensitivity: float
    ) -> Decision:
        step = self._step
        selected = self.champion(results)
        intervals = self._intervals(results)
        table = pd.DataFrame(
            [
                {
                    "candidate": name,
                    step.metric: result.mean,
                    f"{step.metric}_low": intervals[name][0],
                    f"{step.metric}_high": intervals[name][1],
                    step.secondary_metric: result.secondary_mean,
                    "role": _role(name, step),
                    "folds": len(result.folds),
                }
                for name, result in results.items()
            ]
        )
        decision = SelectionDecision(
            self._cfg, step, results, selected.champion, truth, sensitivity
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
        self._write_manifest(selected, results)
        return decision

    def _write_manifest(self, selected: ChampionDecision, results: dict[str, CVResult]) -> None:
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
            baseline=step.baseline_candidate,
            baseline_cv_mean=results[step.baseline_candidate].mean,
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
    sensitivity = selection.sensitivity(results)
    decision = selection.publish(results, truth, sensitivity)
    logger.info(
        "selection decision: %s — improvement %.4f [%.4f, %.4f]",
        decision.verdict,
        decision.value,
        decision.interval_low,
        decision.interval_high,
    )
