"""HPO step — tune the selected champion on training-only repeated CV.

The search space and trial budget come from the experiment's ``hpo`` block;
the search folds come from ``step/hpo.yaml`` and the report is measured on a
DISJOINT fold seed (``report_random_state``), so the tuned score is never read
on the partition the search optimized or the partition that selected the
champion. The holdout split is never touched here: the published
best-vs-default interval is the canonical paired row-level bootstrap on the
out-of-fold predictions.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.base import BaseEstimator

from broadway.config.schema import (
    TARGET_NOTE,
    BootstrapConfig,
    HpoStep,
    ModelHPOSpec,
    PipelineConfig,
)
from broadway.evaluate.metrics import MetricBootstrap, compute_metrics
from broadway.evaluate.repeated_cv import CVResult, RepeatedCV
from broadway.observability.profiling import timed
from broadway.reports.decision import Decision, DecisionRule, Evidence, SegmentEvidence
from broadway.reports.decision_log import DecisionLog
from broadway.reports.step_output import StepOutput
from broadway.selection.module import ChampionRecord, load_selection_frame
from broadway.training.hpo import suggest_params
from broadway.training.trainer import build_model_pipeline

logger = logging.getLogger(__name__)


class TuningResult(BaseModel):
    """Best trial plus the default it was measured against (next-stage handoff)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    best_params: dict[str, float | int | str]
    best_cv: float
    default_cv: float
    trials: int
    metric: str


class TuningDecision:
    """Publish best-vs-default on the HPO folds with a paired interval."""

    def __init__(
        self,
        cfg: PipelineConfig,
        step: HpoStep,
        best: CVResult,
        default: CVResult,
        truth: np.ndarray,
        trials: int,
        best_params: dict[str, float | int | str],
    ) -> None:
        assert cfg.dataset is not None and cfg.experiment is not None
        self._cfg = cfg
        self._step = step
        self._best = best
        self._default = default
        self._truth = truth
        self._trials = trials
        self._best_params = best_params

    def decision(self) -> Decision:
        assert self._cfg.dataset is not None
        step = self._step
        best_prediction = self._best.out_of_fold_prediction()
        default_prediction = self._default.out_of_fold_prediction()
        value, low, high = _improvement(
            self._truth, best_prediction, default_prediction, step.metric, step.bootstrap
        )
        default_value = float(
            compute_metrics(self._truth, default_prediction, metrics=(step.metric,))[step.metric]
        )
        level_90 = MetricBootstrap(
            step.bootstrap.model_copy(update={"level": 0.90})
        ).improvement_interval(
            self._truth, best_prediction, default_prediction, metric=step.metric
        )
        evidence = Evidence(
            baseline="the experiment's default params on the identical report folds",
            delta=value,
            segments=self._segments(),
            sensitivity={
                "level_90pct_low": level_90.lower / default_value,
                "level_90pct_high": level_90.upper / default_value,
                f"best_cv_{step.metric}_n{len(self._truth)}": self._best.mean,
                f"default_cv_{step.metric}_n{len(self._truth)}": self._default.mean,
            },
            falsifier=(
                "the paired best-vs-default interval's lower bound falls at or below "
                "the configured effect floor"
            ),
            provenance={
                "dataset": self._cfg.dataset.name,
                "target": TARGET_NOTE,
                "search_seed": str(step.cv.random_state),
                "report_seed": str(step.report_random_state),
                "folds": f"{step.cv.repeats}x{step.cv.folds} shuffled KFold on the report seed",
                "trials": str(self._trials),
                "model": self._best.model,
                "best_params": str(self._best_params),
                "resampling": "paired row bootstrap on out-of-fold predictions",
            },
        )
        return Decision.evaluate(
            rule=DecisionRule(
                metric=f"improvement_over_default_{step.metric}",
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
                "Do the HPO-best params beat the declared defaults on training-only "
                "repeated CV?"
            ),
            evidence=evidence,
            unit="relative_improvement",
        )

    def _segments(self) -> list[SegmentEvidence]:
        prediction = self._best.out_of_fold_prediction()
        default_prediction = self._default.out_of_fold_prediction()
        split = float(np.median(self._truth))
        segments: list[SegmentEvidence] = []
        for label, mask in (
            ("low_target_half", self._truth <= split),
            ("high_target_half", self._truth > split),
        ):
            if int(mask.sum()) < 2:
                continue
            best = float(
                compute_metrics(
                    self._truth[mask], prediction[mask], metrics=(self._step.metric,)
                )[self._step.metric]
            )
            default = float(
                compute_metrics(
                    self._truth[mask], default_prediction[mask], metrics=(self._step.metric,)
                )[self._step.metric]
            )
            segments.append(
                SegmentEvidence(
                    segment=label,
                    value=(default - best) / default,
                    sample_size=int(mask.sum()),
                )
            )
        return segments


class CvObjective:
    """Optuna objective: mean repeated-CV metric for one sampled param set."""

    def __init__(
        self,
        cfg: PipelineConfig,
        step: HpoStep,
        model: ModelHPOSpec,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        self._cfg = cfg
        self._step = step
        self._model = model
        self._X = X
        self._y = y
        self._cv = RepeatedCV(step.cv, step.metric, step.metric)

    def __call__(self, trial: optuna.Trial) -> float:
        params = suggest_params(self._model, trial)
        result = self._cv.run(
            self._model.name,
            _factory(self._cfg, self._model.name, params),
            self._X,
            self._y,
        )
        trial.set_user_attr("cv_mean", result.mean)
        return result.mean

    def evaluate(self, params: dict[str, float | int | str]) -> CVResult:
        return self._cv.run(
            self._model.name,
            _factory(self._cfg, self._model.name, params),
            self._X,
            self._y,
        )


def _factory(
    cfg: PipelineConfig, model: str, params: Mapping[str, float | int | str]
) -> Callable[[], BaseEstimator]:
    return lambda: build_model_pipeline(cfg, model, params)


class TuningRun:
    """Read the champion handoff, tune it, and publish the HPO evidence."""

    def __init__(self, cfg: PipelineConfig) -> None:
        if cfg.hpo is None or cfg.experiment is None:
            raise ValueError("hpo step requires the hpo and experiment config")
        if cfg.experiment.hpo is None:
            raise ValueError("hpo step requires experiment.hpo (search space + budget)")
        self._cfg = cfg
        self._step: HpoStep = cfg.hpo
        self._champion = ChampionRecord.model_validate_json(
            Path(self._step.selection_record).read_text(encoding="utf-8")
        )
        self._spec = self._model_spec()

    def _model_spec(self) -> ModelHPOSpec:
        assert self._cfg.experiment is not None and self._cfg.experiment.hpo is not None
        if self._cfg.experiment.model.type != self._champion.model:
            raise ValueError(
                f"experiment model '{self._cfg.experiment.model.type}' is not the "
                f"selected champion '{self._champion.model}'"
            )
        for spec in self._cfg.experiment.hpo.models:
            if spec.name == self._champion.model:
                return spec
        raise ValueError(
            f"experiment hpo has no search space for champion '{self._champion.model}'"
        )

    def run(self) -> tuple[CVResult, CVResult, int, dict[str, float | int | str]]:
        assert self._cfg.experiment is not None and self._cfg.experiment.hpo is not None
        X, y = load_selection_frame(self._cfg)
        objective = CvObjective(self._cfg, self._step, self._spec, X, y)
        study = optuna.create_study(
            direction=self._cfg.experiment.hpo.direction,
            sampler=optuna.samplers.TPESampler(seed=self._step.cv.random_state),
        )
        with timed() as elapsed:
            study.optimize(objective, n_trials=self._cfg.experiment.hpo.total_trials)
        logger.info(
            "hpo: %d trials, best %s=%.4f in %.1fs",
            len(study.trials),
            self._step.metric,
            study.best_value,
            elapsed(),
        )
        best_params = dict(study.best_params)
        # Report on a fold seed the search never optimized against.
        report_geometry = self._step.cv.model_copy(
            update={"random_state": self._step.report_random_state}
        )
        report = RepeatedCV(report_geometry, self._step.metric, self._step.metric)
        best = report.run(
            self._spec.name, _factory(self._cfg, self._spec.name, best_params), X, y
        )
        default = report.run(
            self._spec.name,
            _factory(self._cfg, self._spec.name, self._cfg.experiment.model.params),
            X,
            y,
        )
        return best, default, len(study.trials), best_params

    def publish(
        self,
        best: CVResult,
        default: CVResult,
        trials: int,
        best_params: dict[str, float | int | str],
    ) -> Decision:
        step = self._step
        _, y = load_selection_frame(self._cfg)
        truth = y.to_numpy(dtype=float)
        decision = TuningDecision(
            self._cfg, step, best, default, truth, trials, best_params
        ).decision()
        output = StepOutput(
            results_dir=Path(step.output_dir),
            stem="hpo",
            decision=decision,
            log=DecisionLog(tracking_uri=self._cfg.environment.mlflow_tracking_uri),
        )
        output.save_figure(_figure(best, default, step.metric))
        output.add_table(
            "folds",
            pd.DataFrame(
                [
                    {"params": "best", "fold": index, "value": fold.values[step.metric]}
                    for index, fold in enumerate(best.folds)
                ]
                + [
                    {"params": "default", "fold": index, "value": fold.values[step.metric]}
                    for index, fold in enumerate(default.folds)
                ]
            ),
        )
        output.write()
        Path(step.output_file).write_text(
            json.dumps(
                TuningResult(
                    model=best.model,
                    best_params=best_params,
                    best_cv=best.mean,
                    default_cv=default.mean,
                    trials=trials,
                    metric=step.metric,
                ).model_dump(),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return decision


def _improvement(
    truth: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    metric: str,
    bootstrap: BootstrapConfig,
) -> tuple[float, float, float]:
    """Paired row-level improvement of ``candidate`` over ``reference`` on the OOF rows."""
    candidate_value = float(compute_metrics(truth, candidate, metrics=(metric,))[metric])
    reference_value = float(compute_metrics(truth, reference, metrics=(metric,))[metric])
    if reference_value <= 0.0:
        raise ValueError(f"reference {metric} must be positive for a relative delta")
    interval = MetricBootstrap(bootstrap).improvement_interval(
        truth, candidate, reference, metric=metric
    )
    value = (reference_value - candidate_value) / reference_value
    return value, interval.lower / reference_value, interval.upper / reference_value


def _figure(best: CVResult, default: CVResult, metric: str) -> object:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
    values = [default.mean, best.mean]
    bars = axis.bar(["default params", "HPO best"], values, color=["#9ecae1", "#08519c"])
    for bar in bars:
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():,.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.set_ylabel(metric)
    axis.set_title("HPO best vs defaults — training repeated CV")
    return figure


def run(cfg: PipelineConfig) -> None:
    if cfg.hpo is None or cfg.experiment is None:
        raise ValueError("hpo step requires the hpo and experiment config")
    tuning = TuningRun(cfg)
    best, default, trials, best_params = tuning.run()
    decision = tuning.publish(best, default, trials, best_params)
    logger.info(
        "hpo decision: %s — best-vs-default %.4f [%.4f, %.4f]",
        decision.verdict,
        decision.value,
        decision.interval_low,
        decision.interval_high,
    )
