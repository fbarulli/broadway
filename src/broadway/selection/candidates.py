"""Candidate factories for model selection: registry models + the naive references.

Every candidate exposes ``fit(X, y) -> predict(X)``. Registry models are
composed through ``training.trainer.build_model_pipeline`` so selection scores
the exact pipeline the trainer deploys (preprocessing included, when an
experiment declares any). The references are not registry models: the
regression ``naive_mean`` predicts the training target mean and the
classification ``naive_prevalence`` predicts the training prevalence — the
no-skill values the champion must beat on the experiment's target metric.
"""

from __future__ import annotations

from collections.abc import Callable

from sklearn.base import BaseEstimator

from broadway.config.schema import (
    NAIVE_PREVALENCE_REFERENCE,
    NAIVE_REFERENCE,
    CandidateSpec,
    PipelineConfig,
    TaskType,
)
from broadway.evaluate.repeated_cv import NaiveMeanRegressor, NaivePrevalencePredictor
from broadway.training.trainer import build_model_pipeline


class CandidateSet:
    """Resolve configured candidates into fresh estimator factories."""

    def __init__(self, cfg: PipelineConfig, candidates: list[CandidateSpec]) -> None:
        if cfg.experiment is None:
            raise ValueError("model selection requires an experiment config")
        self._task = (
            TaskType.REGRESSION if cfg.dataset is None else cfg.dataset.task
        )
        self._require_task_reference(candidates)
        self._cfg = cfg
        self._candidates = candidates

    def _require_task_reference(self, candidates: list[CandidateSpec]) -> None:
        """Refuse the other task's naive reference (a classification run with naive_mean)."""
        regression = [c.name for c in candidates if c.model == NAIVE_REFERENCE]
        prevalence = [c.name for c in candidates if c.model == NAIVE_PREVALENCE_REFERENCE]
        if self._task is TaskType.CLASSIFICATION and regression:
            raise ValueError(
                "classification selection cannot use the regression naive_mean "
                f"reference ({regression}); use the no-skill prevalence candidate "
                f"(model: '{NAIVE_PREVALENCE_REFERENCE}')"
            )
        if self._task is TaskType.REGRESSION and prevalence:
            raise ValueError(
                "regression selection cannot use the classification naive_prevalence "
                f"reference ({prevalence}); use the naive training-mean candidate "
                f"(model: '{NAIVE_REFERENCE}')"
            )

    def factories(self) -> dict[str, Callable[[], BaseEstimator]]:
        return {candidate.name: self._factory(candidate) for candidate in self._candidates}

    def reference(self, name: str) -> Callable[[], BaseEstimator]:
        matches = [item for item in self._candidates if item.name == name]
        if not matches:
            raise ValueError(f"no candidate named '{name}' in the selection config")
        return self._factory(matches[0])

    def _factory(self, candidate: CandidateSpec) -> Callable[[], BaseEstimator]:
        if candidate.model == NAIVE_REFERENCE:
            return NaiveMeanRegressor
        if candidate.model == NAIVE_PREVALENCE_REFERENCE:
            return NaivePrevalencePredictor
        return lambda: build_model_pipeline(self._cfg, candidate.model, candidate.params)
