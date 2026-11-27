"""Candidate factories for model selection: registry models + the naive mean.

Every candidate exposes ``fit(X, y) -> predict(X)``. Registry models are
composed through ``training.trainer.build_model_pipeline`` so selection scores
the exact pipeline the trainer deploys (preprocessing included, when an
experiment declares any). The naive mean is not a registry model: it predicts
the training target mean for every row — the reference the champion must beat.
"""

from __future__ import annotations

from collections.abc import Callable

from sklearn.base import BaseEstimator

from broadway.config.schema import NAIVE_REFERENCE, CandidateSpec, PipelineConfig, TaskType
from broadway.evaluate.repeated_cv import NaiveMeanRegressor
from broadway.training.trainer import build_model_pipeline

_CLASSIFICATION_REFERENCE = "naive_prevalence"


class CandidateSet:
    """Resolve configured candidates into fresh estimator factories."""

    def __init__(self, cfg: PipelineConfig, candidates: list[CandidateSpec]) -> None:
        if cfg.experiment is None:
            raise ValueError("model selection requires an experiment config")
        if cfg.dataset is not None and cfg.dataset.task is TaskType.CLASSIFICATION:
            naive = [
                candidate.name
                for candidate in candidates
                if candidate.model == NAIVE_REFERENCE
            ]
            if naive:
                raise ValueError(
                    "classification selection cannot use the regression naive_mean "
                    f"reference ({naive}); replace it with the no-skill prevalence "
                    f"candidate (model: '{_CLASSIFICATION_REFERENCE}') — the "
                    "classification reference is the next unit"
                )
        self._cfg = cfg
        self._candidates = candidates

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
        return lambda: build_model_pipeline(self._cfg, candidate.model, candidate.params)
