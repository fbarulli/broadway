"""Repeated shuffled K-fold CV for cross-sectional (row-shuffled) fitting.

A time split gets the forward-chaining geometry in ``evaluate.validation``;
this owner is the other case: rows are exchangeable survey respondents, so a
shuffled KFold repeated with independent seeds is the honest estimator. Every
candidate is scored on the *same* folds (same seed, same n), which is what
makes the paired improvement interval below a like-for-like comparison.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold

from broadway.config.schema import BootstrapSpec, CVGeometry
from broadway.evaluate.metrics import compute_metrics


class FoldScore(BaseModel):
    """One fold's measured metrics, with the rows it was scored on."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repeat: int
    fold: int
    sample_size: int
    values: dict[str, float]


class CVResult(BaseModel):
    """Every fold score of one candidate plus its out-of-fold predictions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    metric: str
    secondary_metric: str
    geometry: CVGeometry
    folds: list[FoldScore]
    out_of_fold: dict[int, list[float]] = Field(exclude=True, repr=False)

    @property
    def mean(self) -> float:
        return float(np.mean([fold.values[self.metric] for fold in self.folds]))

    @property
    def secondary_mean(self) -> float:
        return float(np.mean([fold.values[self.secondary_metric] for fold in self.folds]))

    def interval(self, bootstrap: BootstrapSpec) -> tuple[float, float]:
        """Percentile bootstrap over fold scores (folds are correlated: optimistic)."""
        draws = self._bootstrap(
            np.asarray([fold.values[self.metric] for fold in self.folds], dtype=float),
            bootstrap,
        )
        return _percentile(draws, bootstrap.level)

    def paired_improvement(
        self, reference: CVResult, bootstrap: BootstrapSpec
    ) -> tuple[float, float, float]:
        """Relative paired improvement over ``reference`` on identical folds.

        Positive means this candidate's error is lower. The fold pairs are
        bootstrapped as units because the same fold rows score both models.
        """
        relative, _ = self._paired(reference)
        draws = self._bootstrap(relative, bootstrap)
        low, high = _percentile(draws, bootstrap.level)
        return float(np.mean(relative)), low, high

    def paired_rows(self, reference: CVResult) -> tuple[np.ndarray, np.ndarray]:
        """Out-of-fold truth/prediction arrays for the primary repeat pair."""
        repeat = min(self.out_of_fold)
        prediction = np.asarray(self.out_of_fold[repeat], dtype=float)
        reference_prediction = np.asarray(reference.out_of_fold[repeat], dtype=float)
        return prediction, reference_prediction

    def _paired(
        self, reference: CVResult
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        mine = {(fold.repeat, fold.fold): fold.values[self.metric] for fold in self.folds}
        theirs = {
            (fold.repeat, fold.fold): fold.values[self.metric] for fold in reference.folds
        }
        keys = sorted(mine)
        if keys != sorted(theirs):
            raise ValueError(
                f"candidate '{self.model}' and reference '{reference.model}' were "
                "scored on different folds — the paired interval is undefined"
            )
        mine_values = np.asarray([mine[key] for key in keys], dtype=float)
        their_values = np.asarray([theirs[key] for key in keys], dtype=float)
        if np.any(their_values <= 0.0):
            raise ValueError("reference fold scores must be positive for a relative delta")
        return (their_values - mine_values) / their_values, keys

    @staticmethod
    def _bootstrap(values: np.ndarray, bootstrap: BootstrapSpec) -> np.ndarray:
        generator = np.random.default_rng(bootstrap.seed)
        draws = np.empty(bootstrap.iterations)
        for index in range(bootstrap.iterations):
            sample = generator.integers(0, values.size, size=values.size)
            draws[index] = float(np.mean(values[sample]))
        return draws


def _percentile(draws: np.ndarray, level: float) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    return (
        float(np.percentile(draws, tail * 100.0)),
        float(np.percentile(draws, (1.0 - tail) * 100.0)),
    )


class RepeatedCV:
    """Score one model factory over ``repeats x folds`` shuffled partitions."""

    def __init__(self, geometry: CVGeometry, metric: str, secondary_metric: str) -> None:
        self._geometry = geometry
        self._metric = metric
        self._secondary = secondary_metric

    def run(
        self,
        model: str,
        factory: Callable[[], BaseEstimator],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> CVResult:
        truth = y.to_numpy(dtype=float)
        scores: list[FoldScore] = []
        out_of_fold: dict[int, list[float]] = {}
        for repeat in range(self._geometry.repeats):
            splitter = KFold(
                n_splits=self._geometry.folds,
                shuffle=True,
                random_state=self._geometry.random_state + repeat,
            )
            prediction = np.full(truth.shape, np.nan)
            for fold, (train_index, test_index) in enumerate(splitter.split(X)):
                estimator = factory()
                estimator.fit(X.iloc[train_index], truth[train_index])
                fold_prediction = np.asarray(
                    estimator.predict(X.iloc[test_index]), dtype=float
                )
                prediction[test_index] = fold_prediction
                metrics = compute_metrics(
                    truth[test_index],
                    fold_prediction,
                    metrics=(self._metric, self._secondary),
                )
                scores.append(
                    FoldScore(
                        repeat=repeat,
                        fold=fold,
                        sample_size=len(test_index),
                        values=metrics,
                    )
                )
            out_of_fold[repeat] = prediction.tolist()
        if any(np.isnan(values).any() for values in (np.asarray(p) for p in out_of_fold.values())):
            raise ValueError(f"repeated CV left rows unscored for model '{model}'")
        return CVResult(
            model=model,
            metric=self._metric,
            secondary_metric=self._secondary,
            geometry=self._geometry,
            folds=scores,
            out_of_fold=out_of_fold,
        )


class NaiveMeanRegressor(BaseEstimator, RegressorMixin):
    """Predict the fitted training-mean for every row — the naive reference."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> NaiveMeanRegressor:
        self.mean_ = float(np.mean(np.asarray(y, dtype=float)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_)
