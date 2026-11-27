"""Repeated shuffled K-fold CV for cross-sectional (row-shuffled) fitting.

A time split gets the forward-chaining geometry in ``evaluate.validation``;
this owner is the other case: rows are exchangeable survey respondents, so a
shuffled KFold repeated with independent seeds is the honest estimator. Every
candidate is scored on the *same* folds (same seed, same n), and the paired
improvement interval is measured on the out-of-fold rows by the canonical
``MetricBootstrap`` — never on the (correlated) fold scores.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold

from broadway.config.schema import CVGeometry
from broadway.evaluate.contracts import BootstrapConfig
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

    @property
    def standard_error(self) -> float:
        """One standard error of the mean fold score (the one-SE selection unit)."""
        scores = np.asarray([fold.values[self.metric] for fold in self.folds], dtype=float)
        return float(np.std(scores, ddof=1) / np.sqrt(scores.size))

    def interval(self, bootstrap: BootstrapConfig) -> tuple[float, float]:
        """Fold-score interval for the candidate's own CV score (optimistic)."""
        draws = self._bootstrap(
            np.asarray([fold.values[self.metric] for fold in self.folds], dtype=float),
            bootstrap,
        )
        tail = (1.0 - bootstrap.level) / 2.0
        return (
            float(np.percentile(draws, tail * 100.0)),
            float(np.percentile(draws, (1.0 - tail) * 100.0)),
        )

    def out_of_fold_prediction(self) -> np.ndarray:
        """Row-level OOF prediction, averaged over the independent repeats."""
        return np.mean(
            np.asarray([self.out_of_fold[repeat] for repeat in sorted(self.out_of_fold)]),
            axis=0,
        )

    @staticmethod
    def _bootstrap(values: np.ndarray, bootstrap: BootstrapConfig) -> np.ndarray:
        generator = np.random.default_rng(bootstrap.seed)
        draws = np.empty(bootstrap.iterations)
        for index in range(bootstrap.iterations):
            sample = generator.integers(0, values.size, size=values.size)
            draws[index] = float(np.mean(values[sample]))
        return draws


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
