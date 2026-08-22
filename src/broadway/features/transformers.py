"""Sklearn-compatible smoothed target and frequency encoding transformers.

``TargetEncoding`` and ``FrequencyEncoding`` reproduce the exact smoothed
formulas and output column names of the legacy free functions in
``broadway.features.encodings`` (kept as the golden reference) behind the
standard estimator ``fit``/``transform`` interface, so they can be composed
into sklearn Pipelines. Transform always returns a copy — inputs are never
mutated.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _key_series(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Per-row category key: the single column value, or a tuple across columns."""
    if len(columns) == 1:
        return frame[columns[0]]
    return pd.Series(list(zip(*[frame[col] for col in columns])), index=frame.index)


class TargetEncoding(BaseEstimator, TransformerMixin):
    """Smoothed target encoding keyed by one categorical column or a composite.

    Fit computes ``(count * mean + smoothing * global_mean) / (count + smoothing)``
    per key with ``global_mean = X[target].mean()`` — identical to the legacy
    ``fit_target_encoding``. pandas ``groupby`` drops NaN keys at fit, so NaN
    categories resolve to the global-mean fallback at transform (the legacy
    ``__unknown__`` sentinel equivalent). The output column is named
    ``feature_name`` when provided (verbatim), else ``<joined_cols>_target_enc``.
    """

    def __init__(
        self,
        columns: list[str],
        target: str,
        smoothing: float,
        feature_name: str | None = None,
    ) -> None:
        self.columns = list(columns)
        self.target = target
        self.smoothing = smoothing
        self.feature_name = feature_name

    def __sklearn_clone__(self) -> TargetEncoding:
        return type(self)(**self.get_params(deep=False))

    def fit(self, X: pd.DataFrame, y=None) -> TargetEncoding:
        key = self.columns if len(self.columns) > 1 else self.columns[0]
        stats = X.groupby(key)[self.target].agg(["mean", "count"])
        encoded = (stats["count"] * stats["mean"] + self.smoothing * X[self.target].mean()) / (
            stats["count"] + self.smoothing
        )
        self._mapping = encoded.to_dict()
        self._global_mean = float(X[self.target].mean())
        self._column_name = (
            self.feature_name
            if self.feature_name is not None
            else f"{'_'.join(self.columns)}_target_enc"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        values = _key_series(X, self.columns).map(self._mapping).fillna(self._global_mean)
        result = X.copy()
        result[self._column_name] = values
        return result


class FrequencyEncoding(BaseEstimator, TransformerMixin):
    """Relative-frequency encoding keyed by one categorical column or a composite.

    Fit computes ``value_counts`` per key — normalized by default (identical to
    the legacy ``fit_frequency_encoding``); pass ``normalize=False`` for raw
    counts (the taxi binding's ``route_frequency``). Unseen/NaN keys at
    transform map to the ``fill`` value (default 0, matching the legacy
    signature). The output column is named ``feature_name`` when provided
    (verbatim), else ``<joined_cols>_freq_enc``.
    """

    def __init__(
        self,
        columns: list[str],
        feature_name: str | None = None,
        normalize: bool = True,
    ) -> None:
        self.columns = list(columns)
        self.feature_name = feature_name
        self.normalize = normalize

    def __sklearn_clone__(self) -> FrequencyEncoding:
        return type(self)(**self.get_params(deep=False))

    def fit(self, X: pd.DataFrame, y=None) -> FrequencyEncoding:
        key = self.columns if len(self.columns) > 1 else self.columns[0]
        self._mapping = X[key].value_counts(normalize=self.normalize).to_dict()
        self._column_name = (
            self.feature_name
            if self.feature_name is not None
            else f"{'_'.join(self.columns)}_freq_enc"
        )
        return self

    def transform(self, X: pd.DataFrame, fill: float = 0) -> pd.DataFrame:
        values = _key_series(X, self.columns).map(self._mapping).fillna(fill)
        result = X.copy()
        result[self._column_name] = values
        return result
