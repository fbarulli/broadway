"""Column summary: distributions, skew/kurtosis, cardinality, memory usage."""

from __future__ import annotations

import math
from typing import Literal

import pandas as pd

from broadway.eda.models import (
    CategoricalStats,
    CategoryFrequency,
    ColumnSummary,
    NumericStats,
    SummaryResult,
    TemporalStats,
)


class SummaryAnalyzer:
    """Profiles one already-sampled frame column by column.

    Roles come from the dataset contract when a column is declared there;
    undeclared (joined/derived) columns are reported as ``undeclared``.
    """

    def __init__(self, max_categories: int) -> None:
        self._max_categories = max_categories

    def analyze(
        self, frame: pd.DataFrame, roles: dict[str, str] | None = None
    ) -> SummaryResult:
        roles = roles or {}
        detail: list[ColumnSummary] = []
        numeric = categorical = temporal = 0
        for name in frame.columns:
            column = frame[name]
            kind = self._kind(column)
            if kind == "numeric":
                numeric += 1
            elif kind == "temporal":
                temporal += 1
            else:
                categorical += 1
            detail.append(
                ColumnSummary(
                    name=str(name),
                    dtype=str(column.dtype),
                    role=roles.get(str(name), "undeclared"),
                    kind=kind,
                    memory_bytes=int(column.memory_usage(deep=True)),
                    numeric=self._numeric(column) if kind == "numeric" else None,
                    categorical=self._categorical(column) if kind == "categorical" else None,
                    temporal=self._temporal(column) if kind == "temporal" else None,
                )
            )
        return SummaryResult(
            rows=len(frame),
            columns=int(frame.shape[1]),
            total_memory_bytes=int(frame.memory_usage(deep=True).sum()),
            numeric_columns=numeric,
            categorical_columns=categorical,
            temporal_columns=temporal,
            detail=detail,
        )

    @staticmethod
    def _kind(column: pd.Series) -> Literal["numeric", "categorical", "temporal"]:
        if pd.api.types.is_bool_dtype(column):
            return "categorical"
        if pd.api.types.is_numeric_dtype(column):
            return "numeric"
        if pd.api.types.is_datetime64_any_dtype(column):
            return "temporal"
        return "categorical"

    def _numeric(self, column: pd.Series) -> NumericStats:
        values = column.dropna()
        count = int(values.size)
        quantiles = values.quantile([0.25, 0.5, 0.75]) if count else None
        return NumericStats(
            count=count,
            mean=self._float(values.mean()),
            std=self._float(values.std()) if count > 1 else None,
            minimum=self._float(values.min()),
            q25=self._float(quantiles.loc[0.25]) if quantiles is not None else None,
            median=self._float(quantiles.loc[0.5]) if quantiles is not None else None,
            q75=self._float(quantiles.loc[0.75]) if quantiles is not None else None,
            maximum=self._float(values.max()),
            skew=self._float(values.skew()) if count > 2 else None,
            kurtosis=self._float(values.kurt()) if count > 3 else None,
            zero_count=int((values == 0).sum()),
            negative_count=int((values < 0).sum()),
        )

    def _categorical(self, column: pd.Series) -> CategoricalStats:
        values = column.dropna()
        count = int(values.size)
        counts = values.value_counts(dropna=True).head(self._max_categories)
        return CategoricalStats(
            count=count,
            unique=int(values.nunique(dropna=True)),
            top_values=[
                CategoryFrequency(
                    value=str(value),
                    count=int(frequency),
                    share=self._share(int(frequency), count),
                )
                for value, frequency in counts.items()
            ],
        )

    @staticmethod
    def _temporal(column: pd.Series) -> TemporalStats:
        values = column.dropna()
        return TemporalStats(
            count=int(values.size),
            unique=int(values.nunique(dropna=True)),
            minimum=values.min().isoformat() if not values.empty else None,
            maximum=values.max().isoformat() if not values.empty else None,
        )

    @staticmethod
    def _float(value: float) -> float | None:
        number = float(value)
        return number if math.isfinite(number) else None

    @staticmethod
    def _share(part: int, whole: int) -> float:
        return round(part / whole, 6) if whole else 0.0
