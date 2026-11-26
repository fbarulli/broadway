"""Data-quality flags: outliers, class imbalance, constant and duplicate rows.

Every flag is evidence with the threshold logged beside it — the report never
silently drops or rewrites a row.
"""

from __future__ import annotations

import math

import pandas as pd

from broadway.config.schema import TaskType
from broadway.eda.models import (
    CategoryFrequency,
    ImbalanceStats,
    OutlierStats,
    QualityResult,
)


class QualityAnalyzer:
    """Flags outliers (IQR + z-score), constants, duplicates, class imbalance."""

    def __init__(
        self,
        iqr_multiplier: float,
        z_threshold: float,
        imbalance_min_share: float,
    ) -> None:
        self._iqr_multiplier = iqr_multiplier
        self._z_threshold = z_threshold
        self._imbalance_min_share = imbalance_min_share

    def analyze(
        self,
        frame: pd.DataFrame,
        task: TaskType,
        target: str,
    ) -> QualityResult:
        numeric = [
            name
            for name in frame.columns
            if not pd.api.types.is_bool_dtype(frame[name])
            and pd.api.types.is_numeric_dtype(frame[name])
        ]
        duplicates = int(frame.duplicated().sum())
        return QualityResult(
            duplicate_rows=duplicates,
            duplicate_rate=self._rate(duplicates, len(frame)),
            constant_columns=[
                str(name)
                for name in frame.columns
                if int(frame[name].nunique(dropna=False)) <= 1
            ],
            outliers=[
                self._outliers(frame[str(name)], str(name))
                for name in numeric
                if frame[name].notna().sum() > 0
            ],
            imbalance=self._imbalance(frame, task, target),
        )

    def _outliers(self, column: pd.Series, name: str) -> OutlierStats:
        values = column.dropna().astype(float)
        count = int(values.size)
        q25 = float(values.quantile(0.25))
        q75 = float(values.quantile(0.75))
        spread = q75 - q25
        lower = q25 - self._iqr_multiplier * spread
        upper = q75 + self._iqr_multiplier * spread
        iqr_count = int(((values < lower) | (values > upper)).sum())
        mean = float(values.mean())
        std = float(values.std())
        z_count = (
            int(((values - mean).abs() > self._z_threshold * std).sum()) if std > 0 else 0
        )
        return OutlierStats(
            column=name,
            iqr_count=iqr_count,
            iqr_rate=self._rate(iqr_count, count),
            iqr_lower=self._finite(lower),
            iqr_upper=self._finite(upper),
            z_count=z_count,
            z_rate=self._rate(z_count, count),
        )

    def _imbalance(
        self, frame: pd.DataFrame, task: TaskType, target: str
    ) -> ImbalanceStats:
        if task is not TaskType.CLASSIFICATION:
            return ImbalanceStats(
                applicable=False,
                reason=f"task is {task.value}; class imbalance applies to classification only",
                target=target,
                value_counts=[],
                minority_share=None,
                imbalance_ratio=None,
                flagged=None,
            )
        if target not in frame.columns:
            return ImbalanceStats(
                applicable=False,
                reason=f"target column '{target}' not present in the profiled frame",
                target=target,
                value_counts=[],
                minority_share=None,
                imbalance_ratio=None,
                flagged=None,
            )
        counts = frame[target].value_counts(dropna=False)
        total = int(counts.sum())
        frequencies = [
            CategoryFrequency(
                value=str(value),
                count=int(count),
                share=self._rate(int(count), total),
            )
            for value, count in counts.items()
        ]
        minority = int(counts.min())
        majority = int(counts.max())
        minority_share = self._rate(minority, total)
        ratio = round(majority / minority, 6) if minority else None
        return ImbalanceStats(
            applicable=True,
            reason=None,
            target=target,
            value_counts=frequencies,
            minority_share=minority_share,
            imbalance_ratio=ratio,
            flagged=minority_share < self._imbalance_min_share,
        )

    @staticmethod
    def _finite(value: float) -> float | None:
        return value if math.isfinite(value) else None

    @staticmethod
    def _rate(part: int, whole: int) -> float:
        return round(part / whole, 6) if whole else 0.0
