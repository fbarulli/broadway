"""Semantic segments every EDA decision is re-measured inside.

Row-order halves of an already-shuffled sample carry no independent information:
both halves are draws from the same sample, so they can only confirm it. The
segment that can hide a failure is the slice the domain names — weekday versus
weekend when a datetime column is declared, the levels of a declared
low-cardinality column otherwise. A plan is a *rule*, not row positions: the
same rule labels the pre-ETL frame and the post-ETL sample, so a comparison
re-measures the same slice on both sides. When no semantic column exists the
plan says so and publishes one "all rows" segment rather than fake a split.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

_ALL_ROWS = "all rows"
_MIN_SEGMENT_ROWS = 2
PlanKind = Literal["datetime", "levels", "none"]


class SegmentPlan:
    """A named row-slicing rule shared by the tests, outliers and the comparison."""

    def __init__(self, kind: PlanKind, column: str | None, rule: str) -> None:
        self._kind = kind
        self._column = column
        self._rule = rule

    @property
    def rule(self) -> str:
        """How the segments were chosen (recorded in every decision's provenance)."""
        return self._rule

    @classmethod
    def build(
        cls, frame: pd.DataFrame, datetime_column: str | None, max_categories: int
    ) -> SegmentPlan:
        """Weekday/weekend when a datetime column exists, else a low-cardinality column."""
        if (
            datetime_column is not None
            and datetime_column in frame.columns
            and pd.api.types.is_datetime64_any_dtype(frame[datetime_column])
        ):
            return cls(
                "datetime",
                datetime_column,
                f"weekday/weekend by {datetime_column} day of week",
            )
        for name in frame.columns:
            series = frame[name]
            if pd.api.types.is_float_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
                continue
            unique = int(series.nunique(dropna=True))
            if 2 <= unique <= max_categories:
                return cls("levels", str(name), f"levels of {name}")
        return cls(
            "none",
            None,
            "no semantic split available (row halves of a sample are not independent)",
        )

    def labels(self, frame: pd.DataFrame) -> pd.Series:
        """The segment label of every row of ``frame`` (a rule applied to that frame)."""
        if self._kind == "datetime" and self._column in frame.columns:
            weekend = frame[self._column].dt.dayofweek >= 5
            return pd.Series(np.where(weekend, "weekend", "weekday"), index=frame.index)
        if self._kind == "levels" and self._column in frame.columns:
            return frame[self._column].astype(str).reset_index(drop=True)
        return pd.Series([_ALL_ROWS] * len(frame))

    @staticmethod
    def groups(labels: pd.Series) -> list[tuple[str, np.ndarray]]:
        """The segment name and its positions inside the labelled subset (sorted)."""
        values = labels.to_numpy()
        groups = [
            (str(label), np.flatnonzero(values == label))
            for label in sorted(pd.unique(values))
        ]
        groups = [(name, index) for name, index in groups if index.size >= _MIN_SEGMENT_ROWS]
        if len(groups) < 2:
            return [(_ALL_ROWS, np.arange(values.size))]
        return groups
