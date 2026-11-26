"""The dtype → kind rule every EDA association surface shares (one owner).

A floating-point column is a measurement and is ranked as continuous. An
integer column is a code, a count or a year: testing it as a nominal group keeps
a zone id from being read as an ordering ("LocationID 250 beats LocationID 60"),
which a rank correlation would happily report as an `act`. Datetime columns have
no association test. The declared task — not the dtype — decides whether the
target is a regression target or a class vector, because a 0/1 integer target
looks numeric right up to the moment it is scored as one.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from broadway.config.schema import TaskType

FeatureKind = Literal["numeric", "categorical"]
TargetKind = Literal["numeric", "binary", "multiclass"]


class ColumnKinds:
    """Classify one column for the correlation and association surfaces."""

    @staticmethod
    def feature(series: pd.Series) -> FeatureKind | None:
        """``numeric`` for measurements, ``categorical`` for codes/counts, None for time."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return None
        if pd.api.types.is_timedelta64_dtype(series):
            return None
        if pd.api.types.is_bool_dtype(series):
            return "categorical"
        if pd.api.types.is_float_dtype(series):
            return "numeric"
        return "categorical"

    @staticmethod
    def target(series: pd.Series, task: TaskType) -> TargetKind:
        """The declared task decides regression vs classification; cardinality splits classes."""
        if pd.api.types.is_datetime64_any_dtype(series):
            raise ValueError("eda: a datetime target has no association test")
        if task is TaskType.REGRESSION and pd.api.types.is_numeric_dtype(series):
            return "numeric"
        return "binary" if int(series.nunique()) <= 2 else "multiclass"
