"""Build one daily calendar series from a canonical table.

The grain is declared by the caller (the dataset contract's row grain); a
``count`` series counts rows per day, a ``sum`` series sums one column per day.
Every calendar day in the declared window is present: a day with no rows is a
zero, so a gap never reaches the model as an interpolated observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from broadway.config.schema import SeriesSpec, TimeseriesStep


@dataclass(frozen=True)
class DailySeries:
    """A daily value per calendar day, with the grain and source it was built from."""

    name: str
    label: str
    unit: str
    aggregation: str
    column: str | None
    grain_column: str
    dates: pd.DatetimeIndex
    values: np.ndarray

    @property
    def n(self) -> int:
        """Number of calendar days in the window."""
        return int(self.values.size)

    def frame(self) -> pd.DataFrame:
        """Aggregate evidence table (date + value), never raw rows."""
        return pd.DataFrame({"date": self.dates, self.name: self.values})


class DailySeriesBuilder:
    """Factory for daily series: one window, one timestamp column, one spec per series."""

    def __init__(self, datetime_column: str, grain_column: str, start: date, end: date) -> None:
        if end < start:
            raise ValueError(f"series window end {end} precedes start {start}")
        self._datetime_column = datetime_column
        self._grain_column = grain_column
        self._start = pd.Timestamp(start)
        self._end = pd.Timestamp(end)
        self._index = pd.date_range(self._start, self._end, freq="D")

    @classmethod
    def from_config(cls, step: TimeseriesStep) -> DailySeriesBuilder:
        """Build from the step's declared datetime, grain and window."""
        return cls(step.datetime_column, step.grain_column, step.start, step.end)

    def build(self, frame: pd.DataFrame, name: str, spec: SeriesSpec) -> DailySeries:
        """Aggregate ``frame`` to one value per day in the declared window."""
        timestamps = pd.to_datetime(frame[self._datetime_column])
        days = timestamps.dt.normalize()
        in_window = days.between(self._start, self._end)
        if not bool(in_window.any()):
            raise ValueError(
                f"series {name!r}: no rows between {self._start.date()} and "
                f"{self._end.date()} in column {self._datetime_column!r}"
            )
        selected = days[in_window]
        if spec.aggregation == "count":
            grouped = selected.groupby(selected).size().astype(float)
        else:
            if spec.column is None:
                raise ValueError(f"series {name!r}: sum aggregation needs a column")
            values = pd.to_numeric(frame.loc[in_window, spec.column], errors="raise")
            grouped = values.groupby(selected).sum(min_count=1).astype(float)
        ordered = grouped.reindex(self._index).fillna(0.0)
        return DailySeries(
            name=name,
            label=spec.label,
            unit=spec.unit,
            aggregation=spec.aggregation,
            column=spec.column,
            grain_column=self._grain_column,
            dates=self._index,
            values=ordered.to_numpy(dtype=float),
        )
