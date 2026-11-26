"""Before/after ETL comparison when both frames exist; otherwise a loud skip."""

from __future__ import annotations

import pandas as pd

from broadway.eda.models import ColumnShift, CompareResult


class EtlComparator:
    """Compares the pre-split and post-split frames on counts and numeric drift."""

    def __init__(self, before_label: str, after_label: str) -> None:
        self._before_label = before_label
        self._after_label = after_label

    def compare(
        self,
        before: pd.DataFrame | None,
        after: pd.DataFrame | None,
        before_rows: int,
        after_rows: int,
        skip_reason: str | None,
        notes: list[str] | None = None,
    ) -> CompareResult:
        notes = list(notes or [])
        if before is None or after is None:
            return CompareResult(
                available=False,
                skip_reason=skip_reason or "required before/after frame is missing",
                before_label=self._before_label,
                after_label=self._after_label,
                before_rows=before_rows if before is not None else None,
                after_rows=after_rows if after is not None else None,
                row_delta=None,
                added_columns=[],
                removed_columns=[],
                dtype_changes={},
                numeric_shifts=[],
                notes=notes,
            )
        before_columns = [str(name) for name in before.columns]
        after_columns = [str(name) for name in after.columns]
        common = [name for name in before_columns if name in after_columns]
        dtype_changes = {
            name: f"{before[name].dtype} -> {after[name].dtype}"
            for name in common
            if str(before[name].dtype) != str(after[name].dtype)
        }
        numeric = [
            name
            for name in common
            if not pd.api.types.is_bool_dtype(before[name])
            and pd.api.types.is_numeric_dtype(before[name])
            and not pd.api.types.is_bool_dtype(after[name])
            and pd.api.types.is_numeric_dtype(after[name])
        ]
        return CompareResult(
            available=True,
            skip_reason=None,
            before_label=self._before_label,
            after_label=self._after_label,
            before_rows=before_rows,
            after_rows=after_rows,
            row_delta=after_rows - before_rows,
            added_columns=[name for name in after_columns if name not in before_columns],
            removed_columns=[name for name in before_columns if name not in after_columns],
            dtype_changes=dtype_changes,
            numeric_shifts=[self._shift(before[name], after[name], name) for name in numeric],
            notes=notes,
        )

    @staticmethod
    def _shift(before: pd.Series, after: pd.Series, name: str) -> ColumnShift:
        before_mean = EtlComparator._mean(before)
        after_mean = EtlComparator._mean(after)
        before_median = EtlComparator._median(before)
        after_median = EtlComparator._median(after)
        return ColumnShift(
            column=name,
            before_mean=before_mean,
            after_mean=after_mean,
            before_median=before_median,
            after_median=after_median,
            mean_delta=(
                round(after_mean - before_mean, 6)
                if before_mean is not None and after_mean is not None
                else None
            ),
            median_delta=(
                round(after_median - before_median, 6)
                if before_median is not None and after_median is not None
                else None
            ),
        )

    @staticmethod
    def _mean(column: pd.Series) -> float | None:
        values = column.dropna()
        return float(values.mean()) if not values.empty else None

    @staticmethod
    def _median(column: pd.Series) -> float | None:
        values = column.dropna()
        return float(values.median()) if not values.empty else None
