"""Missingness observation: per-column counts/rates and co-missing patterns.

This module NEVER imputes. Imputation is owned by a later batch; here the
missingness surface is measured and recorded verbatim.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from broadway.eda.models import MissingColumn, MissingPattern, MissingResult


class MissingnessAnalyzer:
    """Counts missing cells and groups rows by their missing-column signature."""

    def __init__(self, max_patterns: int) -> None:
        self._max_patterns = max_patterns

    def analyze(self, frame: pd.DataFrame) -> MissingResult:
        total_rows = len(frame)
        total_cells = int(frame.size)
        mask = frame.isna()
        missing_cells = int(mask.to_numpy().sum())
        missing_columns: list[MissingColumn] = []
        complete_columns: list[str] = []
        for name in frame.columns:
            count = int(mask[name].sum())
            if count:
                missing_columns.append(
                    MissingColumn(
                        column=str(name),
                        missing_count=count,
                        missing_rate=self._rate(count, total_rows),
                    )
                )
            else:
                complete_columns.append(str(name))
        missing_columns.sort(key=lambda item: item.missing_count, reverse=True)
        patterns, pattern_count = self._patterns(mask, total_rows)
        return MissingResult(
            total_rows=total_rows,
            total_cells=total_cells,
            missing_cells=missing_cells,
            missing_rate=self._rate(missing_cells, total_cells),
            columns=missing_columns,
            complete_columns=complete_columns,
            patterns=patterns,
            pattern_count=pattern_count,
        )

    def _patterns(
        self, mask: pd.DataFrame, total_rows: int
    ) -> tuple[list[MissingPattern], int]:
        present = [str(name) for name in mask.columns if bool(mask[name].any())]
        if not present or total_rows == 0:
            return [], 0
        signature = mask[present].to_numpy()
        counter: Counter[tuple[bool, ...]] = Counter(
            tuple(bool(value) for value in row) for row in signature
        )
        patterns = [
            MissingPattern(
                columns=[column for column, missing in zip(present, key) if missing],
                row_count=int(count),
                row_rate=self._rate(int(count), total_rows),
            )
            for key, count in counter.most_common(self._max_patterns)
        ]
        return patterns, len(counter)

    @staticmethod
    def _rate(part: int, whole: int) -> float:
        return round(part / whole, 6) if whole else 0.0
