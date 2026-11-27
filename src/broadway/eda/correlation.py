"""Correlation evidence: which numeric columns move with the target, and with each other?

Pearson measures a linear association, Spearman a monotone one — on a skewed
column (a fare, a duration) the two disagree, and that disagreement is itself
evidence. Every pair carries its pairwise n, both p-values and the
Benjamini-Hochberg q of the family, so a rank order is never read as truth.
The numeric set is bounded by the target ranking (``eda.max_features``) so a
wide frame pays for the columns that matter, not O(p^2) of them.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from broadway.config.schema import EdaStep
from broadway.eda.correction import MultipleTestingCorrection
from broadway.eda.kinds import ColumnKinds
from broadway.eda.models import CorrelationPair, CorrelationResult, TargetCorrelation

_MIN_PAIR_ROWS = 3


class CorrelationAnalyzer:
    """Pearson + Spearman over the numeric columns, BH-corrected, target-ranked."""

    def __init__(self, step: EdaStep) -> None:
        self._method = step.correlation_method
        self._max_features = step.max_features
        self._correction = MultipleTestingCorrection(step.correction)

    def analyze(
        self,
        frame: pd.DataFrame,
        target: str,
        roles: dict[str, str] | None = None,
    ) -> CorrelationResult:
        """Rank the numeric features against the target, then correlate the top set.

        The target ranks the features but never enters the correlation table:
        it is the outcome, and its distribution belongs to the class balance.
        """
        roles = roles or {}
        numeric = self._numeric_columns(frame)
        ranking, reason = self._ranking(frame, numeric, target)
        selected = self._selected(numeric, ranking, target)
        pairs = self._pairs(frame, selected, roles)
        return CorrelationResult(
            target=target,
            method=self._method,
            columns=selected,
            pearson=self._matrix(selected, pairs, "pearson"),
            spearman=self._matrix(selected, pairs, "spearman"),
            pairs=pairs,
            ranking=ranking,
            reason=reason,
        )

    def _numeric_columns(self, frame: pd.DataFrame) -> list[str]:
        """Measurements only: an integer column is a code and stays out of the matrix."""
        return [
            str(name)
            for name in frame.columns
            if ColumnKinds.feature(frame[name]) == "numeric"
        ]

    def _selected(
        self, numeric: list[str], ranking: list[TargetCorrelation], target: str
    ) -> list[str]:
        """The top-ranked features only; the target ranks them but is not a column."""
        ranked = [item.feature for item in ranking][: self._max_features]
        chosen = ranked if target in numeric else numeric[: self._max_features]
        return [name for name in numeric if name in chosen and name != target]

    def _ranking(
        self, frame: pd.DataFrame, numeric: list[str], target: str
    ) -> tuple[list[TargetCorrelation], str | None]:
        if target not in frame.columns:
            return [], f"target '{target}' is not in the profiled frame"
        if target not in numeric:
            return [], f"target '{target}' is not numeric; ranked association belongs to the tests"
        features = [name for name in numeric if name != target]
        if not features:
            return [], "no numeric feature to rank against the target"
        rows = [self._target_pair(frame, name, target) for name in features]
        q_values = self._correction.adjust([row.p_value for row in rows])
        ranked = [
            row.model_copy(update={"q_value": q}) for row, q in zip(rows, q_values)
        ]
        return sorted(ranked, key=lambda item: (-item.value, item.feature)), None

    def _target_pair(
        self, frame: pd.DataFrame, feature: str, target: str
    ) -> TargetCorrelation:
        values = self._complete(frame, feature, target)
        n = len(values)
        if n < _MIN_PAIR_ROWS:
            return TargetCorrelation(
                feature=feature, method=self._method, value=0.0, p_value=1.0, q_value=1.0, n=n
            )
        left = values[feature].to_numpy(dtype=float)
        right = values[target].to_numpy(dtype=float)
        if np.unique(left).size < 2 or np.unique(right).size < 2:
            return TargetCorrelation(
                feature=feature, method=self._method, value=0.0, p_value=1.0, q_value=1.0, n=n
            )
        if self._method == "pearson":
            coefficient, p_value = stats.pearsonr(left, right)
        else:
            coefficient, p_value = stats.spearmanr(left, right)
        return TargetCorrelation(
            feature=feature,
            method=self._method,
            value=abs(self._coefficient(float(coefficient))),
            p_value=self._p_value(float(p_value)),
            q_value=1.0,
            n=n,
        )

    def _pairs(
        self,
        frame: pd.DataFrame,
        columns: list[str],
        roles: dict[str, str] | None = None,
    ) -> list[CorrelationPair]:
        roles = roles or {}
        pairs = [
            self._pair(frame, left, right, roles)
            for left, right in itertools.combinations(columns, 2)
        ]
        pearson_q = self._correction.adjust(
            [item.pearson_p if item.pearson_p is not None else 1.0 for item in pairs]
        )
        spearman_q = self._correction.adjust(
            [item.spearman_p if item.spearman_p is not None else 1.0 for item in pairs]
        )
        return [
            item.model_copy(update={"pearson_q": pq, "spearman_q": sq})
            for item, pq, sq in zip(pairs, pearson_q, spearman_q)
        ]

    def _pair(
        self,
        frame: pd.DataFrame,
        left: str,
        right: str,
        roles: dict[str, str] | None = None,
    ) -> CorrelationPair:
        roles = roles or {}
        values = self._complete(frame, left, right)
        n = len(values)
        if n < _MIN_PAIR_ROWS:
            return CorrelationPair(
                left=left, right=right,
                left_role=roles.get(left), right_role=roles.get(right), n=n,
                pearson=None, pearson_p=None, pearson_q=None,
                spearman=None, spearman_p=None, spearman_q=None,
            )
        first = values[left].to_numpy(dtype=float)
        second = values[right].to_numpy(dtype=float)
        if np.unique(first).size < 2 or np.unique(second).size < 2:
            return CorrelationPair(
                left=left, right=right,
                left_role=roles.get(left), right_role=roles.get(right), n=n,
                pearson=None, pearson_p=None, pearson_q=None,
                spearman=None, spearman_p=None, spearman_q=None,
            )
        pearson, pearson_p = stats.pearsonr(first, second)
        spearman, spearman_p = stats.spearmanr(first, second)
        return CorrelationPair(
            left=left,
            right=right,
            left_role=roles.get(left),
            right_role=roles.get(right),
            n=n,
            pearson=self._coefficient(float(pearson)),
            pearson_p=self._p_value(float(pearson_p)),
            pearson_q=None,
            spearman=self._coefficient(float(spearman)),
            spearman_p=self._p_value(float(spearman_p)),
            spearman_q=None,
        )

    def _matrix(
        self,
        columns: list[str],
        pairs: list[CorrelationPair],
        attribute: str,
    ) -> list[list[float | None]]:
        """Symmetric matrix built from the pairwise table (same rows, same n)."""
        index = {name: position for position, name in enumerate(columns)}
        matrix: list[list[float | None]] = [
            [1.0 if row == column else None for column in range(len(columns))]
            for row in range(len(columns))
        ]
        for pair in pairs:
            value = getattr(pair, attribute)
            row, column = index[pair.left], index[pair.right]
            matrix[row][column] = value
            matrix[column][row] = value
        return matrix

    @staticmethod
    def _complete(frame: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
        """Pairwise-complete finite rows: a pair is measured on what both observe."""
        pair = frame[[left, right]].dropna()
        finite = np.isfinite(pair[left].to_numpy(dtype=float)) & np.isfinite(
            pair[right].to_numpy(dtype=float)
        )
        return pair[finite]

    @staticmethod
    def _coefficient(value: float) -> float:
        return value if np.isfinite(value) else 0.0

    @staticmethod
    def _p_value(value: float) -> float:
        return value if np.isfinite(value) else 1.0
