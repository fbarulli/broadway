"""Per-feature association tests as decisions: does this feature move the target?

The test is chosen by dtype and shape, never by taste: a numeric feature against
a binary target is a Welch t (point-biserial r), a numeric feature against a
multi-level target a one-way ANOVA (η²) or Kruskal-Wallis (ε²) depending on the
per-group normality check, a categorical pair a χ² with Cramér's V. Every
feature then gets an effect size, a Benjamini-Hochberg q over the family, a
permutation null at this n, a bootstrap interval and a computed verdict — a raw
p-value is not evidence, and a significant-but-trivial association is not an act.

Each measure is a Strategy (``EffectTest``) built by ``FeatureTestFactory``; the
analyzer only orchestrates (measure → correct → decide).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from broadway.config.schema import EdaStep, TaskType
from broadway.eda.correction import MultipleTestingCorrection
from broadway.eda.kinds import ColumnKinds, FeatureKind, TargetKind
from broadway.eda.models import FeatureTestBundle, FeatureTestResult
from broadway.reports.decision import Decision, DecisionRule, Evidence, SegmentEvidence
from broadway.stats.assumptions import check_normality
from broadway.stats.effect_size import epsilon_squared

TestName = Literal["spearman", "welch", "mann_whitney", "anova", "kruskal", "chi2"]
EffectName = Literal[
    "abs_spearman_r",
    "point_biserial_r",
    "rank_biserial_r",
    "eta_squared",
    "epsilon_squared",
    "cramers_v",
]

_MIN_ROWS = 3
_MIN_GROUP_SIZE = 2
_CI_LOW = 2.5
_CI_HIGH = 97.5


class EffectTest(ABC):
    """One prepared measure: effect on a row subset, a null draw, its p-value."""

    name: TestName
    effect_name: EffectName
    n: int

    @abstractmethod
    def effect(self, index: np.ndarray | None = None) -> float:
        """The effect size on ``index`` rows (``None`` = every observed row)."""

    @abstractmethod
    def shuffled_effect(self, rng: np.random.Generator) -> float:
        """The effect under the no-effect null: the target's labels permuted."""

    @abstractmethod
    def p_value(self) -> float:
        """The analytic p-value of the selected test at the observed n."""

    @abstractmethod
    def alternative(self) -> tuple[str, float]:
        """The counterpart measure's name and value (the sensitivity check)."""

    @staticmethod
    def correlation(first: np.ndarray, second: np.ndarray) -> float:
        """Pearson correlation of two vectors; 0.0 when either is constant."""
        if first.size < 2:
            return 0.0
        left = first - first.mean()
        right = second - second.mean()
        denominator = float(np.sqrt((left * left).sum() * (right * right).sum()))
        if denominator == 0.0:
            return 0.0
        return float((left * right).sum() / denominator)


class RankCorrelationTest(EffectTest):
    """Numeric vs numeric: |Spearman r| on midranks, |Pearson r| as counterpart."""

    name: TestName = "spearman"
    effect_name: EffectName = "abs_spearman_r"

    def __init__(self, values: np.ndarray, target: np.ndarray) -> None:
        self.n = int(values.size)
        self._values = values
        self._target = target
        self._value_ranks = stats.rankdata(values)
        self._target_ranks = stats.rankdata(target)
        coefficient, p_value = stats.spearmanr(values, target)
        self._observed = abs(float(coefficient)) if np.isfinite(coefficient) else 0.0
        self._p_value = float(p_value) if np.isfinite(p_value) else 1.0

    def effect(self, index: np.ndarray | None = None) -> float:
        if index is None:
            return self._observed
        return abs(self.correlation(self._value_ranks[index], self._target_ranks[index]))

    def shuffled_effect(self, rng: np.random.Generator) -> float:
        return abs(self.correlation(self._value_ranks, rng.permutation(self._target_ranks)))

    def p_value(self) -> float:
        return self._p_value

    def alternative(self) -> tuple[str, float]:
        return "abs_pearson_r", abs(self.correlation(self._values, self._target))


class BinaryGroupTest(EffectTest):
    """Numeric feature vs binary target: two labelled groups."""

    def __init__(self, values: np.ndarray, labels: np.ndarray) -> None:
        self.n = int(values.size)
        self._values = values
        self._labels = labels
        self._floats = labels.astype(float)
        self._ranks = stats.rankdata(values)
        self._observed = self.point_biserial()

    def point_biserial(self, index: np.ndarray | None = None) -> float:
        """|point-biserial r| = |Pearson r| between the values and the 0/1 labels."""
        values = self._values if index is None else self._values[index]
        labels = self._floats if index is None else self._floats[index]
        return abs(self.correlation(values, labels))

    def rank_biserial(self, index: np.ndarray | None = None) -> float:
        """|rank-biserial r| from the Mann-Whitney U of the two groups."""
        labels = self._labels if index is None else self._labels[index]
        ranks = self._ranks if index is None else self._ranks[index]
        return self._rank_biserial(labels, ranks)

    def shuffled_rank_biserial(self, rng: np.random.Generator) -> float:
        """The rank-biserial effect with the group labels permuted (not the rows)."""
        return self._rank_biserial(rng.permutation(self._labels), self._ranks)

    @staticmethod
    def _rank_biserial(labels: np.ndarray, ranks: np.ndarray) -> float:
        first = labels == 0
        count_first = int(first.sum())
        count_second = int(labels.size - count_first)
        if count_first == 0 or count_second == 0:
            return 0.0
        statistic = float(ranks[first].sum()) - count_first * (count_first + 1) / 2.0
        return abs(2.0 * statistic / (count_first * count_second) - 1.0)

    def effect(self, index: np.ndarray | None = None) -> float:
        return self._observed

    def alternative(self) -> tuple[str, float]:
        return "rank_biserial_r", self.rank_biserial()


class WelchTest(BinaryGroupTest):
    """Parametric binary-group test; the effect is the point-biserial |r|."""

    name: TestName = "welch"
    effect_name: EffectName = "point_biserial_r"

    def __init__(self, values: np.ndarray, labels: np.ndarray) -> None:
        super().__init__(values, labels)
        first, second = values[labels == 0], values[labels == 1]
        _statistic, p_value = stats.ttest_ind(first, second, equal_var=False)
        self._p_value = float(p_value) if np.isfinite(p_value) else 1.0

    def effect(self, index: np.ndarray | None = None) -> float:
        return self.point_biserial(index)

    def shuffled_effect(self, rng: np.random.Generator) -> float:
        return abs(self.correlation(self._values, rng.permutation(self._floats)))

    def p_value(self) -> float:
        return self._p_value


class MannWhitneyTest(BinaryGroupTest):
    """Non-parametric binary-group test; the effect is the rank-biserial |r|."""

    name: TestName = "mann_whitney"
    effect_name: EffectName = "rank_biserial_r"

    def __init__(self, values: np.ndarray, labels: np.ndarray) -> None:
        super().__init__(values, labels)
        first, second = values[labels == 0], values[labels == 1]
        _statistic, p_value = stats.mannwhitneyu(first, second, alternative="two-sided")
        self._p_value = float(p_value) if np.isfinite(p_value) else 1.0

    def effect(self, index: np.ndarray | None = None) -> float:
        return self.rank_biserial(index)

    def shuffled_effect(self, rng: np.random.Generator) -> float:
        return self.shuffled_rank_biserial(rng)

    def p_value(self) -> float:
        return self._p_value

    def alternative(self) -> tuple[str, float]:
        return "point_biserial_r", self.point_biserial()


class GroupDifferenceTest(EffectTest):
    """A numeric value across ≥2 groups — the shared bincount algebra."""

    def __init__(self, values: np.ndarray, codes: np.ndarray) -> None:
        self.n = int(values.size)
        self._values = values
        self._codes = codes
        self._levels = int(codes.max()) + 1
        self._ranks = stats.rankdata(values)
        self._groups = [values[codes == level] for level in range(self._levels)]
        self._observed = self.effect()

    def _present(self, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sizes = np.bincount(codes, minlength=self._levels)[: self._levels].astype(float)
        return np.flatnonzero(sizes > 0), sizes

    def eta_squared(self, index: np.ndarray | None = None) -> float:
        """Between-group share of total variance (η²) on the selected rows."""
        values = self._values if index is None else self._values[index]
        codes = self._codes if index is None else self._codes[index]
        return self._eta_squared(values, codes)

    def epsilon_squared(self, index: np.ndarray | None = None) -> float:
        """Kruskal's rank effect ε² = (H - k + 1) / (n - k), on midranks."""
        ranks = self._ranks if index is None else self._ranks[index]
        codes = self._codes if index is None else self._codes[index]
        return self._epsilon_squared(ranks, codes)

    def shuffled_eta_squared(self, rng: np.random.Generator) -> float:
        """η² with the group labels permuted against the values (the null)."""
        return self._eta_squared(self._values, rng.permutation(self._codes))

    def shuffled_epsilon_squared(self, rng: np.random.Generator) -> float:
        """ε² with the group labels permuted against the ranks (the null)."""
        return self._epsilon_squared(self._ranks, rng.permutation(self._codes))

    def _eta_squared(self, values: np.ndarray, codes: np.ndarray) -> float:
        if values.size < _MIN_ROWS:
            return 0.0
        levels, sizes = self._present(codes)
        if levels.size < 2:
            return 0.0
        means = np.bincount(codes, weights=values, minlength=self._levels)[levels] / sizes[levels]
        centre = values.mean()
        total = float(((values - centre) ** 2).sum())
        between = float((sizes[levels] * (means - centre) ** 2).sum())
        return 0.0 if total == 0.0 else float(np.clip(between / total, 0.0, 1.0))

    def _epsilon_squared(self, ranks: np.ndarray, codes: np.ndarray) -> float:
        levels, sizes = self._present(codes)
        if levels.size < 2 or ranks.size <= levels.size:
            return 0.0
        mean_ranks = np.bincount(codes, weights=ranks, minlength=self._levels)[levels] / sizes[
            levels
        ]
        h = float(
            12.0
            / (ranks.size * (ranks.size + 1))
            * (sizes[levels] * mean_ranks**2).sum()
            - 3.0 * (ranks.size + 1)
        )
        return epsilon_squared(h, int(levels.size), int(ranks.size))


class AnovaTest(GroupDifferenceTest):
    """Normality is not rejected in every group: one-way ANOVA, effect = η²."""

    name: TestName = "anova"
    effect_name: EffectName = "eta_squared"

    def __init__(self, values: np.ndarray, codes: np.ndarray) -> None:
        super().__init__(values, codes)
        _statistic, p_value = stats.f_oneway(*self._groups)
        self._p_value = float(p_value) if np.isfinite(p_value) else 1.0

    def effect(self, index: np.ndarray | None = None) -> float:
        return self.eta_squared(index)

    def shuffled_effect(self, rng: np.random.Generator) -> float:
        return self.shuffled_eta_squared(rng)

    def p_value(self) -> float:
        return self._p_value

    def alternative(self) -> tuple[str, float]:
        return "epsilon_squared", self.epsilon_squared()


class KruskalTest(GroupDifferenceTest):
    """Normality is rejected (or untestable): Kruskal-Wallis, effect = ε²."""

    name: TestName = "kruskal"
    effect_name: EffectName = "epsilon_squared"

    def __init__(self, values: np.ndarray, codes: np.ndarray) -> None:
        super().__init__(values, codes)
        _statistic, p_value = stats.kruskal(*self._groups)
        self._p_value = float(p_value) if np.isfinite(p_value) else 1.0

    def effect(self, index: np.ndarray | None = None) -> float:
        return self.epsilon_squared(index)

    def shuffled_effect(self, rng: np.random.Generator) -> float:
        return self.shuffled_epsilon_squared(rng)

    def p_value(self) -> float:
        return self._p_value

    def alternative(self) -> tuple[str, float]:
        return "eta_squared", self.eta_squared()


class CategoricalAssociationTest(EffectTest):
    """Two categorical columns: χ² association, effect = Cramér's V."""

    name: TestName = "chi2"
    effect_name: EffectName = "cramers_v"

    def __init__(self, feature_codes: np.ndarray, target_codes: np.ndarray) -> None:
        self.n = int(feature_codes.size)
        self._feature = feature_codes
        self._target = target_codes
        self._feature_levels = int(feature_codes.max()) + 1
        self._target_levels = int(target_codes.max()) + 1
        self._observed = self._cramers_v(feature_codes, target_codes)
        table = self._table(feature_codes, target_codes)
        _statistic, p_value, _dof, _expected = stats.chi2_contingency(table)
        self._p_value = float(p_value) if np.isfinite(p_value) else 1.0

    def _table(self, feature: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Observed contingency table, empty rows/columns removed."""
        combined = feature * self._target_levels + target
        counts = np.bincount(combined, minlength=self._feature_levels * self._target_levels)
        table = counts.reshape(self._feature_levels, self._target_levels).astype(float)
        rows = np.flatnonzero(table.sum(axis=1) > 0)
        columns = np.flatnonzero(table.sum(axis=0) > 0)
        return table[np.ix_(rows, columns)]

    def _cramers_v(self, feature: np.ndarray, target: np.ndarray) -> float:
        table = self._table(feature, target)
        if table.shape[0] < 2 or table.shape[1] < 2:
            return 0.0
        total = float(table.sum())
        expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True) / total
        mask = expected > 0.0
        chi_square = float(((table[mask] - expected[mask]) ** 2 / expected[mask]).sum())
        denominator = total * min(table.shape[0] - 1, table.shape[1] - 1)
        return 0.0 if denominator <= 0.0 else float(np.sqrt(chi_square / denominator))

    def effect(self, index: np.ndarray | None = None) -> float:
        if index is None:
            return self._observed
        return self._cramers_v(self._feature[index], self._target[index])

    def shuffled_effect(self, rng: np.random.Generator) -> float:
        return self._cramers_v(self._feature, rng.permutation(self._target))

    def p_value(self) -> float:
        return self._p_value

    def alternative(self) -> tuple[str, float]:
        """Bergsma's bias-corrected Cramér's V — the small-cell robustness check."""
        table = self._table(self._feature, self._target)
        rows, columns = table.shape
        total = float(table.sum())
        if rows < 2 or columns < 2 or total <= 1.0:
            return "cramers_v_corrected", 0.0
        expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True) / total
        chi_square = float(((table - expected) ** 2 / expected).sum())
        degrees = float((rows - 1) * (columns - 1))
        denominator = float(min(rows, columns) - 1) - degrees / (total - 1.0)
        if denominator <= 0.0:
            return "cramers_v_corrected", 0.0
        phi_squared = chi_square / total - degrees / (total - 1.0)
        return "cramers_v_corrected", float(np.sqrt(max(phi_squared, 0.0) / denominator))


class FeatureTestFactory:
    """Chooses the measure by dtype and shape (Factory Method + Strategy)."""

    def __init__(self, alpha: float, shapiro_sample: int, random_state: int) -> None:
        self._alpha = alpha
        self._shapiro_sample = shapiro_sample
        self._random_state = random_state

    def build(
        self,
        frame: pd.DataFrame,
        feature: str,
        target: str,
        feature_kind: FeatureKind,
        target_kind: TargetKind,
    ) -> EffectTest | None:
        """One prepared test for this feature, or ``None`` when it is untestable."""
        if len(frame) < _MIN_ROWS:
            return None
        if target_kind == "numeric":
            if feature_kind == "numeric":
                return RankCorrelationTest(
                    frame[feature].to_numpy(dtype=float), frame[target].to_numpy(dtype=float)
                )
            return self._groups(frame[target].to_numpy(dtype=float), self._codes(frame, feature))
        target_codes = self._codes(frame, target)
        if feature_kind == "numeric":
            values = frame[feature].to_numpy(dtype=float)
            if target_kind == "binary":
                return self._binary(values, target_codes)
            return self._groups(values, target_codes)
        feature_codes = self._codes(frame, feature)
        if len(np.unique(feature_codes)) < 2 or len(np.unique(target_codes)) < 2:
            return None
        return CategoricalAssociationTest(feature_codes, target_codes)

    def _groups(self, values: np.ndarray, codes: np.ndarray) -> EffectTest | None:
        sizes = np.bincount(codes)
        if sizes.size < 2:
            return None
        if int(sizes[sizes > 0].min()) < _MIN_GROUP_SIZE:
            return None
        groups = {
            str(level): values[codes == level]
            for level in range(int(codes.max()) + 1)
            if (codes == level).any()
        }
        return AnovaTest(values, codes) if self._normal(groups) else KruskalTest(values, codes)

    def _binary(self, values: np.ndarray, codes: np.ndarray) -> EffectTest | None:
        groups = {str(level): values[codes == level] for level in (0, 1)}
        if any(group.size < _MIN_GROUP_SIZE for group in groups.values()):
            return None
        return WelchTest(values, codes) if self._normal(groups) else MannWhitneyTest(values, codes)

    def _normal(self, groups: dict[str, np.ndarray]) -> bool:
        """Parametric only when no group's normality is rejected at ``alpha``."""
        if len(groups) < 2:
            return False
        if any(np.unique(group).size < 2 for group in groups.values()):
            return False
        checked = check_normality(groups, self._random_state, self._shapiro_sample)
        return all(item["shapiro_p"] >= self._alpha for item in checked.values())

    @staticmethod
    def _codes(frame: pd.DataFrame, column: str) -> np.ndarray:
        return pd.factorize(frame[column], sort=True)[0]


class FeatureTestAnalyzer:
    """Measure → BH-correct → decide: one Decision per ranked feature."""

    def __init__(self, step: EdaStep, task: TaskType) -> None:
        self._step = step
        self._task = task
        self._factory = FeatureTestFactory(
            step.alpha, step.shapiro_sample, step.random_state
        )
        self._correction = MultipleTestingCorrection(step.correction)

    def analyze(
        self, frame: pd.DataFrame, target: str, provenance: dict[str, str]
    ) -> FeatureTestBundle:
        """Test every testable feature against the target and publish the decisions."""
        if target not in frame.columns:
            return self._empty(
                target,
                f"target '{target}' is not in the profiled frame; no association test ran",
                [str(name) for name in frame.columns],
            )
        target_kind = self._target_kind(frame[target])
        tests, skipped = self._tests(frame, target, target_kind)
        if not tests:
            return self._empty(
                target,
                f"no testable feature column beside '{target}' "
                f"(skipped: {', '.join(skipped) or 'none'})",
                skipped,
            )
        q_values = self._correction.adjust([test.p_value() for _, _, test in tests])
        ranked = sorted(
            (
                (str(feature), test, self._row(str(feature), kind, test, q, target_kind))
                for (feature, kind, test), q in zip(tests, q_values)
            ),
            key=lambda entry: (-entry[2].effect, entry[0]),
        )
        rng = np.random.default_rng(self._step.random_state)
        rows: list[FeatureTestResult] = []
        decisions: list[Decision] = []
        for position, (feature, test, row) in enumerate(ranked):
            if position < self._step.max_features:
                decision = self._decide(feature, test, row, rng, provenance, target)
                decisions.append(decision)
                row = row.model_copy(update={"verdict": decision.verdict})
            rows.append(row)
        return FeatureTestBundle(
            target=target,
            target_kind=target_kind,
            alpha=self._step.alpha,
            min_effect=self._step.min_effect,
            features=rows,
            skipped=skipped,
            decisions=decisions,
            summary=self._summary(decisions),
        )

    def _tests(
        self, frame: pd.DataFrame, target: str, target_kind: TargetKind
    ) -> tuple[list[tuple[str, FeatureKind, EffectTest]], list[str]]:
        tests: list[tuple[str, FeatureKind, EffectTest]] = []
        skipped: list[str] = []
        for name in frame.columns:
            if str(name) == target:
                continue
            kind = self._feature_kind(frame[name])
            if kind is None:
                skipped.append(str(name))
                continue
            pair = self._complete(frame, str(name), target, kind)
            test = self._factory.build(pair, str(name), target, kind, target_kind)
            if test is None:
                skipped.append(str(name))
                continue
            tests.append((str(name), kind, test))
        return tests, skipped

    def _row(
        self,
        feature: str,
        kind: FeatureKind,
        test: EffectTest,
        q_value: float,
        target_kind: TargetKind,
    ) -> FeatureTestResult:
        return FeatureTestResult(
            feature=feature,
            feature_kind=kind,
            target_kind=target_kind,
            test=test.name,
            effect_name=test.effect_name,
            effect=test.effect(),
            p_value=test.p_value(),
            q_value=q_value,
            n=test.n,
            significant=q_value < self._step.alpha,
            verdict=None,
        )

    def _decide(
        self,
        feature: str,
        test: EffectTest,
        row: FeatureTestResult,
        rng: np.random.Generator,
        provenance: dict[str, str],
        target: str,
    ) -> Decision:
        null = [test.shuffled_effect(rng) for _ in range(self._step.permutations)]
        draws = [
            test.effect(rng.integers(0, row.n, row.n))
            for _ in range(self._step.permutations)
        ]
        low, high = np.percentile(draws, [_CI_LOW, _CI_HIGH])
        alternative_name, alternative_value = test.alternative()
        evidence = Evidence(
            baseline="no-effect permutation of the target at this n",
            segments=self._halves(test, row.n),
            sensitivity={
                alternative_name: alternative_value,
                "permutations": float(self._step.permutations),
                "alpha": self._step.alpha,
            },
            falsifier=(
                "the association's 95% interval falls at or below "
                "max(min_effect, the permutation null band) — the feature would be "
                f"indistinguishable from a shuffled target at n={row.n:,}"
            ),
            provenance={
                **provenance,
                "feature": feature,
                "test": test.name,
                "rows": str(row.n),
                "seed": str(self._step.random_state),
            },
        )
        rule = DecisionRule(
            metric=f"association_{feature}",
            direction="max",
            min_effect=self._step.min_effect,
            null_quantile=self._step.null_quantile,
            min_sample_size=self._step.min_sample_size,
            action=self._step.decisions.features.action,
            owner=self._step.decisions.features.owner,
        )
        return Decision.evaluate(
            rule=rule,
            value=row.effect,
            interval=(min(float(low), row.effect), max(float(high), row.effect)),
            sample_size=row.n,
            question=(
                f"Does {feature} move the {row.target_kind} target '{target}'"
                " beyond the minimum effect and the no-effect null band?"
            ),
            evidence=evidence,
            null_values=null,
            reference=0.0,
        )

    def _halves(self, test: EffectTest, n: int) -> list[SegmentEvidence]:
        half = n // 2
        segments = [
            SegmentEvidence(segment=label, value=test.effect(index), sample_size=int(index.size))
            for label, index in (
                ("first half", np.arange(0, half)),
                ("second half", np.arange(half, n)),
            )
            if index.size > 0
        ]
        if not segments:
            segments.append(
                SegmentEvidence(segment="whole sample", value=test.effect(), sample_size=n)
            )
        return segments

    def _summary(self, decisions: list[Decision]) -> Decision | None:
        if not decisions:
            return None
        return decisions[0].model_copy(
            update={
                "question": (
                    "Is there at least one feature whose association with the target "
                    "survives multiple-testing correction and the minimum effect?"
                )
            }
        )

    def _empty(self, target: str, reason: str, skipped: list[str]) -> FeatureTestBundle:
        """A recorded no-op: the report states why no association test ran."""
        return FeatureTestBundle(
            target=target,
            target_kind=None,
            alpha=self._step.alpha,
            min_effect=self._step.min_effect,
            features=[],
            skipped=skipped,
            decisions=[],
            summary=None,
            reason=reason,
        )

    @staticmethod
    def _feature_kind(series: pd.Series) -> FeatureKind | None:
        return ColumnKinds.feature(series)

    def _target_kind(self, series: pd.Series) -> TargetKind:
        return ColumnKinds.target(series, self._task)

    @staticmethod
    def _complete(
        frame: pd.DataFrame, feature: str, target: str, kind: FeatureKind
    ) -> pd.DataFrame:
        """Pairwise-complete rows for one feature/target pair (finite numerics)."""
        pair = frame[[feature, target]].dropna()
        if kind == "numeric":
            pair = pair[np.isfinite(pair[feature].to_numpy(dtype=float))]
        if pd.api.types.is_numeric_dtype(pair[target]):
            pair = pair[np.isfinite(pair[target].to_numpy(dtype=float))]
        return pair.reset_index(drop=True)
