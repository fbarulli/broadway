"""Per-feature association tests as decisions: does this feature move the target?

The test is chosen by dtype and shape, never by taste: a numeric feature against
a binary target is a Welch t (point-biserial r), a numeric feature against a
multi-level target a one-way ANOVA (η²) or Kruskal-Wallis (ε²) depending on the
per-group normality check, a categorical pair a χ² with Cramér's V. Float
columns are measurements; integer columns are codes/counts and are tested as
nominal groups (a zone id is never an ordering).

Every published number comes from one owner: ``reports.measurement`` draws the
frame once, bootstraps the effect for the interval and shuffles the target for
the null band at the same n and seed. The effect algebra lives once here
(``EffectStatistic``) and serves both the observed test and the resampling, so
the value, the interval and the null cannot disagree. Per-feature decisions use
semantic segments (``eda.segments``); the summary decision is the family's own
aggregate — act when any feature acts, else the strongest investigate.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from broadway.config.schema import EdaStep, TaskType
from broadway.eda.correction import MultipleTestingCorrection
from broadway.eda.kinds import ColumnKinds, FeatureKind, TargetKind
from broadway.eda.models import FeatureTestBundle, FeatureTestResult
from broadway.eda.segments import SegmentPlan
from broadway.reports.decision import Decision, DecisionRule, Evidence, SegmentEvidence
from broadway.reports.measurement import MeasurementEstimator, MeasurementSpec
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
_CONTEXT_ACTION = "context only — measured for interpretation; never a predictor"
_BIAS_NOTE = (
    "for high-cardinality nominal columns (e.g. zip/city) the row bootstrap "
    "changes the occupied level count; the interval is not centred on the "
    "corrected V, so those decisions are investigate, not act"
)
_EFFECT_NAMES: dict[TestName, EffectName] = {
    "spearman": "abs_spearman_r",
    "welch": "point_biserial_r",
    "mann_whitney": "rank_biserial_r",
    "anova": "eta_squared",
    "kruskal": "epsilon_squared",
    "chi2": "cramers_v",
}


class EffectStatistic:
    """The effect algebra for one test, callable on any feature/target pair frame.

    ``values`` is the measured (continuous) column and ``labels`` the grouping
    or target column; both are numeric codes on the pair frame the caller builds,
    so a resample (bootstrap or shuffle) is just another frame.
    """

    def __init__(
        self, test: TestName, values: str, labels: str, corrected: bool = False
    ) -> None:
        self.test = test
        self.effect_name = _EFFECT_NAMES[test]
        self._values = values
        self._labels = labels
        self._corrected = corrected

    def __call__(self, frame: pd.DataFrame) -> float:
        values = frame[self._values].to_numpy(dtype=float)
        labels = frame[self._labels].to_numpy(dtype=float)
        return self.measure(values, labels)

    def measure(self, values: np.ndarray, labels: np.ndarray) -> float:
        """The effect on two aligned arrays (one code path for observed and resampled)."""
        if values.size < _MIN_ROWS:
            return 0.0
        if self.test == "spearman":
            return abs(self.rank_correlation(values, labels))
        if self.test == "welch":
            return self.point_biserial(values, labels)
        if self.test == "mann_whitney":
            return self.rank_biserial(values, labels)
        if self.test == "anova":
            return self.eta_squared(values, labels)
        if self.test == "kruskal":
            return self.epsilon_squared(values, labels)
        if self._corrected:
            return self.cramers_v_corrected(values, labels)
        return self.cramers_v(values, labels)

    @staticmethod
    def rank_correlation(values: np.ndarray, labels: np.ndarray) -> float:
        """Spearman's rho on midranks; 0.0 when either side is constant."""
        return EffectStatistic.correlation(stats.rankdata(values), stats.rankdata(labels))

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

    @staticmethod
    def point_biserial(values: np.ndarray, labels: np.ndarray) -> float:
        """|point-biserial r| = |Pearson r| between the values and the 0/1 labels."""
        return abs(EffectStatistic.correlation(values, labels))

    @staticmethod
    def rank_biserial(values: np.ndarray, labels: np.ndarray) -> float:
        """|rank-biserial r| from the Mann-Whitney U of the two groups."""
        ranks = stats.rankdata(values)
        first = labels == 0
        count_first = int(first.sum())
        count_second = int(labels.size - count_first)
        if count_first == 0 or count_second == 0:
            return 0.0
        statistic = float(ranks[first].sum()) - count_first * (count_first + 1) / 2.0
        return abs(2.0 * statistic / (count_first * count_second) - 1.0)

    @staticmethod
    def eta_squared(values: np.ndarray, codes: np.ndarray) -> float:
        """Between-group share of total variance (η²) on the groups present."""
        sizes, means, centre, total = EffectStatistic._group_moments(values, codes)
        if sizes.size < 2 or total == 0.0:
            return 0.0
        between = float((sizes * (means - centre) ** 2).sum())
        return float(np.clip(between / total, 0.0, 1.0))

    @staticmethod
    def epsilon_squared(values: np.ndarray, codes: np.ndarray) -> float:
        """Kruskal's rank effect ε² = (H - k + 1) / (n - k), on midranks.

        Ties are handled by midranks in H and the published value does NOT apply
        the tie correction to the denominator; on a heavily tied 1-5 rating the
        corrected variant is larger (e.g. 0.2515 vs the published 0.2299 for the
        the retired survey's rating-accuracy family). The uncorrected definition is the one the
        registry's ``effect_name`` records — a consumer comparing across studies
        must use the same correction, not mix them.
        """
        ranks = stats.rankdata(values)
        sizes, means, _centre, _total = EffectStatistic._group_moments(ranks, codes)
        if sizes.size < 2 or ranks.size <= sizes.size:
            return 0.0
        h = float(
            12.0
            / (ranks.size * (ranks.size + 1))
            * (sizes * means**2).sum()
            - 3.0 * (ranks.size + 1)
        )
        return epsilon_squared(h, int(sizes.size), int(ranks.size))

    @staticmethod
    def cramers_v(feature: np.ndarray, target: np.ndarray) -> float:
        """Cramér's V of the contingency table, empty rows/columns dropped."""
        table = EffectStatistic.contingency(feature, target)
        if table.shape[0] < 2 or table.shape[1] < 2:
            return 0.0
        chi_square = EffectStatistic.chi_square(table)
        denominator = float(table.sum()) * min(table.shape[0] - 1, table.shape[1] - 1)
        return 0.0 if denominator <= 0.0 else float(np.sqrt(chi_square / denominator))

    @staticmethod
    def cramers_v_corrected(feature: np.ndarray, target: np.ndarray) -> float:
        """Bergsma's bias-corrected Cramér's V (the small-cell counterpart)."""
        table = EffectStatistic.contingency(feature, target)
        rows, columns = table.shape
        total = float(table.sum())
        if rows < 2 or columns < 2 or total <= 1.0:
            return 0.0
        degrees = float((rows - 1) * (columns - 1))
        denominator = float(min(rows, columns) - 1) - degrees / (total - 1.0)
        if denominator <= 0.0:
            return 0.0
        phi_squared = EffectStatistic.chi_square(table) / total - degrees / (total - 1.0)
        return float(np.sqrt(max(phi_squared, 0.0) / denominator))

    @staticmethod
    def contingency(feature: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Observed contingency table with empty rows/columns removed.

        ``bincount`` sizes itself to the largest *observed* combined code, which
        is narrower than the full table whenever the top feature code appears
        with only one target class (a high-cardinality city whose last
        alphabetical value is single-class). Reserving the full shape keeps the
        reshape valid; empty rows are then dropped below.
        """
        feature_codes = feature.astype(int)
        target_codes = target.astype(int)
        rows = int(feature_codes.max()) + 1
        columns = int(target_codes.max()) + 1
        combined = feature_codes * columns + target_codes
        counts = np.bincount(combined, minlength=rows * columns).astype(float)
        table = counts.reshape(rows, columns)
        row_index = np.flatnonzero(table.sum(axis=1) > 0)
        column_index = np.flatnonzero(table.sum(axis=0) > 0)
        return table[np.ix_(row_index, column_index)]

    @staticmethod
    def chi_square(table: np.ndarray) -> float:
        total = float(table.sum())
        expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True) / total
        mask = expected > 0.0
        return float(((table[mask] - expected[mask]) ** 2 / expected[mask]).sum())

    @staticmethod
    def _group_moments(
        values: np.ndarray, codes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Present-group sizes and means, plus the grand mean and total SS."""
        codes = codes.astype(int)
        sizes = np.bincount(codes).astype(float)
        level_sums = np.bincount(codes, weights=values).astype(float)
        present = sizes > 0
        sizes = sizes[present]
        means = level_sums[present] / sizes
        centre = float(values.mean())
        total = float(((values - centre) ** 2).sum())
        return sizes, means, centre, total


class FeatureTest:
    """One observed per-feature test: its statistic, analytic p, and counterpart."""

    def __init__(
        self,
        statistic: EffectStatistic,
        p_value: float,
        alternative_name: str,
        alternative_value: float,
    ) -> None:
        self.statistic = statistic
        self._p_value = p_value
        self._alternative = (alternative_name, alternative_value)

    @property
    def name(self) -> TestName:
        return self.statistic.test

    @property
    def effect_name(self) -> EffectName:
        return self.statistic.effect_name

    def p_value(self) -> float:
        return self._p_value

    def alternative(self) -> tuple[str, float]:
        return self._alternative


class FeatureTestFactory:
    """Chooses the measure by dtype and shape (Factory Method + Strategy)."""

    def __init__(
        self,
        alpha: float,
        shapiro_sample: int,
        random_state: int,
        max_categorical_levels: int,
    ) -> None:
        self._alpha = alpha
        self._shapiro_sample = shapiro_sample
        self._random_state = random_state
        self._max_categorical_levels = max_categorical_levels

    def build(
        self,
        pair: pd.DataFrame,
        feature: str,
        target: str,
        feature_kind: FeatureKind,
        target_kind: TargetKind,
    ) -> FeatureTest | None:
        """One prepared test for this feature, or ``None`` when it is untestable."""
        if len(pair) < _MIN_ROWS:
            return None
        if target_kind == "numeric" and feature_kind == "numeric":
            return self._spearman(pair, feature, target)
        if target_kind == "numeric":
            return self._groups(pair[target], pair[feature], feature, target)
        if feature_kind == "numeric":
            if target_kind == "binary":
                return self._binary(pair[feature], pair[target], feature, target)
            return self._groups(pair[feature], pair[target], feature, target)
        return self._categorical(pair[feature], pair[target], feature, target)

    def _spearman(self, pair: pd.DataFrame, feature: str, target: str) -> FeatureTest:
        values = pair[feature].to_numpy(dtype=float)
        labels = pair[target].to_numpy(dtype=float)
        statistic = EffectStatistic("spearman", feature, target)
        _coefficient, p_value = stats.spearmanr(values, labels)
        return FeatureTest(
            statistic,
            self._finite_p(p_value),
            "abs_pearson_r",
            abs(EffectStatistic.correlation(values, labels)),
        )

    def _binary(
        self, values: pd.Series, labels: pd.Series, feature: str, target: str
    ) -> FeatureTest | None:
        groups = {
            str(level): values.to_numpy(dtype=float)[labels.to_numpy() == level]
            for level in (0, 1)
        }
        if any(group.size < _MIN_GROUP_SIZE for group in groups.values()):
            return None
        parametric = self._normal(groups)
        statistic = EffectStatistic(
            "welch" if parametric else "mann_whitney", str(values.name), str(labels.name)
        )
        first, second = groups["0"], groups["1"]
        if parametric:
            _statistic, p_value = stats.ttest_ind(first, second, equal_var=False)
            alternative = ("rank_biserial_r", EffectStatistic.rank_biserial(
                values.to_numpy(dtype=float), labels.to_numpy(dtype=float)
            ))
        else:
            _statistic, p_value = stats.mannwhitneyu(first, second, alternative="two-sided")
            alternative = ("point_biserial_r", EffectStatistic.point_biserial(
                values.to_numpy(dtype=float), labels.to_numpy(dtype=float)
            ))
        return FeatureTest(statistic, self._finite_p(p_value), *alternative)

    def _groups(
        self, values: pd.Series, codes: pd.Series, feature: str, target: str
    ) -> FeatureTest | None:
        numeric = values.to_numpy(dtype=float)
        labels = codes.to_numpy(dtype=float)
        levels = np.unique(labels)
        groups = {str(int(level)): numeric[labels == level] for level in levels}
        if levels.size < 2 or any(g.size < _MIN_GROUP_SIZE for g in groups.values()):
            return None
        parametric = self._normal(groups)
        statistic = EffectStatistic(
            "anova" if parametric else "kruskal", str(values.name), str(codes.name)
        )
        if parametric:
            _statistic, p_value = stats.f_oneway(*groups.values())
            alternative = ("epsilon_squared", EffectStatistic.epsilon_squared(numeric, labels))
        else:
            _statistic, p_value = stats.kruskal(*groups.values())
            alternative = ("eta_squared", EffectStatistic.eta_squared(numeric, labels))
        return FeatureTest(statistic, self._finite_p(p_value), *alternative)

    def _categorical(
        self, feature_values: pd.Series, target_values: pd.Series, feature: str, target: str
    ) -> FeatureTest | None:
        feature_codes = feature_values.to_numpy(dtype=float)
        target_codes = target_values.to_numpy(dtype=float)
        levels = int(np.unique(feature_codes).size)
        if levels < 2 or np.unique(target_codes).size < 2:
            return None
        table = EffectStatistic.contingency(feature_codes, target_codes)
        _statistic, p_value, _dof, _expected = stats.chi2_contingency(table)
        # Above the declared cardinality cap the raw Cramér's V is mostly degrees
        # of freedom (a 14,994-level zip scores ~0.39 on shuffled labels). The
        # published effect is the bias-corrected V; the raw value stays in the
        # decision's sensitivity so the reader can see both.
        corrected = levels > self._max_categorical_levels
        statistic = EffectStatistic("chi2", feature, target, corrected=corrected)
        raw = EffectStatistic.cramers_v(feature_codes, target_codes)
        adjusted = EffectStatistic.cramers_v_corrected(feature_codes, target_codes)
        alternative = ("cramers_v", raw) if corrected else ("cramers_v_corrected", adjusted)
        return FeatureTest(statistic, self._finite_p(p_value), *alternative)

    def _normal(self, groups: dict[str, np.ndarray]) -> bool:
        """Parametric only when no group's normality is rejected at ``alpha``."""
        if len(groups) < 2:
            return False
        if any(np.unique(group).size < 2 for group in groups.values()):
            return False
        checked = check_normality(groups, self._random_state, self._shapiro_sample)
        return all(item["shapiro_p"] >= self._alpha for item in checked.values())

    @staticmethod
    def _finite_p(value: float) -> float:
        return float(value) if np.isfinite(value) else 1.0


class FeatureTestAnalyzer:
    """Measure → BH-correct → decide: one Decision per ranked feature."""

    def __init__(self, step: EdaStep, task: TaskType) -> None:
        self._step = step
        self._task = task
        self._factory = FeatureTestFactory(
            step.alpha,
            step.shapiro_sample,
            step.random_state,
            step.max_categorical_levels,
        )
        self._correction = MultipleTestingCorrection(step.correction)
        self._estimator = MeasurementEstimator()

    def analyze(
        self,
        frame: pd.DataFrame,
        target: str,
        provenance: dict[str, str],
        plan: SegmentPlan,
        roles: dict[str, str] | None = None,
    ) -> FeatureTestBundle:
        """Test every testable column; only role=feature decisions can summarize."""
        roles = roles or {}
        frame = frame.reset_index(drop=True)
        if target not in frame.columns:
            return self._empty(
                target,
                f"target '{target}' is not in the profiled frame; no association test ran",
                [str(name) for name in frame.columns],
            )
        target_kind = ColumnKinds.target(frame[target], self._task)
        tests, skipped = self._tests(frame, target, target_kind)
        if not tests:
            return self._empty(
                target,
                f"no testable feature column beside '{target}' "
                f"(skipped: {', '.join(skipped) or 'none'})",
                skipped,
            )
        q_values = self._correction.adjust([test.p_value() for _, _, test, _, _ in tests])
        rows = [
            self._row(feature, kind, test, pair, q_value, target_kind)
            for (feature, kind, test, pair, _positions), q_value in zip(tests, q_values)
        ]
        ranked = sorted(zip(tests, rows), key=lambda entry: (-entry[1].effect, entry[0][0]))
        results: list[FeatureTestResult] = []
        decisions: list[Decision] = []
        feature_decisions: list[Decision] = []
        for position, ((feature, _kind, test, pair, positions), row) in enumerate(ranked):
            if position < self._step.max_features:
                role = roles.get(feature, "feature")
                decision = self._decide(
                    feature, test, pair, positions, row, plan, provenance, target, frame, role
                )
                decisions.append(decision)
                if role == "feature":
                    feature_decisions.append(decision)
                row = row.model_copy(
                    update={
                        "verdict": decision.verdict,
                        "null_threshold": decision.null_threshold,
                        "bootstrap_biased": decision.bootstrap_biased,
                    }
                )
            results.append(row)
        return FeatureTestBundle(
            target=target,
            target_kind=target_kind,
            alpha=self._step.alpha,
            min_effect=self._step.min_effect,
            features=results,
            skipped=skipped,
            decisions=decisions,
            summary=self._summary(decisions, rows, feature_decisions),
        )

    def _tests(
        self, frame: pd.DataFrame, target: str, target_kind: TargetKind
    ) -> tuple[list[tuple[str, FeatureKind, FeatureTest, pd.DataFrame, np.ndarray]], list[str]]:
        tests: list[tuple[str, FeatureKind, FeatureTest, pd.DataFrame, np.ndarray]] = []
        skipped: list[str] = []
        for name in frame.columns:
            if str(name) == target:
                continue
            kind = ColumnKinds.feature(frame[name])
            if kind is None:
                skipped.append(str(name))
                continue
            pair, positions = self._pair(frame, str(name), target, kind, target_kind)
            test = self._factory.build(pair, str(name), target, kind, target_kind)
            if test is None:
                skipped.append(str(name))
                continue
            tests.append((str(name), kind, test, pair, positions))
        return tests, skipped

    def _row(
        self,
        feature: str,
        kind: FeatureKind,
        test: FeatureTest,
        pair: pd.DataFrame,
        q_value: float,
        target_kind: TargetKind,
    ) -> FeatureTestResult:
        alternative_name, alternative_value = test.alternative()
        return FeatureTestResult(
            feature=feature,
            feature_kind=kind,
            target_kind=target_kind,
            test=test.name,
            effect_name=test.effect_name,
            effect=test.statistic(pair),
            raw_effect_name=alternative_name,
            raw_effect=alternative_value,
            p_value=test.p_value(),
            q_value=q_value,
            n=len(pair),
            significant=q_value < self._step.alpha,
            verdict=None,
        )

    def _decide(
        self,
        feature: str,
        test: FeatureTest,
        pair: pd.DataFrame,
        positions: np.ndarray,
        row: FeatureTestResult,
        plan: SegmentPlan,
        provenance: dict[str, str],
        target: str,
        family: pd.DataFrame,
        role: str,
    ) -> Decision:
        measurement = self._estimator.estimate(
            pair,
            test.statistic,
            MeasurementSpec(
                bootstrap="iid",
                null="shuffle",
                null_column=target,
                iterations=self._step.permutations,
                seed=self._step.random_state,
                confidence=1.0 - self._step.alpha,
            ),
            name=f"association_{feature}",
        )
        segments = self._segments(test, pair, positions, plan, family)
        uncovered = max(measurement.sample_size - sum(s.sample_size for s in segments), 0)
        alternative_name, alternative_value = test.alternative()
        evidence = Evidence(
            baseline="no-effect shuffle of the target at this n",
            segments=segments,
            uncovered=uncovered,
            sensitivity={
                alternative_name: alternative_value,
                "iterations": float(self._step.permutations),
                "alpha": self._step.alpha,
            },
            falsifier=(
                "the association's interval falls at or below "
                "max(min_effect, the shuffle null band) — the feature would be "
                f"indistinguishable from a shuffled target at n={measurement.sample_size:,}"
            ),
            provenance={
                **provenance,
                "feature": feature,
                "role": role,
                "test": test.name,
                "rows": str(measurement.sample_size),
                "seed": str(measurement.seed),
                "resampling": measurement.resampling,
                "segments": plan.rule,
                **({"bootstrap_note": _BIAS_NOTE} if measurement.bootstrap_biased else {}),
            },
        )
        context = role != "feature"
        rule = DecisionRule(
            metric=f"association_{feature}",
            direction="max",
            # A context column is measured for interpretation, never acted on:
            # a 1.0 ceiling makes `act` unreachable (every effect is <= 1).
            min_effect=1.0 if context else self._step.min_effect,
            min_sample_size=self._step.min_sample_size,
            action=(
                _CONTEXT_ACTION if context else self._step.decisions.features.action
            ),
            owner=self._step.decisions.features.owner,
        )
        return measurement.to_decision(
            rule,
            question=(
                f"Does {feature} move the {row.target_kind} target '{target}'"
                " beyond the minimum effect and the no-effect null band?"
            ),
            evidence=evidence,
            reference=0.0,
        )

    def _segments(
        self,
        test: FeatureTest,
        pair: pd.DataFrame,
        positions: np.ndarray,
        plan: SegmentPlan,
        family: pd.DataFrame,
    ) -> list[SegmentEvidence]:
        """Re-measure the effect inside each semantic segment (not row halves)."""
        labels = plan.labels(family).iloc[positions].reset_index(drop=True)
        return [
            SegmentEvidence(
                segment=name,
                value=test.statistic(pair.iloc[index]),
                sample_size=int(index.size),
            )
            for name, index in SegmentPlan.groups(labels)
        ]

    def _summary(
        self,
        decisions: list[Decision],
        rows: list[FeatureTestResult],
        eligible: list[Decision],
    ) -> Decision | None:
        """The family aggregate over FEATURE decisions only.

        Context columns (delivery_days/late/estimated_days) are measured but can
        never be the selected feature; the family counts still report every
        decided column. ``None`` when no feature-role column was decided.
        """
        if not decisions or not eligible:
            return None
        acting = [item for item in eligible if item.verdict == "act"]
        pool = (
            acting
            or [item for item in eligible if item.verdict == "investigate"]
            or eligible
        )
        chosen = max(
            pool,
            key=lambda item: (
                item.interval_low - item.threshold if acting else item.interval_high - item.threshold,
                item.value,
            ),
        )
        effect_name = next(
            (row.effect_name for row in rows if f"association_{row.feature}" == chosen.metric),
            None,
        )
        evidence = chosen.evidence.model_copy(
            update={
                "provenance": {
                    **chosen.evidence.provenance,
                    # The correction family is every TESTED feature (21), not only
                    # the 12 that got a bootstrap/permutation decision.
                    "family_size": str(len(rows)),
                    "decided": str(len(decisions)),
                    "feature_decided": str(len(eligible)),
                    "acted": str(sum(1 for item in decisions if item.verdict == "act")),
                    "investigated": str(
                        sum(1 for item in decisions if item.verdict == "investigate")
                    ),
                    "selected_feature": chosen.metric,
                    "effect_name": effect_name or chosen.metric,
                    **(
                        {"bootstrap_note": _BIAS_NOTE}
                        if any(item.bootstrap_biased for item in decisions)
                        else {}
                    ),
                }
            }
        )
        return chosen.model_copy(
            update={
                "question": (
                    "Is there at least one feature whose association with the target "
                    "survives multiple-testing correction and the minimum effect?"
                ),
                "unit": effect_name or chosen.unit,
                "evidence": evidence,
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
    def _pair(
        frame: pd.DataFrame,
        feature: str,
        target: str,
        kind: FeatureKind,
        target_kind: TargetKind,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Encode one pair as numeric columns and keep the original row positions."""
        values = frame[[feature, target]].dropna()
        if kind == "numeric":
            values = values[np.isfinite(values[feature].to_numpy(dtype=float))]
        if target_kind == "numeric":
            values = values[np.isfinite(values[target].to_numpy(dtype=float))]
        pair = pd.DataFrame(
            {
                feature: FeatureTestAnalyzer._encode(values[feature], kind),
                target: FeatureTestAnalyzer._encode(values[target], target_kind),
            }
        )
        return pair, values.index.to_numpy()

    @staticmethod
    def _encode(series: pd.Series, kind: FeatureKind | TargetKind | None) -> np.ndarray:
        """A measurement stays float; a code/count becomes a nominal integer code."""
        if kind == "numeric":
            return series.to_numpy(dtype=float)
        return pd.factorize(series, sort=True)[0].astype(float)
