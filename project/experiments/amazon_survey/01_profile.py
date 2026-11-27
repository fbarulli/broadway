"""Which reported behaviour tracks shopping satisfaction most strongly at n=602?

The survey asks respondents to rate how important accurate ratings are and how
satisfied they are with shopping; the EDA step ranked ``rating_accuracy_score``
first. This step measures that association as |Spearman rho| on the canonical
ETL frame, with its bootstrap interval and a selection-adjusted permutation
null (the maximum |rho| over the whole EDA-ranked family at the same n, so a
feature picked from a ranking is not judged against a per-feature null), and
re-measures it inside the semantic slices the shared segment plan names.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

from broadway.config.loader import load_config
from broadway.data.loader import canonical_path
from broadway.eda.kinds import ColumnKinds
from broadway.eda.segments import SegmentPlan
from broadway.reports.decision import (
    Decision,
    DecisionRule,
    DecisionRuleSet,
    Evidence,
    SegmentEvidence,
)
from broadway.reports.measurement import MeasurementEstimator, MeasurementSpec
from broadway.reports.step_output import StepOutput
from broadway.utils import require_keys
from project.paths import load_project_paths

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem
PATHS = load_project_paths()
RESULTS = PATHS.results / HERE.name
CONFIG = PATHS.experiment_configs / "amazon_survey.yaml"


class RankAssociation:
    """|Spearman rho| of one feature against the target, callable on any frame.

    The encoding is the EDA kind rule (a measurement stays float, a code becomes a
    nominal code) so the observed value, the bootstrap draws and the permutation
    null all see the same pair algebra.
    """

    def __init__(self, feature: str, target: str) -> None:
        self.feature = feature
        self.target = target

    def pair(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """The aligned encoded pair (pairwise-complete rows)."""
        pair = frame[[self.feature, self.target]].dropna()
        return self.encode(pair[self.feature]), pair[self.target].to_numpy(dtype=float)

    def __call__(self, frame: pd.DataFrame) -> float:
        return self.rho(*self.pair(frame))

    @staticmethod
    def encode(series: pd.Series) -> np.ndarray:
        if ColumnKinds.feature(series) == "numeric":
            return series.to_numpy(dtype=float)
        return pd.factorize(series, sort=True)[0].astype(float)

    @staticmethod
    def rho(values: np.ndarray, target: np.ndarray) -> float:
        """|rho| of two aligned arrays (0.0 when either side is constant)."""
        if values.size < 3 or np.unique(values).size < 2 or np.unique(target).size < 2:
            return 0.0
        coefficient = stats.spearmanr(values, target).statistic
        return 0.0 if not np.isfinite(coefficient) else abs(float(coefficient))


@dataclass(frozen=True)
class SurveyFrame:
    """The measured frame plus the bindings the record's provenance needs."""

    frame: pd.DataFrame
    artifact: Path
    source: str
    dataset: str
    target: str
    segment_plan: SegmentPlan


def _experiment_config() -> dict:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    require_keys(config, ["dataset", "seed", "iterations", "feature", "family"], CONFIG.name)
    return config


def _rule(step: str) -> DecisionRule:
    return DecisionRuleSet.from_yaml(CONFIG).rule(step)


def _load(dataset: str) -> SurveyFrame:
    """The ETL canonical frame, its path/source, target, and the segment plan."""
    cfg = load_config("eda", dataset=dataset)
    if cfg.dataset is None or cfg.eda is None:
        raise ValueError("amazon_survey requires a configured dataset contract")
    artifact = canonical_path(cfg.dataset, cfg.environment)
    frame = pd.read_parquet(artifact)
    plan = SegmentPlan.build(
        frame, cfg.dataset.datetime_column, cfg.eda.max_categories
    )
    return SurveyFrame(
        frame=frame,
        artifact=artifact,
        source=cfg.dataset.path,
        dataset=cfg.dataset.name,
        target=cfg.dataset.target,
        segment_plan=plan,
    )


def _family_null(
    survey: SurveyFrame,
    family: tuple[str, ...],
    seed: int,
    iterations: int,
    quantile: float,
) -> float:
    """The selection-adjusted no-effect band over the whole declared family.

    ONE target permutation is drawn per iteration and every family member is
    measured against that same permutation, so the statistic is the MAXIMUM |rho|
    a pick from the whole ranking could reach under one no-effect world — the band
    already pays for the selection.
    """
    encoded = {
        name: RankAssociation.encode(survey.frame[name]) for name in family
    }
    masks = {
        name: survey.frame[[name, survey.target]].notna().all(axis=1).to_numpy()
        for name in family
    }
    targets = survey.frame[survey.target].to_numpy(dtype=float)
    generator = np.random.default_rng(seed)
    draws = np.empty(iterations)
    for iteration in range(iterations):
        permuted = generator.permutation(targets)
        draws[iteration] = max(
            RankAssociation.rho(encoded[name][masks[name]], permuted[masks[name]])
            for name in family
        )
    return float(np.percentile(draws, quantile * 100.0))


def _half_sample(
    survey: SurveyFrame, feature: str, seed: int, iterations: int, confidence: float
) -> float:
    """|rho| on one seeded half of the frame (its own n, a robustness check)."""
    measurement = MeasurementEstimator().estimate(
        survey.frame,
        RankAssociation(feature, survey.target),
        MeasurementSpec(
            bootstrap="iid",
            sample_size=len(survey.frame) // 2,
            iterations=iterations,
            seed=seed,
            confidence=confidence,
        ),
        name="rho_half_sample",
    )
    return measurement.value


def _segments(
    survey: SurveyFrame, feature: str
) -> list[SegmentEvidence]:
    labels = survey.segment_plan.labels(survey.frame)
    statistic = RankAssociation(feature, survey.target)
    evidence = []
    for name, positions in SegmentPlan.groups(labels):
        pair = survey.frame.iloc[positions][[feature, survey.target]].dropna()
        if len(pair) < 3:
            continue
        evidence.append(
            SegmentEvidence(
                segment=name,
                value=statistic.rho(*statistic.pair(pair)),
                sample_size=len(pair),
            )
        )
    return evidence


def _ranked(survey: SurveyFrame, family: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": list(family),
            "abs_spearman_rho": [
                round(RankAssociation(name, survey.target)(survey.frame), 6)
                for name in family
            ],
            "n": [
                len(survey.frame[[name, survey.target]].dropna()) for name in family
            ],
        }
    ).sort_values("abs_spearman_rho", ascending=False, ignore_index=True)


def _figure(
    survey: SurveyFrame,
    feature: str,
    decision: Decision,
    ranked: pd.DataFrame,
    half: float,
    confidence: float,
) -> object:
    figure, axes = plt.subplots(1, 3, figsize=(18, 4.8), constrained_layout=True)
    multiplier = float(stats.norm.ppf(0.5 + confidence / 2.0))

    grouped = survey.frame.groupby(feature)[survey.target]
    means = grouped.mean()
    counts = grouped.size()
    errors = multiplier * grouped.std(ddof=1) / np.sqrt(counts)
    axes[0].errorbar(
        means.index, means.to_numpy(), yerr=errors.to_numpy(), fmt="o-", color="#08519c", capsize=3
    )
    for level, mean, count in zip(means.index, means.to_numpy(), counts.to_numpy()):
        axes[0].annotate(
            f"n={count}", (level, mean), textcoords="offset points", xytext=(0, 8), fontsize=8
        )
    axes[0].set_xlabel(f"reported importance of {feature} (1-5)")
    axes[0].set_ylabel(f"mean {survey.target} (1-5)")
    axes[0].set_title("satisfaction rises with the reported behaviour")

    points = [("all rows", decision.value, decision.interval_low, decision.interval_high)]
    points.append(("half sample", half, half, half))
    points += [
        (item.segment, item.value, item.value, item.value)
        for item in decision.evidence.segments
    ]
    positions = np.arange(len(points))
    values = [value for _, value, *_ in points]
    lows = [low for *_, low, _ in points]
    highs = [high for *_, high in points]
    axes[1].errorbar(
        positions,
        values,
        yerr=[np.subtract(values, lows), np.subtract(highs, values)],
        fmt="o",
        color="#08519c",
        capsize=4,
    )
    null_band = decision.null_threshold or 0.0
    axes[1].axhline(decision.threshold, color="#d62728", linestyle="--", linewidth=1)
    axes[1].axhline(null_band, color="#7f7f7f", linestyle=":", linewidth=1.2)
    axes[1].text(0.02, decision.threshold, f"rule {decision.threshold:.3f}", fontsize=8, va="bottom")
    axes[1].text(
        0.02, null_band, f"selection null {null_band:.3f}", fontsize=8, va="bottom"
    )
    axes[1].set_xticks(
        positions, [name for name, *_ in points], rotation=30, ha="right", fontsize=8
    )
    axes[1].set_ylabel(f"|Spearman rho| vs {survey.target}")
    axes[1].set_title("full-sample interval, half sample, semantic slices")
    axes[1].set_ylim(0, max(max(highs), decision.threshold, 1.05 * null_band) * 1.15)

    axes[2].bar(ranked["feature"], ranked["abs_spearman_rho"], color="#9ecae1")
    axes[2].axhline(decision.threshold, color="#d62728", linestyle="--", linewidth=1, label="rule")
    axes[2].axhline(
        null_band, color="#7f7f7f", linestyle=":", linewidth=1.2, label="selection null"
    )
    axes[2].set_ylabel(f"|Spearman rho| vs {survey.target}")
    axes[2].set_title("EDA-ranked behaviour family")
    axes[2].tick_params(axis="x", rotation=45, labelsize=7)
    axes[2].legend(fontsize=8)
    return figure


def main() -> None:
    config = _experiment_config()
    seed = int(config["seed"])
    iterations = int(config["iterations"])
    feature = str(config["feature"])
    family = tuple(str(name) for name in config["family"])
    rule = _rule(STEM)
    survey = _load(str(config["dataset"]))
    RESULTS.mkdir(parents=True, exist_ok=True)

    measurement = MeasurementEstimator().estimate(
        survey.frame,
        RankAssociation(feature, survey.target),
        MeasurementSpec(
            bootstrap="iid",
            null="shuffle",
            null_column=feature,
            iterations=iterations,
            seed=seed,
        ),
        name=rule.metric,
        unit="abs_spearman_rho",
    )
    selection_null = _family_null(survey, family, seed, iterations, rule.null_quantile)
    # The published rule IS the selection-adjusted one: the record must not claim a
    # wider null than the threshold it was judged against.
    measurement = measurement.model_copy(update={"null_threshold": selection_null})
    half = _half_sample(survey, feature, seed, iterations, measurement.confidence)
    ranked = _ranked(survey, family)
    evidence = Evidence(
        baseline=(
            f"no-effect permutation of {survey.target} over the whole {len(family)}-feature "
            f"family (max |rho| at this n)"
        ),
        segments=_segments(survey, feature),
        sensitivity={
            f"rho_half_sample_n{len(survey.frame) // 2}": half,
            f"family_max_rho_null_q{rule.null_quantile:.2f}_n{len(survey.frame)}": selection_null,
            f"kendall_tau_abs_n{len(survey.frame)}": abs(
                float(
                    stats.kendalltau(
                        *RankAssociation(feature, survey.target).pair(survey.frame)
                    ).statistic
                )
            ),
        },
        falsifier=(
            "the interval's lower bound falls at or below the selection-adjusted null "
            f"{selection_null:.3f} — the feature would then be indistinguishable from a "
            f"{survey.target} shuffle over the same family"
        ),
        provenance={
            "dataset": survey.dataset,
            "artifact": str(survey.artifact),
            "artifact_sha256": hashlib.sha256(survey.artifact.read_bytes()).hexdigest()[:16],
            "source": survey.source,
            "rows": str(measurement.sample_size),
            "seed": str(measurement.seed),
            "resampling": measurement.resampling,
            "selection": (
                f"family of {len(family)} EDA-ranked features declared in {CONFIG.name}; "
                "the null takes their max |rho|"
            ),
            "segments": survey.segment_plan.rule,
            "exclusions": "none beyond the pairwise-complete pair and the ETL contract",
        },
    )
    decision = measurement.to_decision(
        rule,
        question=(
            f"Does the reported importance of {feature} track {survey.target} beyond the "
            "selection-adjusted no-effect null?"
        ),
        evidence=evidence,
        reference=0.0,
    )
    # The record must be judged against the same number it publishes: the family
    # null is the rule's null band, and the configured minimum effect must dominate
    # it (otherwise the rule's floor, not the null, would set the threshold).
    published = decision.evidence.sensitivity[
        f"family_max_rho_null_q{rule.null_quantile:.2f}_n{len(survey.frame)}"
    ]
    if not (decision.null_threshold == selection_null == published):
        raise ValueError(
            "selection null drifted between the measurement, the record and its "
            f"evidence (record={decision.null_threshold}, sensitivity={published})"
        )
    if selection_null >= rule.min_effect:
        raise ValueError(
            f"the selection-adjusted null {selection_null:.4f} no longer clears the "
            f"configured minimum effect {rule.min_effect:.4f}; restate the decision rule"
        )
    output = StepOutput(results_dir=RESULTS, stem=STEM, decision=decision, source=Path(__file__))
    output.add_table("ranked", ranked)
    output.add_table(
        "segments",
        pd.DataFrame(
            {
                "segment": [item.segment for item in decision.evidence.segments],
                "abs_spearman_rho": [
                    round(item.value, 6) for item in decision.evidence.segments
                ],
                "n": [item.sample_size for item in decision.evidence.segments],
            }
        ),
    )
    output.save_figure(
        _figure(survey, feature, decision, ranked, half, measurement.confidence)
    )
    plt.close("all")
    report, figure, decision_path = output.write()
    print(f"report: {report}")
    print(f"figure: {figure}")
    print(
        f"decision: {decision_path} -> {decision.verdict} "
        f"({decision.metric} {decision.value:.4f} "
        f"[{decision.interval_low:.4f}, {decision.interval_high:.4f}] "
        f"vs {decision.threshold:.4f}, null {decision.null_threshold:.4f}, "
        f"n={decision.sample_size})"
    )


if __name__ == "__main__":
    main()
