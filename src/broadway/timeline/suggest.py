from __future__ import annotations

from broadway.timeline.models import (
    Alternative,
    AnalysisDecision,
    AnalysisStep,
    StepStatus,
    Suggestion,
)
from broadway.timeline.sequence import WalkthroughConfig, WalkthroughSequence


def _alternative(label: str, intent: str, rationale: str, command: str = "") -> Alternative:
    return Alternative(label=label, command=command, intent=intent, rationale=rationale)


def _suggest_describe(step, cfg: WalkthroughConfig, analysis_name: str) -> Suggestion:
    if step.status == StepStatus.WARNING:
        rationale = []
        ratio = step.result_summary.get("imbalance_ratio")
        absent = step.result_summary.get("absent_groups", 0)
        if absent:
            rationale.append(f"absent_groups={absent}")
        if ratio is not None:
            rationale.append(f"imbalance_ratio={ratio}")
        return Suggestion(
            step_id=step.step_id,
            headline="Group sizes are imbalanced or a group is absent",
            rationale=rationale,
            command=f"ds-pipeline walkthrough --analysis {analysis_name}",
            alternatives=[
                _alternative(
                    "Inspect the data audit to confirm the row population",
                    "support",
                    "Confirm the row population matches the intended analysis units",
                    "ds-pipeline audit",
                ),
                _alternative(
                    "Confirm the imbalance reflects the population, not a sampling artifact",
                    "challenge",
                    "Rule out a sampling artifact before trusting the imbalance signal",
                ),
            ],
        )
    return Suggestion(
        step_id=step.step_id,
        headline="Group sizes look adequate",
        rationale=["proceed to the normality check"],
        command=f"ds-pipeline walkthrough --analysis {analysis_name}",
        alternatives=[],
    )


def _suggest_normality(step, cfg: WalkthroughConfig, analysis_name: str) -> Suggestion:
    if step.status == StepStatus.WARNING:
        rationale = []
        for group, values in step.result_summary.items():
            skew = values.get("skew")
            kurtosis = values.get("kurtosis")
            shapiro = values.get("shapiro_p")
            if skew is not None:
                rationale.append(f"{group}: skew={skew}")
            if kurtosis is not None:
                rationale.append(f"{group}: kurtosis={kurtosis}")
            if shapiro is not None:
                rationale.append(f"{group}: shapiro_p={shapiro}")
        return Suggestion(
            step_id=step.step_id,
            headline="Distributional shape is flagged in some groups",
            rationale=rationale,
            command=f"ds-pipeline walkthrough --analysis {analysis_name}",
            alternatives=[
                _alternative(
                    "A significant Shapiro on large n is expected — inspect the Q-Q plots before treating it as decisive",
                    "challenge",
                    "Shapiro is sensitive to sample size; confirm with the Q-Q plots",
                ),
                _alternative(
                    "Review skew/kurtosis per group to locate the affected groups",
                    "support",
                    "Locate which groups drive the flagged shape",
                ),
            ],
        )
    return Suggestion(
        step_id=step.step_id,
        headline="Distributional shape looks broadly reasonable",
        rationale=["proceed to the variance check"],
        command=f"ds-pipeline walkthrough --analysis {analysis_name}",
        alternatives=[],
    )


def _suggest_variance(step, cfg: WalkthroughConfig, analysis_name: str) -> Suggestion:
    if step.status == StepStatus.WARNING:
        statistic = step.result_summary.get("statistic")
        p_value = step.result_summary.get("p_value")
        return Suggestion(
            step_id=step.step_id,
            headline="Variance evidence favors considering Welch's ANOVA",
            rationale=[f"Levene statistic={statistic}, p_value={p_value}"],
            command=(
                f'ds-pipeline decide --analysis {analysis_name} '
                f'--method welch --reason "..."'
            ),
            alternatives=[
                _alternative(
                    "Inspect group variances and sample sizes",
                    "support",
                    "Confirm the variance signal is not an artifact of group size",
                ),
                _alternative(
                    "Check whether the variance signal is dominated by a very small group",
                    "challenge",
                    "A single small group can drive Levene's statistic",
                ),
                _alternative(
                    "Consider Kruskal-Wallis if the estimand and distributional concerns make a rank-based analysis appropriate",
                    "alternative",
                    "Rank-based analysis is robust to both shape and variance",
                    command=(
                        f'ds-pipeline decide --analysis {analysis_name} '
                        f'--method kruskal --reason "..."'
                    ),
                ),
            ],
        )
    return Suggestion(
        step_id=step.step_id,
        headline="No evidence of unequal variances",
        rationale=["proceed to the omnibus analysis"],
        command=f"ds-pipeline decide --analysis {analysis_name} --method anova --reason \"...\"",
        alternatives=[],
    )


def _suggest_omnibus(step, cfg: WalkthroughConfig, analysis_name: str) -> Suggestion:
    if step.result_summary.get("passed") is True:
        return Suggestion(
            step_id=step.step_id,
            headline="Omnibus result is significant",
            rationale=["at least one group mean differs from the others"],
            command=(
                f'ds-pipeline decide --analysis {analysis_name} '
                f'--kind posthoc --method games_howell --reason "..."'
            ),
            alternatives=[
                _alternative(
                    "Confirm the effect size is meaningful, not just the p-value",
                    "challenge",
                    "Statistical significance does not imply practical significance",
                ),
                _alternative(
                    "Review which specific pairs to compare",
                    "support",
                    "Select the post-hoc comparisons that answer the question",
                ),
            ],
        )
    return Suggestion(
        step_id=step.step_id,
        headline="Omnibus result is not significant",
        rationale=["no evidence of a group mean difference"],
        command=f"ds-pipeline walkthrough --analysis {analysis_name}",
        alternatives=[
            _alternative(
                "Consider whether the sample is underpowered for the effect you care about",
                "challenge",
                "A null result may reflect power, not absence of effect",
            ),
        ],
    )


def _suggest_decide_omnibus(cfg: WalkthroughConfig, analysis_name: str) -> Suggestion:
    return Suggestion(
        step_id="decide_omnibus",
        headline="A principal-method decision is required",
        rationale=["choose the omnibus method guided by the evidence gathered"],
        command=(
            f'ds-pipeline decide --analysis {analysis_name} '
            f'--method <method> --reason "..."'
        ),
        alternatives=[
            _alternative(
                "Review the variance and normality evidence before choosing",
                "support",
                "The evidence gathered so far constrains the method",
            ),
            _alternative(
                "Verify the variance evidence is not dominated by one small group",
                "challenge",
                "Rule out a single group driving the variance signal",
            ),
        ],
    )


def _suggest_decide_posthoc(cfg: WalkthroughConfig, analysis_name: str) -> Suggestion:
    return Suggestion(
        step_id="decide_posthoc",
        headline="A post-hoc comparison decision is required",
        rationale=["choose the post-hoc method guided by the omnibus result"],
        command=(
            f'ds-pipeline decide --analysis {analysis_name} '
            f'--kind posthoc --method games_howell --reason "..."'
        ),
        alternatives=[
            _alternative(
                "Only Games-Howell is currently implemented for unequal variance",
                "support",
                "Games-Howell is the available method for unequal variance",
            ),
            _alternative(
                "Confirm post-hoc is warranted given the omnibus effect size",
                "challenge",
                "Verify the omnibus effect size justifies pairwise comparisons",
            ),
        ],
    )


def _suggest_posthoc(step, cfg: WalkthroughConfig, analysis_name: str) -> Suggestion:
    return Suggestion(
        step_id=step.step_id,
        headline="Post-hoc comparisons complete",
        rationale=["pairwise comparisons have been computed"],
        command=f"ds-pipeline walkthrough --analysis {analysis_name}",
        alternatives=[
            _alternative(
                "Review Cohen's d / Hedges' g per pair for practical significance",
                "support",
                "Assess practical significance alongside statistical significance",
            ),
        ],
    )


def _suggest_not_started(step_id: str, analysis_name: str) -> Suggestion:
    return Suggestion(
        step_id=step_id,
        headline=f"Next step: {step_id}",
        rationale=["this step has not produced evidence yet"],
        command=f"ds-pipeline walkthrough --analysis {analysis_name}",
        alternatives=[],
    )


def suggest_after(
    step_id: str,
    steps: list[AnalysisStep],
    decisions: list[AnalysisDecision],
    cfg: WalkthroughConfig,
    analysis_name: str,
) -> Suggestion | None:
    by_id = {s.step_id: s for s in steps}
    step = by_id.get(step_id)
    if step_id == "conclusion":
        return None
    if step_id == "decide_omnibus":
        return _suggest_decide_omnibus(cfg, analysis_name)
    if step_id == "decide_posthoc":
        return _suggest_decide_posthoc(cfg, analysis_name)
    if step is None:
        return _suggest_not_started(step_id, analysis_name)
    if step.step_id == "describe_groups":
        return _suggest_describe(step, cfg, analysis_name)
    if step.step_id == "normality":
        return _suggest_normality(step, cfg, analysis_name)
    if step.step_id == "variance":
        return _suggest_variance(step, cfg, analysis_name)
    if step.step_id == "omnibus":
        return _suggest_omnibus(step, cfg, analysis_name)
    if step.step_id == "posthoc":
        return _suggest_posthoc(step, cfg, analysis_name)
    return None


def suggest_next(
    sequence: WalkthroughSequence,
    steps: list[AnalysisStep],
    decisions: list[AnalysisDecision],
    cfg: WalkthroughConfig,
    analysis_name: str,
) -> Suggestion | None:
    by_id = {s.step_id: s for s in steps}
    decided_ids = {d.id for d in decisions if d.status == "resolved"}
    for step in sorted(sequence.steps, key=lambda s: s.order):
        if step.kind == "decision":
            kind = step.id.removeprefix("decide_")
            if kind not in decided_ids:
                return suggest_after(step.id, steps, decisions, cfg, analysis_name)
        else:
            if step.id not in by_id:
                return suggest_after(step.id, steps, decisions, cfg, analysis_name)
    return None
