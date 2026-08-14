from __future__ import annotations

from datetime import datetime, timezone

from broadway.analysis.contracts import AnalysisContract
from broadway.timeline.models import AnalysisDecision
from broadway.timeline.sequence import load_walkthrough_sequence

ALLOWED_METHODS: dict[str, frozenset[str]] = {
    "omnibus": frozenset({"welch", "anova", "kruskal"}),
    "posthoc": frozenset({"games_howell"}),
}

PARENTS_BY_KIND: dict[str, list[str]] = {
    "omnibus": ["describe_groups", "normality", "variance"],
    "posthoc": ["omnibus"],
}


def _question_for(kind: str) -> str:
    for step in load_walkthrough_sequence().steps:
        if step.id == f"decide_{kind}":
            return step.question
    raise ValueError(f"no decision step found for kind '{kind}'")


def record(
    analysis: AnalysisContract,
    kind: str,
    method: str,
    reason: str,
    parents: list[str],
) -> AnalysisDecision:
    allowed = ALLOWED_METHODS.get(kind)
    if allowed is None:
        raise ValueError(f"unknown decision kind '{kind}'")
    if method not in allowed:
        raise ValueError(
            f"method '{method}' is not allowed for kind '{kind}'; "
            f"allowed: {sorted(allowed)}"
        )
    return AnalysisDecision(
        analysis=analysis.name,
        id=kind,
        kind=kind,
        question=_question_for(kind),
        method=method,
        reason=[reason],
        status="resolved",
        parents=parents,
        decided_at=datetime.now(timezone.utc).isoformat(),
    )
