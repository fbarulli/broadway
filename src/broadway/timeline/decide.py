from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from broadway.analysis.contracts import AnalysisContract
from broadway.timeline.models import AnalysisDecision
from broadway.timeline.sequence import (
    load_walkthrough_config,
    load_walkthrough_sequence,
)


def _question_for(kind: str) -> str:
    for step in load_walkthrough_sequence().steps:
        if step.id == f"decide_{kind}":
            return step.question
    raise ValueError(f"no decision step found for kind '{kind}'")


def record(
    analysis: AnalysisContract,
    kind: Literal["omnibus", "posthoc"],
    method: str,
    reason: str,
) -> AnalysisDecision:
    spec = load_walkthrough_config().decisions.get(kind)
    if spec is None:
        raise ValueError(f"unknown decision kind '{kind}'")
    if method not in spec.methods:
        raise ValueError(
            f"method '{method}' is not allowed for kind '{kind}'; "
            f"allowed: {spec.methods}"
        )
    return AnalysisDecision(
        analysis=analysis.name,
        id=kind,
        kind=kind,
        question=_question_for(kind),
        method=method,
        reason=[reason],
        status="resolved",
        parents=list(spec.parents),
        decided_at=datetime.now(UTC).isoformat(),
    )
