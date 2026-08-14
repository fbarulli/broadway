from __future__ import annotations

from broadway.timeline.models import AnalysisDecision, AnalysisStep, StepStatus
from broadway.timeline.sequence import WalkthroughStepConfig, WalkthroughSequence

COMPLETED = "✓ completed"
WARNING = "⚠ warning"
DECIDED = "✓ decided"
DECISION_REQUIRED = "◆ decision required"
BLOCKED = "⏸ blocked"
NOT_STARTED = "○ not started"


def _decision_for(step_id: str, decisions: list[AnalysisDecision]) -> AnalysisDecision | None:
    expected = step_id.removeprefix("decide_")
    for decision in decisions:
        if decision.id == expected and decision.kind == expected:
            return decision
    return None


def _is_blocked(
    step: WalkthroughStepConfig,
    seq_steps: list[WalkthroughStepConfig],
    decisions: list[AnalysisDecision],
) -> bool:
    idx = next(i for i, s in enumerate(seq_steps) if s.id == step.id)
    for upstream in seq_steps[:idx]:
        if upstream.kind == "decision" and _decision_for(upstream.id, decisions) is None:
            return True
    return False


def _status(
    step: WalkthroughStepConfig,
    steps_by_id: dict[str, AnalysisStep],
    decisions: list[AnalysisDecision],
    seq_steps: list[WalkthroughStepConfig],
) -> str:
    persisted = steps_by_id.get(step.id)
    if persisted is not None:
        if persisted.status == StepStatus.COMPLETED:
            return COMPLETED
        return WARNING

    if step.kind == "decision":
        decision = _decision_for(step.id, decisions)
        if decision is not None:
            return f"{DECIDED} (method={decision.method})"
        return DECISION_REQUIRED

    if _is_blocked(step, seq_steps, decisions):
        return BLOCKED
    return NOT_STARTED


def _summary_lines(summary: dict) -> list[str]:
    return [f"  - {key}: {value}" for key, value in summary.items()]


def render_timeline(
    analysis: str,
    sequence: WalkthroughSequence,
    steps: list[AnalysisStep],
    decisions: list[AnalysisDecision],
) -> str:
    steps_by_id = {s.step_id: s for s in steps}
    seq_steps = sorted(sequence.steps, key=lambda s: s.order)

    lines = [f"# Timeline — {analysis}", ""]
    lines.append("| # | Step | Question | Status |")
    lines.append("| --- | --- | --- | --- |")
    for step in seq_steps:
        status = _status(step, steps_by_id, decisions, seq_steps)
        lines.append(f"| {step.order} | {step.id} | {step.question} | {status} |")

    for step in seq_steps:
        persisted = steps_by_id.get(step.id)
        if persisted is None:
            continue
        lines.append("")
        lines.append(f"## {step.id}")
        lines.append("")
        lines.append(f"- ramification: {persisted.ramification}")
        refs = ", ".join(persisted.evidence_refs) if persisted.evidence_refs else "-"
        lines.append(f"- evidence_refs: {refs}")
        lines.append("- result_summary:")
        lines.extend(_summary_lines(persisted.result_summary))

    return "\n".join(lines)
