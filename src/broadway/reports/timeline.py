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


def _derive_status(
    step: WalkthroughStepConfig,
    steps_by_id: dict[str, AnalysisStep],
    decisions: list[AnalysisDecision],
    all_prior_resolved: bool,
) -> tuple[str, bool]:
    persisted = steps_by_id.get(step.id)
    if persisted is not None:
        status = COMPLETED if persisted.status == StepStatus.COMPLETED else WARNING
        return status, True

    if step.kind == "decision":
        if not all_prior_resolved:
            return BLOCKED, False
        decision = _decision_for(step.id, decisions)
        if decision is not None:
            return f"{DECIDED} (method={decision.method})", True
        return DECISION_REQUIRED, False

    if not all_prior_resolved:
        return BLOCKED, False
    return NOT_STARTED, False


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
    all_prior_resolved = True
    for step in seq_steps:
        status, all_prior_resolved = _derive_status(
            step, steps_by_id, decisions, all_prior_resolved
        )
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
