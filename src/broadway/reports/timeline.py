from __future__ import annotations

from broadway.formatting import humanize_float, humanize_pvalue
from broadway.reports.results import (
    attrition_line,
    derive_status,
    effect_size_lines,
    humanize_summary,
    is_figure_ref,
    posthoc_headline,
    posthoc_pair_rows,
)
from broadway.timeline.models import AnalysisDecision, AnalysisStep
from broadway.timeline.sequence import WalkthroughSequence


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
        status, all_prior_resolved = derive_status(
            step, steps_by_id, decisions, all_prior_resolved
        )
        lines.append(f"| {step.order} | {step.label} | {step.question} | {status} |")

    for step in seq_steps:
        persisted = steps_by_id.get(step.id)
        if persisted is None:
            continue
        lines.append("")
        lines.append(f"## {step.label}")
        lines.append("")
        lines.append(f"- ramification: {persisted.ramification}")
        machine_refs = [r for r in persisted.evidence_refs if not is_figure_ref(r)]
        refs = ", ".join(machine_refs) if machine_refs else "-"
        lines.append(f"- evidence_refs: {refs}")
        lines.append("- result_summary:")
        for bullet in humanize_summary(persisted.result_summary):
            lines.append(f"  - {bullet}")
        if persisted.step_id == "omnibus":
            for line in effect_size_lines(persisted.result_summary):
                lines.append(f"  - {line}")
        if persisted.step_id == "describe_groups":
            lines.append(f"  - {attrition_line(persisted.result_summary)}")
        if persisted.step_id == "posthoc":
            lines.append(f"  - {posthoc_headline(persisted.result_summary)}:")
            for row in posthoc_pair_rows(persisted.result_summary):
                lines.append(
                    f"    - {row.get('a')} vs {row.get('b')}: "
                    f"p {humanize_pvalue(float(row.get('p_value')))}, "
                    f"Cohen's d {humanize_float(float(row.get('cohens_d')))}, "
                    f"Hedges' g {humanize_float(float(row.get('hedges_g')))}"
                )
        for fig in persisted.figures:
            lines.append(f"![{fig.caption}]({fig.path})")

    return "\n".join(lines)
