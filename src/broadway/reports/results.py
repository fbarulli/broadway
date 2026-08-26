from __future__ import annotations

import re

from broadway.formatting import humanize_float, humanize_pvalue
from broadway.reports import paths
from broadway.timeline.models import (
    RENDERED_STATUS,
    AnalysisDecision,
    AnalysisStep,
    StepStatus,
)
from broadway.timeline.sequence import WalkthroughSequence

COMPLETED_STATUSES = frozenset({StepStatus.COMPLETED, StepStatus.NOTE, StepStatus.WARNING})


def _is_pvalue_key(key: str) -> bool:
    lowered = key.lower()
    return (
        key == "p"
        or lowered.endswith(("_p", "p_value"))
        or "pval" in lowered
    )


def humanize_value(value: object, key: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if _is_pvalue_key(key):
            return humanize_pvalue(value)
        return humanize_float(value)
    return str(value)


def _flatten(summary: dict, prefix: str = "") -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    for key, value in summary.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.extend(_flatten(value, full_key))
        elif isinstance(value, list):
            continue
        else:
            out.append((full_key, value))
    return out


def slugify(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def statistic_label(step_id: str, summary: dict) -> str:
    if step_id == "variance":
        return "Levene statistic"
    if step_id == "omnibus":
        method = str(summary.get("method", ""))
        if method == "kruskal":
            return "H"
        if method in ("anova", "welch"):
            return "F"
    return "statistic"


def humanize_summary(summary: dict, step_id: str = "") -> list[str]:
    lines: list[str] = []
    for key, value in _flatten(summary):
        if isinstance(value, str) and value == "":
            continue
        label = statistic_label(step_id, summary) if key == "statistic" else key
        lines.append(f"{label}: {humanize_value(value, key)}")
    return lines


def attrition_line(summary: dict) -> str:
    n_used = summary.get("n_used")
    n_total = summary.get("n_total")
    n_excluded = summary.get("n_excluded", 0)
    reason = summary.get("exclusion_reason", "")
    if n_excluded and int(n_excluded) > 0:
        return f"N used {n_used} of {n_total} ({n_excluded} excluded: {reason}); see audit"
    return "none excluded"


def effect_size_lines(summary: dict) -> list[str]:
    if "epsilon_squared" in summary:
        eps = humanize_float(float(summary["epsilon_squared"]))
        return [f"rank-based ε² = {eps}: proportion of variance in ranks explained by group membership."]
    if summary.get("effect_size") == "not_computed":
        return ["Rank-based effect size deliberately not computed — epsilon² pending."]
    if "eta_squared" in summary and "omega_squared" in summary:
        eta = humanize_float(float(summary["eta_squared"]))
        omega = humanize_float(float(summary["omega_squared"]))
        return [
            (f"eta² = {eta}: proportion of outcome variance explained by group membership "
             "(can be inflated under extreme imbalance)"),
            f"omega² = {omega}: corrects for small-sample bias; the more conservative estimate",
        ]
    return []


def posthoc_headline(summary: dict) -> str:
    total = summary.get("pairs", 0)
    significant = summary.get("significant_pairs", 0)
    return f"{significant} of {total} pairs significant"


def posthoc_pair_rows(summary: dict) -> list[dict]:
    return summary.get("significant_pair_details", [])


def _decision_for(step_id: str, decisions: list[AnalysisDecision]) -> AnalysisDecision | None:
    expected = step_id.removeprefix("decide_")
    for decision in decisions:
        if decision.id == expected and decision.kind == expected:
            return decision
    return None


def derive_status(
    step,
    steps_by_id: dict[str, AnalysisStep],
    decisions: list[AnalysisDecision],
    all_prior_resolved: bool,
) -> tuple[str, bool]:
    persisted = steps_by_id.get(step.id)
    if persisted is not None:
        return RENDERED_STATUS.get(persisted.status, persisted.status.value), True
    if step.kind == "decision":
        if not all_prior_resolved:
            return "blocked", False
        decision = _decision_for(step.id, decisions)
        if decision is not None:
            return "completed", True
        return "awaiting decision", False
    if not all_prior_resolved:
        return "blocked", False
    return "not started", False


def _render_index(
    analysis: str,
    sequence: WalkthroughSequence,
    steps_by_id: dict[str, AnalysisStep],
    decisions: list[AnalysisDecision],
) -> str:
    lines = [f"# Results — {analysis}", "", "| Step | Status |", "| --- | --- |"]
    all_prior_resolved = True
    for step in sorted(sequence.steps, key=lambda s: s.order):
        status, all_prior_resolved = derive_status(
            step, steps_by_id, decisions, all_prior_resolved
        )
        persisted = steps_by_id.get(step.id)
        linkable = persisted is not None and persisted.status in COMPLETED_STATUSES
        name = f"[{step.label}]({slugify(step.label)}.md)" if linkable else step.label
        lines.append(f"| {name} | {status} |")
    return "\n".join(lines)


def _render_step_page(seq_step, step: AnalysisStep) -> str:
    lines = [f"# {seq_step.label}", ""]
    lines.append("## Question")
    lines.append("")
    lines.append(seq_step.question)
    lines.append("")
    lines.append("## What was run")
    lines.append("")
    lines.append(step.method or "-")
    lines.append("")
    lines.append("## What it found")
    lines.append("")
    bullets = humanize_summary(step.result_summary, step.step_id)
    if bullets:
        lines.extend(f"- {b}" for b in bullets)
    else:
        lines.append("- nothing to report")
    lines.append("")
    lines.append("## Why it matters")
    lines.append("")
    lines.append(step.ramification)
    lines.append("")
    if step.step_id in ("omnibus", "conclusion"):
        effect = effect_size_lines(step.result_summary)
        if effect:
            lines.append("## Effect size")
            lines.append("")
            lines.extend(f"- {e}" for e in effect)
            lines.append("")
    if step.step_id == "describe_groups":
        lines.append("## Attrition")
        lines.append("")
        lines.append(attrition_line(step.result_summary))
        lines.append("")
    if step.step_id == "posthoc":
        rows = posthoc_pair_rows(step.result_summary)
        lines.append("## Significant pairs")
        lines.append("")
        lines.append(posthoc_headline(step.result_summary))
        lines.append("")
        if rows:
            lines.append("| Pair | p | Cohen's d | Hedges' g | Note |")
            lines.append("| --- | --- | --- | --- | --- |")
            for row in rows:
                lines.append(
                    f"| {row.get('a')} vs {row.get('b')} | "
                    f"{humanize_pvalue(float(row['p_value']))} | "
                    f"{humanize_float(float(row['cohens_d']))} | "
                    f"{humanize_float(float(row['hedges_g']))} | "
                    f"{row.get('effect_size_note')} |"
                )
        else:
            lines.append("none")
        lines.append("")
    if step.figures:
        lines.append("## Figures")
        lines.append("")
        lines.extend(f"![{fig.caption}](../{fig.path})" for fig in step.figures)
        lines.append("")
    return "\n".join(lines)


def render_results(
    analysis: str,
    sequence: WalkthroughSequence,
    steps: list[AnalysisStep],
    decisions: list[AnalysisDecision],
) -> dict[str, str]:
    steps_by_id = {s.step_id: s for s in steps}
    pages: dict[str, str] = {}
    pages["index.md"] = _render_index(analysis, sequence, steps_by_id, decisions)
    for step in sorted(sequence.steps, key=lambda s: s.order):
        persisted = steps_by_id.get(step.id)
        if persisted is None or persisted.status not in COMPLETED_STATUSES:
            continue
        pages[f"{slugify(step.label)}.md"] = _render_step_page(step, persisted)
    return pages


def write_results(
    analysis: str,
    sequence: WalkthroughSequence,
    steps: list[AnalysisStep],
    decisions: list[AnalysisDecision],
) -> None:
    pages = render_results(analysis, sequence, steps, decisions)
    paths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    completed_slugs = {name[:-3] for name in pages if name.endswith(".md") and name != "index.md"}
    for path in paths.RESULTS_DIR.glob("*.md"):
        if path.name == "index.md":
            continue
        if path.stem not in completed_slugs:
            path.unlink()
    for name, content in pages.items():
        (paths.RESULTS_DIR / name).write_text(content, encoding="utf-8")
