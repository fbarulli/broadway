from __future__ import annotations

from pathlib import Path

from broadway.reports import paths, registry
from broadway.reports.markdown import render_result
from broadway.reports.sequence import load_stats_sequence
from broadway.timeline.models import AnalysisDecision, AnalysisStep, Suggestion
from broadway.timeline.sequence import WalkthroughSequence


def _done(step: str) -> bool:
    return (paths.RESULTS_DIR / f"{step}.md").exists()


def render_index(question: str, stats_dir: Path) -> str:
    sequence = load_stats_sequence()

    rows = ["| status | step |", "| --- | --- |"]
    for step in sequence.steps:
        if _done(step):
            rows.append(f"| [x] | [{step}](results/{step}.md) |")
        else:
            rows.append(f"| [ ] | {step} |")
    results_table = "\n".join(rows)

    next_test = next((s for s in sequence.steps if not _done(s)), None)

    latest = next(
        (s for s in reversed(sequence.steps) if _done(s) and s in registry.RESULT_RENDERERS),
        None,
    )
    if latest is None:
        latest_text = "none yet"
    else:
        renderer = registry.RESULT_RENDERERS[latest]
        try:
            latest_text = renderer.headline(renderer.load_artifact(stats_dir))
        except Exception:
            latest_text = f"see results/{latest}.md"

    sections = [
        ("Question", question),
        ("Latest result", latest_text),
        ("Next test", next_test if next_test is not None else "all complete"),
        ("Results", results_table),
        ("Navigation", "[data audit](audit/index.md)\n[lineage graph](lineage/graph.md)"),
    ]
    return render_result("Broadway Results Index", sections)


def _resolved(step, by_id: dict[str, AnalysisStep], decided_ids: set[str]) -> bool:
    if step.kind == "decision":
        return step.id.removeprefix("decide_") in decided_ids
    return step.id in by_id


def render_dashboard(
    analysis: str,
    goal: str,
    sequence: WalkthroughSequence,
    steps: list[AnalysisStep],
    decisions: list[AnalysisDecision],
    suggestion: Suggestion | None,
) -> str:
    by_id = {s.step_id: s for s in steps}
    decided_ids = {d.id for d in decisions if d.status == "resolved"}
    seq_steps = sorted(sequence.steps, key=lambda s: s.order)

    completed = 0
    frontier = None
    for step in seq_steps:
        if _resolved(step, by_id, decided_ids):
            completed += 1
        elif frontier is None:
            frontier = step

    total = len(seq_steps)
    if frontier is None:
        status = "COMPLETE"
    elif frontier.kind == "decision":
        status = "DECISION REQUIRED"
    else:
        status = "IN PROGRESS"

    if suggestion is not None:
        next_action = f"{suggestion.headline}\n\n{suggestion.command}"
    else:
        next_action = "none — analysis complete"

    lines = [
        f"# {analysis}",
        "",
        "## Question",
        "",
        goal,
        "",
        "## Progress",
        "",
        f"{completed} of {total} stages completed",
        "",
        "## Status",
        "",
        status,
        "",
        "## Next action",
        "",
        next_action,
        "",
        "## Navigation",
        "",
        "- [analysis timeline](timeline.md)",
        "- [data audit](audit/index.md)",
        "- [results](results/)",
        "- [lineage graph](lineage/graph.md)",
        "",
    ]
    return "\n".join(lines)
