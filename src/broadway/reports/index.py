from __future__ import annotations

from pathlib import Path

from broadway.reports import paths, registry
from broadway.reports.markdown import render_result
from broadway.reports.sequence import load_stats_sequence


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
        ("Navigation", "[lineage graph](lineage/graph.md)"),
    ]
    return render_result("Broadway Results Index", sections)
