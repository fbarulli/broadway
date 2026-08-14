from __future__ import annotations

import logging
from pathlib import Path

from broadway.analysis.contracts import AnalysisContract, AnalysisMode, require_mode
from broadway.config.schema import PipelineConfig
from broadway.lineage.models import SampleSpec
from broadway.reports import paths
from broadway.reports.timeline import render_timeline
from broadway.timeline import module as timeline_module
from broadway.timeline import runners
from broadway.timeline.models import AnalysisStep
from broadway.timeline.sequence import WalkthroughSequence, load_walkthrough_sequence

logger = logging.getLogger(__name__)

TIMELINE_DIR = timeline_module.TIMELINE_DIR

EXECUTABLE_STEPS = {"describe_groups", "normality", "variance"}

_OMNIBUS_METHODS = ["welch", "anova", "kruskal"]


def _omnibus_decided(analysis: str) -> bool:
    return any(d.kind == "omnibus" for d in timeline_module.load_decisions(analysis))


def _write_timeline(analysis: str, sequence: WalkthroughSequence) -> None:
    paths.TIMELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    paths.TIMELINE_PATH.write_text(
        render_timeline(
            analysis,
            sequence,
            timeline_module.load_steps(analysis),
            timeline_module.load_decisions(analysis),
        ),
        encoding="utf-8",
    )


def _print_decision_required(
    analysis: AnalysisContract, steps: list[AnalysisStep]
) -> None:
    by_id = {s.step_id: s for s in steps}
    lines = ["=" * 60, "DECISION REQUIRED — decide_omnibus", "=" * 60, ""]
    lines.append(f"Goal: {analysis.goal}")
    lines.append("")
    lines.append("Evidence gathered:")
    describe = by_id.get("describe_groups")
    if describe is not None:
        rs = describe.result_summary
        lines.append(
            f"  - describe_groups: total_n={rs.get('total_n')}, "
            f"imbalance_ratio={rs.get('imbalance_ratio')}, "
            f"absent_groups={rs.get('absent_groups')}"
        )
    normality = by_id.get("normality")
    if normality is not None:
        lines.append(f"  - normality: {normality.status.value}")
    variance = by_id.get("variance")
    if variance is not None:
        rs = variance.result_summary
        lines.append(
            f"  - variance: Levene statistic={rs.get('statistic')}, p_value={rs.get('p_value')}"
        )
    lines.append("")
    lines.append("Eligible methods:")
    for method in _OMNIBUS_METHODS:
        lines.append(f"  - {method}")
    lines.append("")
    lines.append(
        f'Next: ds-pipeline decide --analysis {analysis.name} --method welch --reason "..."'
    )
    print("\n".join(lines))


def run(cfg: PipelineConfig, sample: SampleSpec | None, force: bool) -> None:
    analysis = require_mode(cfg.analysis, AnalysisMode.HYPOTHESIS)
    if analysis.hypothesis is None:
        raise ValueError("hypothesis mode requires a 'hypothesis' block (group_column, group_values)")
    if cfg.dataset is None:
        raise ValueError("walkthrough requires a dataset config")
    sequence = load_walkthrough_sequence()
    df, group_column, source_group_column, groups = runners.load_frame_and_groups(cfg, sample)
    out_dir = TIMELINE_DIR / analysis.name
    figures_dir = paths.FIGURES_DIR
    source = "sample" if sample else "canonical"
    sample_name = sample.name if sample else None
    source_path = sample.path if sample else str(runners.canonical_path(cfg.dataset, cfg.environment))
    group_values = analysis.hypothesis.group_values
    target = cfg.dataset.target

    completed = 0
    for step in sorted(sequence.steps, key=lambda s: s.order):
        if step.id == "decide_omnibus":
            if _omnibus_decided(analysis.name):
                logger.info("decide_omnibus already resolved — continuing past the gate")
                continue
            _write_timeline(analysis.name, sequence)
            _print_decision_required(analysis, timeline_module.load_steps(analysis.name))
            return
        if step.id not in EXECUTABLE_STEPS:
            continue
        if not force and timeline_module.load_step(analysis.name, step.id) is not None:
            logger.info("skipped %s (already exists)", step.id)
            continue
        if step.id == "describe_groups":
            step_result = runners.run_describe(
                analysis, step.order, step.question, df, group_column,
                source_group_column, group_values, target, source_path,
                sample_name, source, out_dir,
            )
        elif step.id == "normality":
            step_result = runners.run_normality(
                analysis, step.order, step.question, groups, out_dir,
                figures_dir, source, sample_name,
            )
        elif step.id == "variance":
            step_result = runners.run_variance(
                analysis, step.order, step.question, groups, out_dir, source, sample_name,
            )
        else:
            continue
        timeline_module.save_step(step_result)
        completed += 1

    _write_timeline(analysis.name, sequence)
    print(f"walkthrough: completed {completed} step(s)")
