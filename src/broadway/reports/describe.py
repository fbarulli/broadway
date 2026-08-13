from __future__ import annotations

from pathlib import Path

from broadway.reports.markdown import render_result
from broadway.stats.describe import GroupSummary


def load_artifact(stats_dir: Path) -> GroupSummary:
    return GroupSummary.model_validate_json(
        (stats_dir / "describe.json").read_text(encoding="utf-8")
    )


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def render(summary: GroupSummary) -> str:
    sample = "\n".join(
        [
            f"- sample_name: {summary.sample_name}",
            f"- sample_role: {summary.sample_role}",
            f"- source_path: {summary.source_path}",
            f"- source_group_column: {summary.source_group_column}",
            f"- group_column: {summary.group_column}",
        ]
    )

    group_rows = ["| group | n | mean | std |", "| --- | --- | --- | --- |"]
    for name, stat in summary.groups.items():
        group_rows.append(
            f"| {name} | {stat.n} | {_fmt(stat.mean)} | {_fmt(stat.std)} |"
        )
    groups_table = "\n".join(group_rows)

    imbalance = (
        f"imbalance_ratio: {summary.imbalance_ratio}\n\n"
        f"absent_groups: {summary.absent_groups or 'none'}"
    )

    warnings = (
        "\n".join(f"- {w}" for w in summary.warnings)
        if summary.warnings
        else "none"
    )

    figures = "\n".join(
        [
            "[describe_boxplot.png](figures/describe_boxplot.png)",
            "[describe_group_sizes.png](figures/describe_group_sizes.png)",
        ]
    )

    sections = [
        ("Question", "whether trip duration differs across pickup boroughs"),
        ("Sample", sample),
        ("Groups", groups_table),
        ("Imbalance", imbalance),
        ("Warnings", warnings),
        ("Figures", figures),
    ]
    return render_result("describe", sections)


def headline(summary: GroupSummary) -> str:
    return (
        f"describe: {summary.total_n} rows across {len(summary.groups)} groups; "
        f"imbalance ratio {summary.imbalance_ratio}; "
        f"absent: {summary.absent_groups or 'none'}"
    )
