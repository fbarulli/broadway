from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel, ConfigDict

from broadway import viz
from broadway.analysis.contracts import AnalysisMode, require_mode
from broadway.config.schema import PipelineConfig
from broadway.lineage.ids import node_id
from broadway.lineage.models import SampleRole, SampleSpec
from broadway.lineage.records import write_record

logger = logging.getLogger(__name__)


class GroupStat(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int
    mean: float | None
    std: float | None


class GroupSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    group_column: str
    source_group_column: str
    target: str
    total_n: int
    source_path: str
    sample_name: str
    sample_role: SampleRole
    groups: dict[str, GroupStat]      # ALL configured groups, incl. n=0
    absent_groups: list[str]
    imbalance_ratio: float            # evidence only — NO balanced/unbalanced verdict
    proportions: dict[str, float]
    warnings: list[str]


def describe(df: pd.DataFrame, group_column: str, source_group_column: str, group_values: list[str], target: str, source_path: str, sample_name: str, sample_role: SampleRole) -> GroupSummary:
    total_n = len(df)
    groups: dict[str, GroupStat] = {}
    absent: list[str] = []
    for g in group_values:
        vals = df[df[source_group_column] == g][target].dropna()
        n = len(vals)
        if n == 0:
            absent.append(g)
            groups[g] = GroupStat(n=0, mean=None, std=None)
        else:
            groups[g] = GroupStat(n=n, mean=float(vals.mean()), std=float(vals.std()))
    present_n = [s.n for s in groups.values() if s.n > 0]
    imbalance_ratio = (max(present_n) / min(present_n)) if len(present_n) >= 2 else 0.0
    proportions = {g: (s.n / total_n if total_n else 0.0) for g, s in groups.items()}
    warnings: list[str] = []
    return GroupSummary(
        group_column=group_column, source_group_column=source_group_column, target=target, total_n=total_n,
        source_path=source_path, sample_name=sample_name, sample_role=sample_role,
        groups=groups, absent_groups=absent,
        imbalance_ratio=round(imbalance_ratio, 4), proportions=proportions, warnings=warnings,
    )


def _draw_group_distribution(ax, data, labels, group_column, target, group_values, colors) -> None:
    # boxplot of target by group, present groups only (absent groups have no data)
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xlabel(group_column)
    ax.set_ylabel(target)
    ax.set_title("Group distribution (target by group)")
    absent = [g for g, v in zip(group_values, data) if len(v) == 0]
    if absent:
        ax.text(0.99, 0.02, f"absent (n=0): {', '.join(absent)}", transform=ax.transAxes, ha="right", va="bottom", color="red")
    viz.despine(ax)


def _draw_group_sizes(ax, summary: GroupSummary, names: list[str], sizes: list[int], colors) -> None:
    ax.bar(names, sizes, color=colors)
    ax.set_ylabel("n")
    ax.set_xlabel(summary.group_column)
    ax.set_title("Group sizes (imbalance evidence)")
    for i, s in enumerate(sizes):
        ax.text(i, s, f"n={s}", ha="center", va="bottom")
    ax.text(0.99, 0.02, f"imbalance ratio = {summary.imbalance_ratio}", transform=ax.transAxes, ha="right", va="bottom")
    viz.despine(ax)


def plot_describe_figures(
    df: pd.DataFrame,
    source_group_column: str,
    group_column: str,
    group_values: list[str],
    target: str,
    summary: GroupSummary,
    out_path: Path,
) -> None:
    data = [df[df[source_group_column] == g][target].dropna().to_numpy() for g in group_values]
    labels = [f"{g} (n={len(v)})" for g, v in zip(group_values, data)]
    names = list(summary.groups.keys())
    sizes = [summary.groups[g].n for g in names]
    colors = viz.palette_colors(len(data))

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 10), layout="constrained")
    _draw_group_distribution(ax_top, data, labels, group_column, target, group_values, colors)
    _draw_group_sizes(ax_bottom, summary, names, sizes, colors)
    fig.savefig(out_path)
    plt.close(fig)


def run(cfg: PipelineConfig, sample: SampleSpec) -> None:
    if not cfg.dataset or not cfg.stats:
        raise ValueError("stats describe requires dataset and stats config")
    analysis = require_mode(cfg.analysis, AnalysisMode.HYPOTHESIS)
    if analysis.hypothesis is None:
        raise ValueError("hypothesis mode requires a 'hypothesis' block (group_column, group_values)")
    sample_path = Path(sample.path)
    if not sample_path.exists():
        raise FileNotFoundError(f"sample dataset not found: {sample_path}")
    df = pd.read_parquet(sample_path)
    group_column = analysis.hypothesis.group_column
    source_group_column = sample.column_mapping.get(group_column, group_column)
    group_values = analysis.hypothesis.group_values
    if source_group_column not in df.columns:
        raise ValueError(f"group column '{source_group_column}' not found in sample data")
    summary = describe(df, group_column, source_group_column, group_values, cfg.dataset.target, str(sample.path), sample.name, sample.role)
    out_dir = Path(cfg.stats.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "describe.json"
    json_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    logger.info("describe: %d groups, total_n=%d, imbalance=%.2f -> %s", len(group_values), summary.total_n, summary.imbalance_ratio, json_path)
    write_record(
        node_id("describe", analysis.name), "describe", str(json_path),
        [node_id("etl", cfg.dataset.name), node_id("analysis", analysis.name)],
        sample_name=sample.name, sample_role=sample.role,
    )
