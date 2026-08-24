"""Basic statistical analysis — group-level summary and ANOVA."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from broadway.analysis.contracts import AnalysisMode, require_mode
from broadway.config.schema import PipelineConfig
from broadway.data.loader import canonical_path
from broadway.lineage.ids import node_id
from broadway.lineage.models import SampleSpec
from broadway.lineage.records import write_record
from broadway.stats.anova import run_anova
from broadway.stats.groups import build_declared_groups
from broadway.stats.plan import save_plan

logger = logging.getLogger(__name__)


def run(cfg: PipelineConfig, sample: SampleSpec | None = None) -> None:
    if not cfg.dataset or not cfg.stats:
        raise ValueError("stats step requires dataset and stats config")
    analysis = require_mode(cfg.analysis, AnalysisMode.HYPOTHESIS)
    logger.info("stats: goal — %s", analysis.goal)
    if analysis.hypothesis is None:
        raise ValueError("hypothesis mode requires a 'hypothesis' block (group_column, group_values)")
    if sample is None:
        canonical = canonical_path(cfg.dataset, cfg.environment)
        if not canonical.exists():
            raise FileNotFoundError(f"canonical dataset not found: {canonical} — run the etl step first")
        df = pd.read_parquet(canonical)
    else:
        sample_path = Path(sample.path)
        if not sample_path.exists():
            raise FileNotFoundError(f"sample dataset not found: {sample_path}")
        df = pd.read_parquet(sample_path)
    group_col = analysis.hypothesis.group_column
    if sample is not None:
        group_col = sample.column_mapping.get(analysis.hypothesis.group_column, analysis.hypothesis.group_column)
    if group_col not in df.columns:
        raise ValueError(f"group column '{group_col}' not found in data")
    groups, absent_groups = build_declared_groups(
        df, group_col, analysis.hypothesis.group_values, cfg.dataset.target
    )
    if absent_groups:
        raise ValueError(f"declared groups absent from data: {absent_groups}")
    plan = run_anova(groups, small_group_threshold=cfg.stats.min_rows_for_sampling)
    plan = plan.model_copy(
        update={
            "analysis_goal": analysis.goal,
            "sample_name": sample.name if sample else None,
            "sample_role": sample.role if sample else None,
        }
    )
    out_dir = Path(cfg.stats.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg.stats.output_file
    save_plan(plan, out_path)
    logger.info(
        f"stats: {plan.test_name} p={plan.statistics['p_value']:.4f}, "
        f"plan written to {out_path}"
    )
    write_record(
        node_id("stats", analysis.name),
        "stats",
        str(out_path),
        [node_id("baseline", analysis.name), node_id("analysis", analysis.name)],
        sample_name=sample.name if sample else None,
        sample_role=sample.role if sample else None,
    )
