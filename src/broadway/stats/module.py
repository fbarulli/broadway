"""Basic statistical analysis — group-level summary and ANOVA."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from broadway.analysis.contracts import AnalysisMode, require_mode
from broadway.config.schema import PipelineConfig
from broadway.data.loader import load
from broadway.stats.anova import run_anova
from broadway.stats.plan import save_plan

logger = logging.getLogger(__name__)


def run(cfg: PipelineConfig) -> None:
    if not cfg.dataset or not cfg.stats:
        raise ValueError("stats step requires dataset and stats config")
    analysis = require_mode(cfg.analysis, AnalysisMode.HYPOTHESIS)
    logger.info("stats: goal — %s", analysis.goal)
    df = load(cfg.dataset)
    group_col = cfg.stats.group_column
    if group_col not in df.columns:
        raise ValueError(f"group column '{group_col}' not found in data")
    groups: dict[str, np.ndarray] = {
        g: df[df[group_col] == g][cfg.dataset.target].dropna().to_numpy()
        for g in cfg.stats.group_values
        if not df[df[group_col] == g].empty
    }
    plan = run_anova(groups)
    plan = plan.model_copy(update={"analysis_goal": analysis.goal})
    out_dir = Path(cfg.stats.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg.stats.output_file
    save_plan(plan, out_path)
    logger.info(
        f"stats: {plan.test_name} p={plan.statistics['p_value']:.4f}, "
        f"plan written to {out_path}"
    )
