"""Basic statistical analysis — group-level summary and ANOVA."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from scipy import stats as sp_stats

from broadway.config.schema import PipelineConfig
from broadway.data.loader import load

logger = logging.getLogger(__name__)


def run(cfg: PipelineConfig) -> None:
    if not cfg.dataset or not cfg.stats:
        raise ValueError("stats step requires dataset and stats config")
    df = load(cfg.dataset)
    group_col = cfg.stats.group_column
    if group_col not in df.columns:
        raise ValueError(f"group column '{group_col}' not found in data")
    groups = {
        g: df[df[group_col] == g][cfg.dataset.target].dropna()
        for g in cfg.stats.group_values
        if not df[df[group_col] == g].empty
    }
    if len(groups) < 2:
        logger.warning("stats: fewer than 2 groups — skipping ANOVA")
        return
    _, p_value = sp_stats.f_oneway(*groups.values())
    result = {
        "test": "one-way ANOVA",
        "group_column": group_col,
        "p_value": round(float(p_value), 6),
        "group_stats": {g: {"mean": round(float(vals.mean()), 2), "count": int(len(vals))} for g, vals in groups.items()},
    }
    out_dir = Path("artifacts/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stats.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info(f"stats: ANOVA p={p_value:.4f}, results written to {out_path}")
