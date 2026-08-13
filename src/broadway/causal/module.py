"""Experiment design, power analysis, causal analysis — thin pipeline step."""

from __future__ import annotations

import logging
from pathlib import Path

from broadway.causal.contracts import save_design
from broadway.causal.design import design_experiment
from broadway.config.schema import PipelineConfig

logger = logging.getLogger(__name__)


def run(cfg: PipelineConfig) -> None:
    if not cfg.causal:
        raise ValueError("causal step requires causal config")
    causal = cfg.causal
    design = design_experiment(
        effect_size=causal.effect_size,
        power=causal.power,
        alpha=causal.alpha,
        treatment_column=causal.treatment_column,
        outcome_column=causal.outcome_column,
    )
    out_dir = Path(causal.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / causal.output_file
    save_design(design, out_path)
    logger.info("causal design written to %s", out_path)
