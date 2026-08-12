"""Experiment design, power analysis, causal analysis — thin pipeline step."""

from __future__ import annotations

import logging

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
    logger.info("causal design: %s", design.model_dump_json(indent=2))
