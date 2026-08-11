"""Experiment design, power analysis, causal analysis — barebones."""

from __future__ import annotations

import logging

from broadway.config.schema import PipelineConfig

logger = logging.getLogger(__name__)


def run(cfg: PipelineConfig) -> None:
    if not cfg.dataset or not cfg.causal:
        raise ValueError("causal step requires dataset and causal config")
    logger.info("causal: power=%.2f, alpha=%.2f — analysis not yet implemented", cfg.causal.power, cfg.causal.alpha)
