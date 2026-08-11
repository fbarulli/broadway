"""Orchestrator — imports each step module and calls run(cfg) in sequence."""

from __future__ import annotations

import importlib
import logging

from broadway.config.loader import STEP_MODULES
from broadway.config.schema import PipelineConfig

logger = logging.getLogger(__name__)


def run(cfg: PipelineConfig, steps: list[str]) -> None:
    for step in steps:
        if step == "discover":
            logger.warning("discover skipped in pipeline — run via CLI directly")
            continue
        logger.info(f"step starting: {step}")
        module = importlib.import_module(STEP_MODULES[step])
        try:
            module.run(cfg)
        except Exception:
            logger.exception(f"step failed: {step}")
            raise
        logger.info(f"step complete: {step}")
