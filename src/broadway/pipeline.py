"""Orchestrator — imports each step module and calls run(cfg) in sequence."""

from __future__ import annotations

import importlib
import logging
from functools import partial

from broadway.config.loader import STEP_MODULES
from broadway.config.schema import PipelineConfig
from broadway.observability.profiling import StepProfiler

logger = logging.getLogger(__name__)


def run(cfg: PipelineConfig, steps: list[str]) -> None:
    profiler = StepProfiler.from_default_config()
    dataset = cfg.dataset.name if cfg.dataset else None
    for step in steps:
        if step == "discover":
            logger.warning("discover skipped in pipeline — run via CLI directly")
            continue
        logger.info(f"step starting: {step}")
        module = importlib.import_module(STEP_MODULES[step])
        profiler.run(step, dataset, partial(module.run, cfg))
        logger.info(f"step complete: {step}")
