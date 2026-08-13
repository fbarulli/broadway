from __future__ import annotations

import logging
from pathlib import Path

from broadway.analysis.contracts import AnalysisMode
from broadway.baseline import causal, hypothesis, prediction
from broadway.baseline.contracts import BaselineResult, save_result
from broadway.config.schema import PipelineConfig
from broadway.data.loader import load

logger = logging.getLogger(__name__)


def _compute_baseline(cfg: PipelineConfig) -> BaselineResult:
    mode = cfg.analysis.mode
    if mode == AnalysisMode.PREDICTION:
        if not cfg.dataset:
            raise ValueError("prediction baseline requires a dataset config")
        df = load(cfg.dataset)
        return prediction.run(df, cfg.dataset.target, cfg.dataset.task)
    if mode == AnalysisMode.HYPOTHESIS:
        if not cfg.dataset:
            raise ValueError("hypothesis baseline requires a dataset config")
        if not cfg.stats:
            raise ValueError("hypothesis baseline requires stats config (group_column/group_values)")
        df = load(cfg.dataset)
        return hypothesis.run(df, cfg.dataset.target, cfg.stats.group_column, cfg.stats.group_values)
    if mode == AnalysisMode.CAUSAL:
        if not cfg.causal:
            raise ValueError("causal baseline requires causal config")
        return causal.run(cfg.causal)
    raise ValueError(f"unsupported analysis mode: {mode}")


def run(cfg: PipelineConfig) -> None:
    if not cfg.analysis:
        raise ValueError("baseline step requires an analysis contract (--analysis)")
    if not cfg.baseline:
        raise ValueError("baseline step requires baseline config")
    result = _compute_baseline(cfg)
    out_dir = Path(cfg.baseline.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg.baseline.output_file
    save_result(result, out_path)
    logger.info(f"baseline: {result.strategy} {result.metric}={result.value:.4f} -> {out_path}")
