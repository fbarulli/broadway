from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from broadway.analysis.contracts import AnalysisMode
from broadway.baseline import causal, hypothesis, prediction
from broadway.baseline.contracts import BaselineResult, load_result, save_result
from broadway.config.schema import PipelineConfig
from broadway.data.loader import load
from broadway.lineage.ids import node_id
from broadway.lineage.records import write_record
from broadway.trace import ArtifactTrace

logger = logging.getLogger(__name__)


def _git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return proc.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _build_trace(cfg: PipelineConfig) -> ArtifactTrace:
    return ArtifactTrace(
        created_at=datetime.now(timezone.utc),
        commit=_git_commit(),
        dataset=cfg.dataset.name if cfg.dataset else None,
        analysis_goal=cfg.analysis.goal if cfg.analysis else None,
    )


def load_persisted(cfg: PipelineConfig) -> BaselineResult | None:
    if cfg.baseline is None:
        return None
    path = Path(cfg.baseline.output_dir) / cfg.baseline.output_file
    if not path.exists():
        return None
    return load_result(path)


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
        if cfg.analysis is None or cfg.analysis.hypothesis is None:
            raise ValueError("hypothesis baseline requires an analysis contract with a hypothesis group")
        df = load(cfg.dataset)
        return hypothesis.run(
            df,
            cfg.dataset.target,
            cfg.analysis.hypothesis.group_column,
            cfg.analysis.hypothesis.group_values,
        )
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
    result = result.model_copy(update={"trace": _build_trace(cfg)})
    out_dir = Path(cfg.baseline.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg.baseline.output_file
    save_result(result, out_path)
    logger.info(f"baseline: {result.strategy} {result.metric}={result.value:.4f} -> {out_path}")
    parents = (
        [node_id("analysis", cfg.analysis.name), node_id("profile", cfg.dataset.name)]
        if cfg.dataset is not None
        else [node_id("analysis", cfg.analysis.name)]
    )
    write_record(node_id("baseline", cfg.analysis.name), "baseline", str(out_path), parents)
