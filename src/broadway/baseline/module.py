from __future__ import annotations

import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from broadway.analysis.contracts import AnalysisMode
from broadway.baseline import causal, hypothesis, prediction
from broadway.baseline.contracts import BaselineResult, load_result, save_result
from broadway.config.schema import PipelineConfig
from broadway.data.loader import load
from broadway.lineage.ids import node_id
from broadway.lineage.records import write_record
from broadway.reports.decision import Decision, DecisionRule, Evidence, SegmentEvidence
from broadway.reports.measurement import MeasurementEstimator, MeasurementSpec
from broadway.reports.step_output import StepOutput
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
    # DOCUMENTED SILENCE (determinism ledger d): wall-clock bytes — created_at
    # stamps run time into persisted artifacts by design; byte-stability
    # awaits a freeze flag (pinned-timestamp config/env).
    return ArtifactTrace(
        created_at=datetime.now(UTC),
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
    assert cfg.analysis is not None
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


class BaselineDecision:
    """Publish which naive baseline the candidate must beat (mean vs median).

    The metric is ``mean_mae / median_mae``: above 1 the mean predictor is
    worse, so the median is the honest naive reference. Measured (and
    day-blocked when a datetime column exists) on the same rows the baseline
    scores, with the two MAE values as the evidence slices.
    """

    def __init__(self, cfg: PipelineConfig, result: BaselineResult, df: pd.DataFrame) -> None:
        assert cfg.analysis is not None and cfg.dataset is not None
        self._cfg = cfg
        self._result = result
        self._df = df

    def rule(self) -> DecisionRule:
        assert self._cfg.analysis is not None
        return DecisionRule(
            metric="mean_over_median_mae",
            direction="max",
            min_effect=1.0,
            min_sample_size=100,
            action="adopt the median as the naive baseline",
            owner=self._cfg.analysis.name or "baseline",
        )

    def _frame(self) -> pd.DataFrame:
        assert self._cfg.dataset is not None
        target = self._cfg.dataset.target
        frame = pd.DataFrame({"y": self._df[target].to_numpy(dtype=float)})
        column = self._cfg.dataset.datetime_column
        if column is not None and column in self._df.columns:
            frame["day"] = pd.to_datetime(self._df[column]).dt.floor("D")
        return frame

    @staticmethod
    def _ratio(frame: pd.DataFrame) -> float:
        mean_mae = float((frame["y"] - frame["y"].mean()).abs().mean())
        median_mae = float((frame["y"] - frame["y"].median()).abs().mean())
        return mean_mae / median_mae if median_mae else 0.0

    def decision(self) -> Decision:
        frame = self._frame()
        block = "day" if "day" in frame.columns else None
        spec = MeasurementSpec(
            bootstrap="block" if block else "iid",
            block_column=block,
            iterations=200,
            seed=7,
        )
        measurement = MeasurementEstimator().estimate(
            frame, self._ratio, spec, name=self.rule().metric, unit="ratio"
        )
        details = self._result.details
        mean_mae = float(details.get("mean_mae", 0.0))
        median_mae = float(details.get("median_mae", 0.0))
        midpoint = len(frame) // 2
        evidence = Evidence(
            baseline="mean and median predict the same value (ratio 1)",
            segments=[
                SegmentEvidence(
                    segment="first_half",
                    value=self._ratio(frame.iloc[:midpoint]),
                    sample_size=midpoint,
                ),
                SegmentEvidence(
                    segment="second_half",
                    value=self._ratio(frame.iloc[midpoint:]),
                    sample_size=len(frame) - midpoint,
                ),
            ],
            sensitivity={
                f"mean_mae_n{len(frame)}": mean_mae,
                f"median_mae_n{len(frame)}": median_mae,
            },
            falsifier="the mean/median MAE ratio interval falls to 1 (either ties)",
            provenance={
                "artifact": self._cfg.baseline.output_file if self._cfg.baseline else "baseline",
                "dataset": self._cfg.dataset.name if self._cfg.dataset else "",
                "seed": str(measurement.seed),
                "analysis_sample": str(measurement.sample_size),
                "block": block or "iid",
            },
        )
        return measurement.to_decision(
            self.rule(),
            question="Is the mean predictor worse than the median, so the candidate must beat the median?",
            evidence=evidence,
        )

    def publish(self, out_dir: Path) -> Decision:
        decision = self.decision()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        details = self._result.details
        figure, axis = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
        labels = ["mean predictor", "median predictor"]
        values = [float(details.get("mean_mae", 0.0)), float(details.get("median_mae", 0.0))]
        bars = axis.bar(labels, values, color=["#9ecae1", "#08519c"])
        for bar in bars:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():,.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axis.set_ylabel("holdout MAE")
        axis.set_title("naive baselines: mean vs median")
        output = StepOutput(results_dir=out_dir, stem="baseline", decision=decision)
        output.save_figure(figure)
        plt.close("all")
        output.write()
        return decision


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
    if result.mode == AnalysisMode.PREDICTION and cfg.dataset is not None:
        decision = BaselineDecision(cfg, result, load(cfg.dataset)).publish(out_dir)
        logger.info(
            "baseline decision: %s — mean/median MAE ratio %.4f",
            decision.verdict,
            decision.value,
        )
    logger.info(f"baseline: {result.strategy} {result.metric}={result.value:.4f} -> {out_path}")
    parents = (
        [node_id("analysis", cfg.analysis.name), node_id("profile", cfg.dataset.name)]
        if cfg.dataset is not None
        else [node_id("analysis", cfg.analysis.name)]
    )
    write_record(node_id("baseline", cfg.analysis.name), "baseline", str(out_path), parents)
