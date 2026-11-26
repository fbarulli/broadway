"""Experiment design, power analysis, causal analysis — thin pipeline step."""

from __future__ import annotations

import logging
from pathlib import Path

from statsmodels.stats.power import TTestIndPower

from broadway.analysis.contracts import AnalysisMode, require_mode
from broadway.causal.contracts import ExperimentDesign, save_design
from broadway.causal.design import design_experiment
from broadway.config.schema import PipelineConfig
from broadway.lineage.ids import node_id
from broadway.lineage.records import write_record
from broadway.reports.decision import Decision, DecisionRule, Evidence, SegmentEvidence
from broadway.reports.step_output import StepOutput

logger = logging.getLogger(__name__)


class DesignDecision:
    """Publish the design verdict: does the planned n reach the required power?

    The metric is the ACHIEVED power at the design's sample size for the
    declared effect; the rule is the configured power floor. A design that
    cannot reach it fails loud before any experiment runs. The power curve is
    the figure; the effect-size and alpha variants are the sensitivity.
    """

    def __init__(self, cfg: PipelineConfig, design: ExperimentDesign) -> None:
        self._cfg = cfg
        self._design = design

    def _achieved_power(self, sample_size: int) -> float:
        design = self._design
        return float(
            TTestIndPower().solve_power(
                effect_size=design.effect_size,
                nobs1=sample_size,
                alpha=design.alpha,
                power=None,
                ratio=1.0,
                alternative="two-sided",
            )
        )

    def decision(self) -> Decision:
        design = self._design
        achieved = self._achieved_power(design.sample_size)
        evidence = Evidence(
            baseline="the power floor declared in the causal config",
            segments=[
                SegmentEvidence(
                    segment="per_arm", value=achieved, sample_size=max(design.sample_size // 2, 1)
                ),
                SegmentEvidence(
                    segment="pooled", value=achieved, sample_size=design.sample_size
                ),
            ],
            sensitivity={
                f"mde_at_design_n{design.sample_size}": design.mde,
                f"effect_size_n{design.sample_size}": design.effect_size,
                f"alpha_n{design.sample_size}": design.alpha,
            },
            falsifier="the achieved power at the design n falls below the configured power",
            provenance={
                "artifact": (self._cfg.dataset.name if self._cfg.dataset else "causal_design.json"),
                "treatment": design.treatment_column,
                "outcome": design.outcome_column,
                "design_sample_size": str(design.sample_size),
            },
        )
        return Decision.evaluate(
            rule=DecisionRule(
                metric="achieved_power",
                direction="max",
                min_effect=design.power,
                min_sample_size=2,
                action="run the experiment at the designed sample size",
                owner="causal",
            ),
            value=achieved,
            interval=(achieved, achieved),
            sample_size=design.sample_size,
            question="Can the designed sample size detect the declared effect at the required power?",
            evidence=evidence,
            unit="power",
        )

    def publish(self, out_dir: Path) -> Decision:
        """Write the record plus the power curve."""
        decision = self.decision()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        design = self._design
        sizes = [max(int(design.sample_size * factor), 2) for factor in (0.25, 0.5, 1.0, 2.0)]
        powers = [self._achieved_power(size) for size in sizes]
        figure, axis = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
        axis.plot(sizes, powers, marker="o", color="#0072B2")
        axis.axhline(design.power, color="#D55E00", linestyle="--", label="power floor")
        axis.axvline(design.sample_size, color="#999999", linestyle=":", label="design n")
        axis.set_xlabel("sample size per arm")
        axis.set_ylabel("achieved power")
        axis.set_title("power curve at the declared effect")
        axis.legend(fontsize=8)
        output = StepOutput(results_dir=out_dir, stem="causal", decision=decision)
        output.save_figure(figure)
        plt.close("all")
        output.write()
        return decision


def run(cfg: PipelineConfig) -> None:
    if not cfg.causal:
        raise ValueError("causal step requires causal config")
    analysis = require_mode(cfg.analysis, AnalysisMode.CAUSAL)
    logger.info("causal: goal — %s", analysis.goal)
    causal = cfg.causal
    design = design_experiment(
        effect_size=causal.effect_size,
        power=causal.power,
        alpha=causal.alpha,
        treatment_column=causal.treatment_column,
        outcome_column=causal.outcome_column,
    )
    design = design.model_copy(update={"analysis_goal": analysis.goal})
    out_dir = Path(causal.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / causal.output_file
    save_design(design, out_path)
    decision = DesignDecision(cfg, design).publish(out_dir)
    logger.info(
        "causal decision: %s — achieved power %.4f at n=%d",
        decision.verdict,
        decision.value,
        design.sample_size,
    )
    logger.info("causal design written to %s", out_path)
    write_record(
        node_id("causal", analysis.name),
        "causal",
        str(out_path),
        [node_id("baseline", analysis.name), node_id("analysis", analysis.name)],
    )
