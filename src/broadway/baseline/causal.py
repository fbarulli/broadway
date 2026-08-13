from __future__ import annotations

from broadway.analysis.contracts import AnalysisMode
from broadway.baseline.contracts import BaselineResult
from broadway.causal.design import design_experiment
from broadway.config.schema import CausalStep


def run(causal: CausalStep) -> BaselineResult:
    design = design_experiment(
        effect_size=causal.effect_size,
        power=causal.power,
        alpha=causal.alpha,
        treatment_column=causal.treatment_column,
        outcome_column=causal.outcome_column,
    )
    return BaselineResult(
        mode=AnalysisMode.CAUSAL,
        strategy="power_analysis",
        metric="sample_size",
        value=float(design.sample_size),
        details={
            "mde": design.mde,
            "effect_size": design.effect_size,
            "power": design.power,
            "alpha": design.alpha,
            "treatment_column": design.treatment_column,
            "outcome_column": design.outcome_column,
        },
        notes=[
            f"estimand: effect of {causal.treatment_column} on {causal.outcome_column}"
        ],
    )
