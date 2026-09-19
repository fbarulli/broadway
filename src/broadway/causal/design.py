"""Power analysis, minimum detectable effect, sample size calculation."""

from __future__ import annotations

import math

from statsmodels.stats.power import TTestIndPower

from broadway.causal.contracts import ExperimentDesign


def minimum_detectable_effect(sample_size: int, power: float, alpha: float) -> float:
    analysis = TTestIndPower()
    return float(
        analysis.solve_power(
            effect_size=None,
            nobs1=sample_size,
            alpha=alpha,
            power=power,
            ratio=1.0,
            alternative="two-sided",
        )
    )


def design_experiment(
    effect_size: float,
    power: float,
    alpha: float,
    treatment_column: str,
    outcome_column: str,
) -> ExperimentDesign:
    analysis = TTestIndPower()
    sample_size = math.ceil(
        analysis.solve_power(
            effect_size=effect_size,
            power=power,
            alpha=alpha,
            ratio=1.0,
            alternative="two-sided",
        )
    )
    mde = minimum_detectable_effect(sample_size, power, alpha)
    return ExperimentDesign(
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        power=power,
        alpha=alpha,
        effect_size=effect_size,
        sample_size=sample_size,
        mde=mde,
    )
