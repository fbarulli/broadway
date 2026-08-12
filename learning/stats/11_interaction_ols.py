"""
11_interaction_ols.py

The baseline model (08) was trip_duration_minutes ~ trip_distance +
C(pickup_borough) -- this assumes the SAME distance->duration slope in
every borough, with borough only shifting the intercept up or down. That's
a strong, probably wrong assumption: a mile in Manhattan traffic and a mile
on a Staten Island highway almost certainly convert to duration at
different rates, not just different baselines.

This script fits the interaction model:
  trip_duration_minutes ~ trip_distance * C(pickup_borough)

which expands to distance + borough + (distance x borough), giving each
borough its own slope AND its own intercept. Compare:
  - R^2 vs baseline (08): does letting slopes vary meaningfully improve fit?
  - F-test (anova_lm) on nested models: is the improvement significant, or
    just noise from added parameters?
  - Per-borough slopes: which boroughs actually have a different
    distance-duration relationship, and does that ranking make sense
    (e.g. do highway-heavy boroughs show a flatter minutes-per-mile slope
    than dense Manhattan traffic)?
  - Residual diagnostics again: interaction terms fix non-linearity, not
    heteroskedasticity -- expect Breusch-Pagan to still reject.
"""

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.anova import anova_lm

from _config import (
    RANDOM_STATE, PICKUP_BOROUGH_COL, TARGET_COL, TRIP_DISTANCE_COL,
    load_stratified_sample,
)


def main():
    print("Loading cached stratified sample...")
    df = load_stratified_sample()

    print("\n=== Baseline (no interaction, from 08) ===")
    baseline = smf.ols(
        f"{TARGET_COL} ~ {TRIP_DISTANCE_COL} + C({PICKUP_BOROUGH_COL})", data=df
    ).fit()
    print(f"R^2 = {baseline.rsquared:.4f}")

    print("\n=== Interaction model: distance * borough ===")
    interaction = smf.ols(
        f"{TARGET_COL} ~ {TRIP_DISTANCE_COL} * C({PICKUP_BOROUGH_COL})", data=df
    ).fit()
    print(interaction.summary())

    print("\n=== Nested F-test: does the interaction earn its parameters? ===")
    comparison = anova_lm(baseline, interaction)
    print(comparison)
    print(
        "\nIf the p-value in the comparison above is < 0.05, the "
        "interaction terms explain significantly more variance than the "
        "added parameters would by chance -- i.e. slopes genuinely differ "
        "by borough, not just intercepts."
    )

    print("\n=== Implied minutes-per-mile slope by borough ===")
    base_slope = interaction.params[TRIP_DISTANCE_COL]
    print(f"Reference borough slope: {base_slope:.3f} min/mile")
    interaction_prefix = f"{TRIP_DISTANCE_COL}:C({PICKUP_BOROUGH_COL})"
    for name, coef in interaction.params.items():
        if interaction_prefix in name:
            borough = name.split("[T.")[-1].rstrip("]")
            print(f"{borough}: {base_slope + coef:.3f} min/mile "
                  f"(delta={coef:+.3f})")

    print("\n=== Residual diagnostics (interaction model) ===")
    resid = interaction.resid
    exog = interaction.model.exog
    bp_stat, bp_pval, _, _ = het_breuschpagan(resid, exog)
    jb_stat, jb_pval, skew, kurtosis = jarque_bera(resid)
    print(f"Breusch-Pagan: p={bp_pval:.4g} "
          f"-> {'still heteroskedastic' if bp_pval < 0.05 else 'homoskedastic'}")
    print(f"Jarque-Bera: skew={skew:.3f}, kurtosis={kurtosis:.3f}, "
          f"p={jb_pval:.4g}")
    print(
        "\nExpectation: interaction should raise R^2 and may reduce "
        "skew/kurtosis somewhat if part of the heavy tail was mis-specified "
        "non-linearity rather than pure variance -- but Breusch-Pagan will "
        "most likely still reject, since interaction terms address slope "
        "mis-specification, not the underlying variance-heterogeneity "
        "problem confirmed by Levene's test."
    )


if __name__ == "__main__":
    main()