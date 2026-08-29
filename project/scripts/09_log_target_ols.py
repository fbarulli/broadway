"""
09_log_target_ols.py

Follow-up promised but not yet run in the residual diagnostics step.

The baseline OLS (08) had:
  - R^2 = 0.642
  - Breusch-Pagan: p=0 (heteroskedastic)
  - Jarque-Bera: skew=2.264, kurtosis=25.491 (heavy right tail)
  - Residual std ranged 6.3 (Manhattan) -> 21.4 (Bronx)

From 06_anova_comparison.py we already know log-transforming duration fixed
SKEW at the group level but did NOT fix VARIANCE (boroughs have genuinely
different spread, not just different shape). This script checks whether that
same pattern holds for residuals of a regression model, not just raw group
distributions -- it's a different question even though the earlier finding
predicts the answer.

Two things are checked:
  1. log(trip_duration_minutes) ~ trip_distance + C(pickup_borough)
     -> re-run Breusch-Pagan + Jarque-Bera on the new residuals.
  2. Same untransformed model as 08, refit with HC3 robust standard errors.
     This does NOT fix heteroskedasticity -- it just makes the reported
     p-values/CIs valid despite it. Useful if these coefficients are ever
     reported to anyone; not a modeling fix.
"""

import numpy as np

from broadway.stats import regression
from project import data


def run_diagnostics(model: object, label: str) -> None:
    result = regression.bp_jb(model)
    bp_stat = result["bp_stat"]
    bp_pval = result["bp_pval"]
    jb_pval = result["jb_pval"]
    skew = result["skew"]
    kurtosis = result["kurtosis"]

    print(f"--- {label} ---")
    print(f"R^2 = {model.rsquared:.3f}")
    print(f"Breusch-Pagan: stat={bp_stat:.2f}, p={bp_pval:.4g} "
          f"-> {'heteroskedastic' if bp_pval < 0.05 else 'homoskedastic'}")
    print(f"Jarque-Bera: skew={skew:.3f}, kurtosis={kurtosis:.3f}, "
          f"p={jb_pval:.4g} "
          f"-> {'non-normal' if jb_pval < 0.05 else 'normal'}")
    print()


def main() -> None:
    print("Loading cached stratified sample...")
    df = data.load_stratified_sample()

    # Duration is always > 0 post-filtering (config/data.py enforces
    # min_trip_duration_minutes), so plain log is safe -- no log1p needed.
    df["log_duration"] = np.log(df[data.TARGET_COL])

    print("\n=== 1. Log-target model ===")
    log_model = regression.fit_ols(
        df,
        f"log_duration ~ {data.TRIP_DISTANCE_COL} + C({data.PICKUP_BOROUGH_COL})",
    )
    print(log_model.summary())
    run_diagnostics(log_model, "log(duration) ~ distance + borough")

    print("=== 2. Original-scale model, robust (HC3) SEs ===")
    baseline = regression.fit_ols(
        df,
        f"{data.TARGET_COL} ~ {data.TRIP_DISTANCE_COL} + C({data.PICKUP_BOROUGH_COL})",
    )
    baseline_robust = regression.fit_robust(baseline)
    print(baseline_robust.summary())
    print(
        "\nNote: HC3 does not change coefficients or R^2, only standard "
        "errors/p-values/CIs. Compare the std err column above to the "
        "nonrobust version from 08 -- expect them to widen, since HC3 "
        "correctly accounts for the borough-level variance differences "
        "instead of assuming a single pooled variance."
    )


if __name__ == "__main__":
    main()
