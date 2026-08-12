"""
08_ols_residual_diagnostics.py

Before doing more feature engineering or modeling, check whether a baseline
OLS model's residuals actually satisfy OLS's assumptions. We already have
strong priors from the ANOVA work (04-07) that they won't:

  - Levene's test showed ~7x variance ratio across boroughs
    (Bronx std=25.6 vs Manhattan std=9.6) -> expect heteroskedasticity.
  - Every borough group is right-skewed and non-normal (Shapiro p=0)
    -> expect non-normal residuals, especially a right tail from
       slow/outlier trips and a compressed left tail (duration >= 0).

This script fits a simple baseline OLS (trip_duration_minutes ~ trip_distance
+ pickup_borough), then runs the standard residual diagnostic suite:

  1. Residuals vs Fitted plot   -> non-linearity / heteroskedasticity (visual)
  2. Q-Q plot                   -> normality (visual)
  3. Histogram of residuals     -> skew/shape (visual)
  4. Breusch-Pagan test         -> heteroskedasticity (formal)
  5. Jarque-Bera test           -> normality, valid at large n
                                    (Shapiro is NOT — scipy caps it at n=5000
                                    and it over-rejects long before that; see
                                    the CLT discussion from the ANOVA work)
  6. Durbin-Watson stat         -> autocorrelation, relevant here because
                                    pickup_datetime ordering means trips
                                    aren't independent draws

Note on scale: statsmodels OLS on 8.6M rows is unnecessary and slow for a
diagnostic pass. We take a stratified sample (by borough, to preserve the
variance structure we're testing for) capped at SAMPLE_SIZE.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt
import scipy.stats as stats

from _config import (
    RANDOM_STATE, PICKUP_BOROUGH_COL, TARGET_COL,
    TRIP_DISTANCE_COL, load_stratified_sample,
    bp_jb_diagnostics,
)


def main():
    print("Loading cached stratified sample...")
    df = load_stratified_sample()
    model = smf.ols(
        f"{TARGET_COL} ~ {TRIP_DISTANCE_COL} + C({PICKUP_BOROUGH_COL})",
        data=df,
    ).fit()
    return model, df


def plot_diagnostics(model, out_path="residual_diagnostics.png"):
    fitted = model.fittedvalues
    resid = model.resid

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Residuals vs Fitted
    axes[0].scatter(fitted, resid, s=2, alpha=0.15)
    axes[0].axhline(0, color="red", linewidth=1)
    axes[0].set_xlabel("Fitted values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Fitted")

    # 2. Q-Q plot
    sm.qqplot(resid, line="45", ax=axes[1], markersize=2, alpha=0.15)
    axes[1].set_title("Q-Q Plot of Residuals")

    # 3. Histogram
    axes[2].hist(resid, bins=100)
    axes[2].set_title("Residual Distribution")
    axes[2].set_xlabel("Residual")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved diagnostic plots to {out_path}")


def run_formal_tests(model, df: pd.DataFrame) -> None:
    resid = model.resid

    print("=== Formal residual diagnostics ===\n")

    exog = model.model.exog
    bp_stat, bp_pval, jb_stat, jb_pval, skew, kurtosis = bp_jb_diagnostics(resid, exog)

    print(f"Breusch-Pagan: stat={bp_stat:.2f}, p={bp_pval:.4g}")
    print("  -> ", "REJECT H0: heteroskedastic" if bp_pval < 0.05
          else "fail to reject H0: homoskedastic")

    print(f"\nJarque-Bera: stat={jb_stat:.2f}, p={jb_pval:.4g}, "
          f"skew={skew:.3f}, kurtosis={kurtosis:.3f}")
    print("  -> ", "REJECT H0: non-normal residuals" if jb_pval < 0.05
          else "fail to reject H0: normal residuals")

    # Durbin-Watson: ~2.0 = no autocorrelation, <2 = positive autocorrelation
    dw = durbin_watson(resid)
    print(f"\nDurbin-Watson: {dw:.3f} "
          "(2.0 = none, <1.5 = notable positive autocorrelation)")

    # Per-borough residual variance -- does the heteroskedasticity
    # actually track the borough variance structure from Levene's test?
    print("\n=== Residual std by borough (compare to Levene's test) ===")
    resid_df = df.assign(resid=resid.values)
    print(resid_df.groupby(PICKUP_BOROUGH_COL)["resid"].std().sort_values())


def fit_baseline_ols(model, df: pd.DataFrame):
    print(f"Sample size: {len(df)}\n")
    print(model.summary())
    print()

    run_formal_tests(model, df)
    plot_diagnostics(model)


if __name__ == "__main__":
    model, df = main()
    fit_baseline_ols(model, df)