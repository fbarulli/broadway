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

import pandas as pd

from project import data
from broadway.stats import diagnostics, regression


def run_formal_tests(model: object, df: pd.DataFrame) -> None:
    resid = model.resid

    print("=== Formal residual diagnostics ===\n")

    result = regression.bp_jb(model)
    bp_stat = result["bp_stat"]
    bp_pval = result["bp_pval"]
    jb_stat = result["jb_stat"]
    jb_pval = result["jb_pval"]
    skew = result["skew"]
    kurtosis = result["kurtosis"]

    print(f"Breusch-Pagan: stat={bp_stat:.2f}, p={bp_pval:.4g}")
    print("  -> ", "REJECT H0: heteroskedastic" if bp_pval < 0.05
          else "fail to reject H0: homoskedastic")

    print(f"\nJarque-Bera: stat={jb_stat:.2f}, p={jb_pval:.4g}, "
          f"skew={skew:.3f}, kurtosis={kurtosis:.3f}")
    print("  -> ", "REJECT H0: non-normal residuals" if jb_pval < 0.05
          else "fail to reject H0: normal residuals")

    # Durbin-Watson: ~2.0 = no autocorrelation, <2 = positive autocorrelation
    dw = diagnostics.durbin_watson(resid)
    print(f"\nDurbin-Watson: {dw:.3f} "
          "(2.0 = none, <1.5 = notable positive autocorrelation)")

    # Per-borough residual variance -- does the heteroskedasticity
    # actually track the borough variance structure from Levene's test?
    print("\n=== Residual std by borough (compare to Levene's test) ===")
    resid_df = df.assign(resid=resid.values)
    print(resid_df.groupby(data.PICKUP_BOROUGH_COL)["resid"].std().sort_values())


def main() -> None:
    print("Loading cached stratified sample...")
    df = data.load_stratified_sample()
    print(f"Sample size: {len(df)}\n")

    formula = (
        f"{data.TARGET_COL} ~ {data.TRIP_DISTANCE_COL} + C({data.PICKUP_BOROUGH_COL})"
    )
    model = regression.fit_ols(df, formula)
    print(model.summary())
    print()

    run_formal_tests(model, df)

    out_path = str(data.RESULTS_DIR / "residual_diagnostics.png")
    diagnostics.plot_residuals(model, out_path)
    print(f"Saved diagnostic plots to {out_path}")


if __name__ == "__main__":
    main()
