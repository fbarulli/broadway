"""03.1: Structural and Analytical Verification of the Linear Model.

1. Structural (Raw OLS): Fits the model on raw, unstandardized features to 
   reveal the exact dollar-per-unit coefficients (e.g., $/mile, $/minute).
2. Analytical (Residual Diagnostics): Generates Residuals vs. Fitted and 
   Residual Distribution plots to check for linearity and homoscedasticity.
"""

import matplotlib

matplotlib.use("Agg")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from _common import INDEPENDENT_NUMERIC_FEATURES, RESULTS, TARGET, load_sample

CSV_OUT = RESULTS / "03_1_raw_ols_coefs.csv"
MD_OUT = RESULTS / "03_1_raw_ols_summary.md"
PNG_OUT = RESULTS / "03_1_residual_diagnostics.png"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    df = load_sample()
    
    X = df[INDEPENDENT_NUMERIC_FEATURES]
    y = df[TARGET]
    
    # Drop missing values
    valid_idx = X.notna().all(axis=1) & y.notna()
    X_clean, y_clean = X[valid_idx], y[valid_idx]
    
    # Add constant for intercept (base fare)
    X_raw = sm.add_constant(X_clean)
    
    print(f"Fitting raw OLS on {len(X_raw):,} observations...")
    model = sm.OLS(y_clean, X_raw).fit()
    
    # 1. Structural Verification (Raw Coefficients)
    coefs = pd.DataFrame({
        "feature": model.params.index,
        "raw_coef_dollars": model.params.values,
        "std_err": model.bse.values,
        "p_value": model.pvalues.values
    }).set_index("feature")
    
    coefs.to_csv(CSV_OUT)
    print("\n--- Raw OLS Coefficients (Dollars per unit) ---")
    print(coefs[["raw_coef_dollars", "p_value"]].to_string())
    print(f"wrote {CSV_OUT}")
    
    # Save Markdown Summary
    md_lines = [
        "# Structural Verification: Raw OLS Coefficients",
        "",
        f"Unstandardized OLS regression predicting `{TARGET}` (in dollars).",
        "",
        "| Feature | Coefficient ($/unit) | Std. Error | p-value |",
        "| --- | --- | --- | --- |"
    ]
    for feat, row in coefs.iterrows():
        md_lines.append(f"| {feat} | ${row['raw_coef_dollars']:.4f} | ${row['std_err']:.4f} | {row['p_value']:.2e} |")
    md_lines.extend([
        "",
        f"**R-squared:** {model.rsquared:.4f}",
        f"**F-statistic:** {model.fvalue:,.2f} (p={model.f_pvalue:.2e})"
    ])
    MD_OUT.write_text("\n".join(md_lines))
    print(f"wrote {MD_OUT}")

    # 2. Analytical Verification (Residual Diagnostics)
    print("\nGenerating residual diagnostics...")
    fitted = model.fittedvalues
    residuals = model.resid
    
    # Downsample for plotting to avoid the 1M-point "black blob" effect
    rng = np.random.default_rng(42)
    n_sample = min(50_000, len(fitted))
    idx = rng.choice(len(fitted), n_sample, replace=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    
    # Plot A: Residuals vs Fitted (Check for homoscedasticity / linearity)
    ax1.scatter(fitted.iloc[idx], residuals.iloc[idx], s=2, alpha=0.3, color="#4c72b0")
    ax1.axhline(0, color="#d62728", linestyle="--", lw=1.5)
    ax1.set_xlabel("Fitted Fare ($)")
    ax1.set_ylabel("Residuals ($)")
    ax1.set_title("Residuals vs. Fitted (Structural Check)")
    ax1.grid(True, alpha=0.3)
    
    # Plot B: Residual Distribution (Check for normality)
    sns.histplot(residuals, bins=100, kde=True, ax=ax2, color="#4c72b0", stat="density")
    ax2.axvline(0, color="#d62728", linestyle="--", lw=1.5)
    ax2.set_xlabel("Residuals ($)")
    ax2.set_title("Residual Distribution (Normality Check)")
    ax2.grid(True, alpha=0.3, axis="y")
    
    fig.savefig(PNG_OUT, dpi=150)
    plt.close(fig)
    print(f"wrote {PNG_OUT}")


if __name__ == "__main__":
    main()
