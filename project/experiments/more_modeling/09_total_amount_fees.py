import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib

matplotlib.use("Agg")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from _common import INDEPENDENT_NUMERIC_FEATURES, RESULTS, TARGET, load_sample

TOTAL_TARGET = "total_amount"
CSV_OUT = RESULTS / "09_total_amount_fees.csv"
PNG_COEFS = RESULTS / "09_meter_vs_total_coefs.png"
PNG_FUNNEL = RESULTS / "09_total_amount_funnel_risk.png"
MD_OUT = RESULTS / "09_total_amount_fees.md"

def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_sample()
    
    # Ensure total_amount exists; fallback to sum of parts if the pipeline dropped it
    if TOTAL_TARGET not in df.columns:
        print("total_amount missing, attempting to compute from components...")
        fee_cols = [c for c in ["fare_amount", "tip_amount", "tolls_amount", 
                                "mta_tax", "improvement_surcharge", "congestion_surcharge", "airport_fee"] 
                    if c in df.columns]
        df[TOTAL_TARGET] = df[fee_cols].sum(axis=1)

    valid = df[INDEPENDENT_NUMERIC_FEATURES + [TARGET, TOTAL_TARGET]].dropna()
    
    X = sm.add_constant(valid[INDEPENDENT_NUMERIC_FEATURES])
    y_fare = valid[TARGET]
    y_total = valid[TOTAL_TARGET]
    
    # Fit both models to show the exact dollar value of the "hidden fees"
    model_fare = sm.OLS(y_fare, X).fit()
    model_total = sm.OLS(y_total, X).fit()
    
    compare_df = pd.DataFrame({
        "Raw Meter ($/unit)": model_fare.params,
        "Total Amount ($/unit)": model_total.params,
    })
    compare_df["Hidden Fee Premium"] = compare_df["Total Amount ($/unit)"] - compare_df["Raw Meter ($/unit)"]
    
    print("--- Raw Meter vs Total Amount Coefficients ---")
    print(compare_df.round(4).to_string())
    print(f"\nFare R-squared: {model_fare.rsquared:.4f}")
    print(f"Total R-squared: {model_total.rsquared:.4f}")
    
    compare_df.to_csv(CSV_OUT)
    print(f"wrote {CSV_OUT}")
    
    # --- Visualization 1: Coefficient Comparison ---
    fig1, ax1 = plt.subplots(figsize=(8, 5), constrained_layout=True)
    x = np.arange(len(compare_df))
    width = 0.35
    plot_df = compare_df.drop("Hidden Fee Premium", axis=1)
    
    ax1.bar(x - width/2, plot_df["Raw Meter ($/unit)"], width, label="Raw Meter", color="#4c72b0")
    ax1.bar(x + width/2, plot_df["Total Amount ($/unit)"], width, label="Total Amount", color="#d62728")
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_df.index)
    ax1.set_ylabel("Coefficient ($)")
    ax1.set_title("Raw Meter vs Total Amount (Where do the hidden fees live?)")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    fig1.savefig(PNG_COEFS, dpi=150)
    plt.close(fig1)
    
    # --- Visualization 2: The Funnel of Risk ---
    preds_total = model_total.predict(X)
    residuals_total = y_total - preds_total
    
    # Downsample for scatter plot (avoid black blob)
    rng = np.random.default_rng(42)
    n_samp = min(50_000, len(preds_total))
    idx = rng.choice(len(preds_total), n_samp, replace=False)
    
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    
    # Plot A: Actual vs Predicted (The y=x line)
    ax2.scatter(preds_total.iloc[idx], y_total.iloc[idx], s=3, alpha=0.15, color="#4c72b0")
    lim = [0, max(preds_total.max(), y_total.max())]
    ax2.plot(lim, lim, "--", color="#d62728", lw=2, label="Perfect Prediction (y=x)")
    ax2.set_xlabel("Predicted Total Amount ($)")
    ax2.set_ylabel("Actual Total Amount ($)")
    ax2.set_title("Predicted vs Actual Total Amount")
    ax2.set_xlim(lim); ax2.set_ylim(lim)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot B: The Funnel (Residuals vs Fitted)
    ax3.scatter(preds_total.iloc[idx], residuals_total.iloc[idx], s=3, alpha=0.15, color="#4c72b0")
    ax3.axhline(0, color="#d62728", ls="--", lw=2)
    ax3.set_xlabel("Predicted Total Amount ($)")
    ax3.set_ylabel("Residual / Error ($)")
    ax3.set_title("The Funnel of Risk: Residuals vs Predicted")
    ax3.grid(True, alpha=0.3)
    
    fig2.savefig(PNG_FUNNEL, dpi=150)
    plt.close(fig2)
    
    print(f"wrote {PNG_COEFS}, {PNG_FUNNEL}")
    
    # --- Markdown Summary ---
    md_lines = [
        "# 09: The Unmodeled Hidden Fees (Total Amount vs Raw Meter)",
        "",
        "By shifting the target from the raw meter (`fare_amount`) to the final credit card charge (`total_amount`), the model naturally absorbs the 7% unexplained variance: tips, bridge tolls, and state taxes.",
        "",
        "## Coefficient Shift",
        "",
        "| Feature | Raw Meter ($/unit) | Total Amount ($/unit) | Hidden Fee Premium |",
        "| --- | --- | --- | --- |",
    ]
    for feat, row in compare_df.iterrows():
        md_lines.append(f"| {feat} | ${row['Raw Meter ($/unit)']:.4f} | ${row['Total Amount ($/unit)']:.4f} | ${row['Hidden Fee Premium']:.4f} |")
        
    md_lines.extend([
        "",
        "## The Funnel of Risk",
        "",
        "Even modeling the total amount, the residual plot reveals a heteroscedastic 'funnel'. As trips get longer and more expensive, the absolute variance in fares widens. This is driven by highly variable discretionary behavior (e.g., a passenger tipping 30% vs 10% on a $100 ride, or taking an $18 toll bridge vs a free route).",
    ])
    MD_OUT.write_text("\n".join(md_lines))
    print(f"wrote {MD_OUT}")

if __name__ == "__main__":
    main()
