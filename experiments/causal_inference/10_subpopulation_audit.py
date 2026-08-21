import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from pathlib import Path

from _common import RESULTS, TARGET, load_sample, INDEPENDENT_NUMERIC_FEATURES

CSV_OUT = RESULTS / "10_subpopulation_audit.csv"
PNG_KDE = RESULTS / "10_subpopulation_kde.png"
PNG_SCATTER = RESULTS / "10_subpopulation_scatter.png"
MD_OUT = RESULTS / "10_subpopulation_audit.md"

# Residual bands based on known NYC taxi physics
SUBPOPULATIONS = {
    "JFK/LGA Flat Rate (Discounted)": (-25.0, -5.0),
    "No Toll / Base Meter": (-2.0, 2.0),
    "Minor Surcharge (Night/Peak)": (2.0, 4.0),
    "Minor Bridge/Tunnel Toll (~$6.55)": (5.0, 9.0),
    "Major Bridge/Tunnel Toll (~$12-$17)": (10.0, 19.0),
    "Unmodeled Outlier": (-np.inf, -25.0) # Catch-all for massive errors
}

# Define exact palette and order to match SUBPOPULATIONS keys
PALETTE = {
    "No Toll / Base Meter": "#cccccc",
    "Minor Surcharge (Night/Peak)": "#ff7f0e",
    "Minor Bridge/Tunnel Toll (~$6.55)": "#2ca02c",
    "Major Bridge/Tunnel Toll (~$12-$17)": "#9467bd",
    "JFK/LGA Flat Rate (Discounted)": "#d62728",
    "Unmodeled Outlier": "#8c564b"
}
ORDER = list(PALETTE.keys())

def assign_subpopulation(resid: float) -> str:
    for name, (low, high) in SUBPOPULATIONS.items():
        if low <= resid <= high:
            return name
    return "Other"

def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_sample()
    
    valid = df[INDEPENDENT_NUMERIC_FEATURES + [TARGET]].dropna()
    X = sm.add_constant(valid[INDEPENDENT_NUMERIC_FEATURES])
    y = valid[TARGET]
    
    model = sm.OLS(y, X).fit()
    preds = model.predict(X)
    residuals = y - preds
    
    eval_df = pd.DataFrame({"pred": preds, "actual": y, "residual": residuals})
    
    # Assign subpopulations
    eval_df["subpop"] = eval_df["residual"].apply(assign_subpopulation)
    
    stats = eval_df.groupby("subpop").agg(
        count=("residual", "size"),
        pct=("residual", lambda x: len(x) / len(eval_df) * 100),
        mean_residual=("residual", "mean"),
        median_actual=("actual", "median")
    ).sort_values("count", ascending=False)
    
    print("--- Subpopulation Audit ---")
    print(stats.round(2).to_string())
    stats.to_csv(CSV_OUT)
    print(f"wrote {CSV_OUT}")
    
    # Plot 1: KDE of Residuals (Finding the peaks)
    fig1, ax1 = plt.subplots(figsize=(10, 5), constrained_layout=True)
    sns.kdeplot(eval_df["residual"], bw_adjust=0.5, color="#4c72b0", fill=True, ax=ax1)
    ax1.axvline(0, color="black", ls="--", lw=1.5, label="Perfect Meter Match")
    ax1.axvline(6.55, color="#d62728", ls=":", lw=2, label="~$6.55 Toll Peak")
    ax1.axvline(-15, color="#2ca02c", ls=":", lw=2, label="JFK Flat Rate Discount")
    ax1.set_xlim(-30, 30)
    ax1.set_xlabel("Residual (Actual - Predicted Meter)")
    ax1.set_ylabel("Density")
    ax1.set_title("Residual Distribution (KDE) - Finding the Hidden Peaks")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    fig1.savefig(PNG_KDE, dpi=150)
    plt.close(fig1)
    
    # Plot 2: Actual vs Predicted colored by subpopulation
    fig2, ax2 = plt.subplots(figsize=(10, 8), constrained_layout=True)
    
    # Downsample for readability
    rng = np.random.default_rng(42)
    idx = rng.choice(len(eval_df), min(20000, len(eval_df)), replace=False)
    sample = eval_df.iloc[idx]
    
    # Draw y=x line
    lim = [0, max(sample["pred"].max(), sample["actual"].max())]
    ax2.plot(lim, lim, "--", color="black", lw=1.5, label="Perfect Prediction")
    
    sns.scatterplot(
        data=sample, x="pred", y="actual", hue="subpop", 
        palette=PALETTE, s=15, alpha=0.4, ax=ax2,
        hue_order=ORDER
    )
    
    ax2.set_xlabel("Predicted Meter ($)")
    ax2.set_ylabel("Actual Fare ($)")
    ax2.set_title("Actual vs Predicted by Subpopulation")
    ax2.set_xlim(lim); ax2.set_ylim(lim)
    ax2.legend(title="Subpopulation", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, alpha=0.3)
    fig2.savefig(PNG_SCATTER, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    
    print(f"wrote {PNG_KDE}, {PNG_SCATTER}")
    
    md_lines = [
        "# 10: Subpopulation Audit (The Hidden Physics)",
        "",
        "A 2-variable linear model attempts to fit a single straight line through multiple distinct physical realities. This audit isolates the 'hidden' subpopulations based on their residual signatures.",
        "",
        "## Subpopulation Breakdown",
        "",
        stats.to_markdown(),
        "",
        "## The Peaks (KDE)",
        "",
        "The Kernel Density Estimate (KDE) reveals distinct bumps in the error distribution. A peak near $0 represents perfect metered trips. A peak near $6.55 represents trips crossing standard toll bridges. A negative peak represents flat-rate airport discounts.",
    ]
    MD_OUT.write_text("\n".join(md_lines))
    print(f"wrote {MD_OUT}")

if __name__ == "__main__":
    main()
