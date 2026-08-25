import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

from _common import RESULTS, TARGET, load_sample

CSV_OUT = RESULTS / "20_surge_matrix.csv"
PNG_OUT = RESULTS / "20_surge_matrix.png"
MD_OUT = RESULTS / "20_surge_matrix.md"

GRIDLOCK_HOURS = [7, 8, 9, 16, 17, 18, 19]
MULTIPLIERS = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
ELASTICITIES = [-0.2, -0.4, -0.6, -0.8, -1.0, -1.2]


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_sample()

    df = df.copy()
    df["pickup_hour"] = pd.to_datetime(df["pickup_datetime"]).dt.hour
    rev_col = "total_amount" if "total_amount" in df.columns else TARGET
    
    # Split baseline revenue
    grid_mask = df["pickup_hour"].isin(GRIDLOCK_HOURS)
    baseline_grid_rev = df.loc[grid_mask, rev_col].sum()
    baseline_non_grid_rev = df.loc[~grid_mask, rev_col].sum()
    baseline_total_rev = baseline_grid_rev + baseline_non_grid_rev
    
    print(f"Baseline Gridlock Revenue Share: {baseline_grid_rev / baseline_total_rev * 100:.1f}%")
    
    # Build the decision matrix
    rows = []
    for e in ELASTICITIES:
        row = []
        for m in MULTIPLIERS:
            # Volume factor (floored at 0)
            vol = max(0.0, 1.0 + (m - 1.0) * e)
            
            # Calculate new revenue
            sim_grid_rev = baseline_grid_rev * m * vol
            sim_total_rev = baseline_non_grid_rev + sim_grid_rev
            
            # Percentage change
            pct_change = (sim_total_rev / baseline_total_rev) - 1.0
            row.append(pct_change)
        rows.append(row)
        
    matrix = pd.DataFrame(rows, index=ELASTICITIES, columns=MULTIPLIERS)
    matrix.index.name = "Elasticity"
    
    matrix.to_csv(CSV_OUT)
    print(f"wrote {CSV_OUT}")

    # --- Plotting the Heatmap ---
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    # RdYlGn: Red = lose money, Green = make money
    sns.heatmap(
        matrix, 
        annot=True, 
        fmt=".1%",          # Format as percentage
        cmap="RdYlGn",      # Red-Yellow-Green color palette
        center=0,           # Center the colors exactly at 0%
        linewidths=1,       # White grid lines
        linecolor="white",
        cbar_kws={"label": "Net Fleet Revenue Change (%)"},
        ax=ax
    )
    
    ax.set_title("Surge Pricing Decision Matrix", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Surge Multiplier (Applied to Gridlock Hours)", fontsize=12)
    ax.set_ylabel("Demand Elasticity (Rider Price Sensitivity)", fontsize=12)
    
    # Rotate y-axis labels so they read horizontally
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels([f"{m}x" for m in MULTIPLIERS])

    fig.savefig(PNG_OUT, dpi=150)
    plt.close(fig)
    print(f"wrote {PNG_OUT}")

    md_lines = [
        "# 20: Surge Pricing Decision Matrix",
        "",
        f"Gridlock hours ({GRIDLOCK_HOURS}) currently represent **{baseline_grid_rev / baseline_total_rev * 100:.1f}%** of total fleet revenue.",
        "",
        "This matrix answers the core business question: **If riders are X% sensitive to price, and we surge by Y%, do we make or lose money?**",
        "",
        f"![Decision Matrix]({PNG_OUT.name})",
        "",
        "## How to read this matrix",
        "",
        "- **Green cells:** The surge multiplier generated a net positive return for the fleet.",
        "- **Red cells:** The surge multiplier destroyed value (volume loss outweighed price gain).",
        "- **Yellow cells:** Revenue-neutral.",
        "",
        "**Note on Elasticity:**",
        "- An elasticity of `-0.2` means riders are highly inelastic (they will pay surge prices because they desperately need a ride, e.g., during a rainstorm or late-night commute).",
        "- An elasticity of `-1.0` means riders are perfectly elastic (a 20% price hike causes exactly a 20% drop in rides, resulting in $0 net revenue gain)."
    ]
    MD_OUT.write_text("\n".join(md_lines))
    print(f"wrote {MD_OUT}")


if __name__ == "__main__":
    main()
