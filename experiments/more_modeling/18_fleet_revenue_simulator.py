import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

from _common import RESULTS, TARGET, load_sample

CSV_OUT = RESULTS / "18_fleet_revenue_simulator.csv"
PNG_OUT = RESULTS / "18_fleet_revenue_simulator.png"
MD_OUT = RESULTS / "18_fleet_revenue_simulator.md"

# --- Simulation Parameters ---
SURGE_MULTIPLIER = 1.20  # 20% price increase
PRICE_ELASTICITY = -0.3  # 1% price increase = 0.3% volume drop

# Gridlock hours identified from Script 17
GRIDLOCK_HOURS = [7, 8, 9, 16, 17, 18, 19]

def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_sample()

    # Extract temporal features
    df = df.copy()
    df["pickup_hour"] = pd.to_datetime(df["pickup_datetime"]).dt.hour
    
    # We use the actual total_amount as the baseline market revenue
    rev_col = "total_amount" if "total_amount" in df.columns else TARGET
    df["revenue"] = df[rev_col]

    # 1. Baseline Metrics
    hourly_metrics = df.groupby("pickup_hour").agg(
        rides=("revenue", "size"),
        baseline_revenue=("revenue", "sum"),
        avg_fare=("revenue", "mean")
    ).reset_index()
    
    # 2. Apply Surge & Elasticity
    hourly_metrics["is_gridlock"] = hourly_metrics["pickup_hour"].isin(GRIDLOCK_HOURS)
    
    # New Price
    hourly_metrics["surge_price"] = np.where(
        hourly_metrics["is_gridlock"],
        hourly_metrics["avg_fare"] * SURGE_MULTIPLIER,
        hourly_metrics["avg_fare"]
    )
    
    # New Volume (Elasticity applied only during surge)
    # A 20% increase (0.20) * -0.3 elasticity = -0.06 (-6% volume drop)
    volume_change = np.where(
        hourly_metrics["is_gridlock"],
        (SURGE_MULTIPLIER - 1.0) * PRICE_ELASTICITY,
        0.0
    )
    hourly_metrics["new_rides"] = hourly_metrics["rides"] * (1 + volume_change)
    
    # New Revenue
    hourly_metrics["simulated_revenue"] = hourly_metrics["new_rides"] * hourly_metrics["surge_price"]
    
    # Deltas
    hourly_metrics["revenue_delta"] = hourly_metrics["simulated_revenue"] - hourly_metrics["baseline_revenue"]
    hourly_metrics["ride_delta"] = hourly_metrics["new_rides"] - hourly_metrics["rides"]
    
    # Print summary
    print("--- Fleet Revenue Simulation ---")
    print(f"Surge Multiplier: {SURGE_MULTIPLIER}x")
    print(f"Price Elasticity: {PRICE_ELASTICITY}")
    print(f"Gridlock Hours: {GRIDLOCK_HOURS}")
    
    total_baseline_rev = hourly_metrics["baseline_revenue"].sum()
    total_sim_rev = hourly_metrics["simulated_revenue"].sum()
    total_baseline_rides = hourly_metrics["rides"].sum()
    total_sim_rides = hourly_metrics["new_rides"].sum()
    
    print("\n--- Sample Totals ---")
    print(f"Baseline Revenue:   ${total_baseline_rev:,.0f}")
    print(f"Simulated Revenue:  ${total_sim_rev:,.0f}")
    print(f"Net Revenue Delta:  ${total_sim_rev - total_baseline_rev:,.0f} ({(total_sim_rev/total_baseline_rev - 1)*100:.2f}%)")
    print(f"Net Ride Delta:     {total_sim_rides - total_baseline_rides:,.0f} rides ({(total_sim_rides/total_baseline_rides - 1)*100:.2f}%)")
    
    hourly_metrics.to_csv(CSV_OUT, index=False)
    print(f"wrote {CSV_OUT}")
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    
    # Top: Revenue by Hour
    x = hourly_metrics["pickup_hour"]
    ax1.bar(x, hourly_metrics["baseline_revenue"], label="Baseline Revenue", color="#4c72b0", alpha=0.7)
    ax1.bar(x, hourly_metrics["simulated_revenue"], label="Simulated Revenue (Gridlock Surge)", color="#d62728", alpha=0.4)
    
    for h in GRIDLOCK_HOURS:
        ax1.axvspan(h - 0.4, h + 0.4, color='red', alpha=0.1)
        
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Revenue ($)")
    ax1.set_title(f"Fleet Revenue Simulation: {SURGE_MULTIPLIER}x Surge During Gridlock Hours")
    ax1.set_xticks(range(24))
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    
    # Bottom: Ride Volume Delta
    ax2.bar(x, hourly_metrics["ride_delta"], color="#2ca02c")
    ax2.axhline(0, color="black", linewidth=1)
    for h in GRIDLOCK_HOURS:
        ax2.axvspan(h - 0.4, h + 0.4, color='red', alpha=0.1)
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Change in Ride Volume")
    ax2.set_title(f"Ride Volume Impact (Elasticity = {PRICE_ELASTICITY})")
    ax2.set_xticks(range(24))
    ax2.grid(True, alpha=0.3, axis="y")
    
    fig.savefig(PNG_OUT, dpi=150)
    plt.close(fig)
    print(f"wrote {PNG_OUT}")
    
    md_lines = [
        "# 18: Fleet Revenue Simulator (Surge Pricing Scenario)", "",
        f"**Scenario:** Apply a **{SURGE_MULTIPLIER}x surge multiplier** during gridlock hours ({GRIDLOCK_HOURS}).",
        f"**Assumption:** Price elasticity of demand is **{PRICE_ELASTICITY}** (a 10% price increase yields a 3% drop in ride volume).", "",
        "## Sample Totals", "",
        f"- **Baseline Revenue:** ${total_baseline_rev:,.0f}",
        f"- **Simulated Revenue:** ${total_sim_rev:,.0f}",
        f"- **Net Revenue Delta:** ${total_sim_rev - total_baseline_rev:,.0f} ({(total_sim_rev/total_baseline_rev - 1)*100:.2f}%)",
        f"- **Net Ride Delta:** {total_sim_rides - total_baseline_rides:,.0f} rides ({(total_sim_rides/total_baseline_rides - 1)*100:.2f}%)", "",
        "## Hourly Breakdown", "",
        hourly_metrics[["pickup_hour", "rides", "new_rides", "baseline_revenue", "simulated_revenue", "revenue_delta"]].to_markdown(index=False), "",
        "![Simulation](18_fleet_revenue_simulator.png)"
    ]
    MD_OUT.write_text("\n".join(md_lines))
    print(f"wrote {MD_OUT}")

if __name__ == "__main__":
    main()
