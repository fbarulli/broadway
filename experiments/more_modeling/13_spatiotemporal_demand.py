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

from _common import RESULTS, load_sample

CSV_OUT = RESULTS / "13_demand_matrix.csv"
PNG_HEATMAP = RESULTS / "13_city_demand_heatmap.png"
PNG_WEEK = RESULTS / "13_weekly_demand_curve.png"
MD_OUT = RESULTS / "13_spatiotemporal_demand.md"


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract core temporal drivers from pickup_datetime."""
    if "pickup_datetime" not in df.columns:
        raise ValueError("Dataset missing pickup_datetime for temporal extraction.")
        
    dt = pd.to_datetime(df["pickup_datetime"])
    df = df.copy()
    df["pickup_hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek  # 0=Mon, 6=Sun
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_sample()
    
    print("Extracting temporal features...")
    df = extract_temporal_features(df)
    
    # --- 1. City-Wide Temporal Demand (The Weekly Curve) ---
    hourly_demand = df.groupby(["dayofweek", "pickup_hour"]).size().reset_index(name="ride_count")
    hourly_demand["day_name"] = hourly_demand["dayofweek"].map({
        0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"
    })
    
    fig1, ax1 = plt.subplots(figsize=(14, 6), constrained_layout=True)
    sns.lineplot(
        data=hourly_demand, x="pickup_hour", y="ride_count", 
        hue="day_name", hue_order=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        ax=ax1, marker="o", markersize=4
    )
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Total Ride Volume (City-Wide)")
    ax1.set_title("Spatio-Temporal Demand: The Weekly City-Wide Curve")
    ax1.set_xticks(range(0, 24))
    ax1.grid(True, alpha=0.3)
    ax1.legend(title="Day of Week")
    fig1.savefig(PNG_WEEK, dpi=150)
    plt.close(fig1)
    
    # --- 2. Spatio-Temporal Heatmap (Zone x Hour) ---
    # Find the top 15 zones by total volume to avoid a 247-row unreadable heatmap
    top_zones = df["pickup_location_id"].value_counts().head(15).index.tolist()
    
    zone_hour_demand = df[df["pickup_location_id"].isin(top_zones)].groupby(
        ["pickup_location_id", "pickup_hour"]
    ).size().unstack(fill_value=0)
    
    fig2, ax2 = plt.subplots(figsize=(14, 8), constrained_layout=True)
    sns.heatmap(zone_hour_demand, cmap="YlOrRd", ax=ax2, cbar_kws={"label": "Ride Count"})
    ax2.set_xlabel("Pickup Hour (0-23)")
    ax2.set_ylabel("Pickup Zone ID (Top 15)")
    ax2.set_title("Spatio-Temporal Demand Heatmap: Top 15 Zones by Hour")
    fig2.savefig(PNG_HEATMAP, dpi=150)
    plt.close(fig2)
    
    # --- 3. Save Matrix ---
    full_matrix = df.groupby(["pickup_location_id", "dayofweek", "pickup_hour"]).size().reset_index(name="ride_count")
    full_matrix.to_csv(CSV_OUT, index=False)
    print(f"wrote {CSV_OUT}, {PNG_WEEK}, {PNG_HEATMAP}")
    
    md_lines = [
        "# 13: Spatio-Temporal Demand Forecasting",
        "",
        "While the physical meter dictates the *price*, time and space dictate the *volume*. This script aggregates the 1M trips into a Spatio-Temporal matrix to map city-wide demand.",
        "",
        "## The Weekly Curve",
        "",
        "![Weekly Demand Curve](13_weekly_demand_curve.png)",
        "",
        "## The Spatio-Temporal Heatmap",
        "",
        "![Zone x Hour Heatmap](13_city_demand_heatmap.png)",
        "",
        "## Business Implications",
        "",
        "By mapping the exact ride volume per Zone × Hour × Day, we can identify massive supply/demand imbalances. These imbalances dictate exactly when and where **surge pricing** should be applied, and where the **deadheading (empty driving) costs** are highest.",
    ]
    MD_OUT.write_text("\n".join(md_lines))
    print(f"wrote {MD_OUT}")


if __name__ == "__main__":
    main()
