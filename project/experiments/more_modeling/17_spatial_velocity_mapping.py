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

CSV_OUT = RESULTS / "17_velocity_impact.csv"
PNG_OUT = RESULTS / "17_spatial_velocity_mapping.png"
MD_OUT = RESULTS / "17_spatial_velocity_mapping.md"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_sample()

    # 1. Calculate Velocity & Linear Model Residuals
    valid = df[INDEPENDENT_NUMERIC_FEATURES + [TARGET]].dropna().copy()
    
    # Calculate MPH (Distance in miles / Time in hours)
    valid["speed_mph"] = valid["trip_distance"] / (valid["trip_duration_minutes"] / 60.0)
    valid["speed_mph"] = valid["speed_mph"].clip(upper=60) # Cap outliers
    
    # Get linear model residuals
    X = sm.add_constant(valid[INDEPENDENT_NUMERIC_FEATURES])
    model = sm.OLS(valid[TARGET], X).fit()
    valid["pred"] = model.predict(X)
    valid["residual"] = valid[TARGET] - valid["pred"]
    
    # Extract hour for gridlock mapping
    if "pickup_datetime" in df.columns:
        valid["pickup_hour"] = pd.to_datetime(df.loc[valid.index, "pickup_datetime"]).dt.hour
    else:
        valid["pickup_hour"] = np.nan

    # 2. Group by Time of Day (The Gridlock Curve)
    hourly_speed = valid.groupby("pickup_hour")["speed_mph"].mean().reset_index()
    
    # 3. Group by Velocity (The Residual Impact)
    speed_bins = [0, 5, 8, 12, 15, 20, 30, np.inf]
    speed_labels = ['0-5 (Gridlock)', '5-8 (Heavy)', '8-12 (Moderate)', '12-15 (Light)', '15-20 (Cruising)', '20-30 (Fast)', '30+ (Highway)']
    valid["speed_bin"] = pd.cut(valid["speed_mph"], bins=speed_bins, labels=speed_labels)
    
    speed_impact = valid.groupby("speed_bin", observed=False).agg(
        count=("residual", "size"),
        mean_speed=("speed_mph", "mean"),
        mean_residual=("residual", "mean"),
        pct_90_residual=("residual", lambda x: x.quantile(0.90))
    ).reset_index()
    
    print("--- Velocity Impact on Prediction Error ---")
    print(speed_impact.round(2).to_string(index=False))
    speed_impact.to_csv(CSV_OUT, index=False)
    
    # 4. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)
    
    # Top: The Gridlock Curve (Speed by Hour)
    ax1.plot(hourly_speed["pickup_hour"], hourly_speed["speed_mph"], marker='o', color="#d62728", linewidth=2)
    ax1.fill_between(hourly_speed["pickup_hour"], hourly_speed["speed_mph"], alpha=0.2, color="#d62728")
    ax1.set_xlabel("Hour of Day (0-23)")
    ax1.set_ylabel("Average Fleet Speed (MPH)")
    ax1.set_title("The Gridlock Curve: Fleet Velocity by Time of Day")
    ax1.set_xticks(range(24))
    ax1.grid(True, alpha=0.3)
    ax1.axhline(12, color="gray", linestyle="--", label="12 MPH (Cruising Threshold)")
    ax1.legend()
    
    # Bottom: Velocity vs Residuals (Do fast trips = higher errors?)
    ax2.bar(speed_impact["speed_bin"], speed_impact["mean_residual"], color="#4c72b0", alpha=0.8, label="Mean Residual (Underprediction)")
    ax2.plot(speed_impact["speed_bin"], speed_impact["pct_90_residual"], 'o-', color="#d62728", linewidth=2, label="90th Pct Error")
    ax2.set_xlabel("Average Trip Speed (MPH)")
    ax2.set_ylabel("Prediction Error (Actual - Predicted)")
    ax2.set_title("Velocity Impact: Does Highway Speed Explain the Unmodeled Variance?")
    ax2.axhline(0, color="black", linewidth=1)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    
    fig.savefig(PNG_OUT, dpi=150)
    plt.close(fig)
    
    md_lines = [
        "# 17: Spatial Velocity Mapping", "",
        "Velocity (MPH) acts as a hidden proxy for route type. Low velocity means city gridlock (time meter dominates). High velocity means highway driving (tolls and bridges dominate).", "",
        "## The Gridlock Curve", "",
        "![Velocity Mapping](17_spatial_velocity_mapping.png)", "",
        "## Velocity vs Prediction Error", "",
        speed_impact.round(2).to_markdown(index=False), "",
        "## Key Insight", "",
        "If the 90th percentile error spikes in the '30+ (Highway)' bin, it mathematically proves that our linear model's failures on long trips are driven by **highway tolls**, not just distance."
    ]
    MD_OUT.write_text("\n".join(md_lines))
    print(f"wrote {CSV_OUT}, {PNG_OUT}, {MD_OUT}")

if __name__ == "__main__":
    main()
