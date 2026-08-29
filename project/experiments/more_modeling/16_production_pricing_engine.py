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

CSV_OUT = RESULTS / "16_production_engine_comparison.csv"
PNG_OUT = RESULTS / "16_production_engine_comparison.png"
MD_OUT = RESULTS / "16_production_pricing_engine.md"

# --- Production Pricing Engine Configuration ---
CORE_INTERCEPT = 4.13
CORE_DISTANCE_RATE = 3.03
CORE_TIME_RATE = 0.29
MINIMUM_FARE = 5.00

# Flat-rate overrides
AIRPORT_ZONES = {132, 138, 237}
MANHATTAN_ZONES = set(range(1, 101))
FLAT_RATE_JFK = 70.0
FLAT_RATE_LGA = 52.0
FLAT_RATE_EWR = 70.0

# Option C: Scaled risk premium for long-haul trips
LONG_HAUL_THRESHOLD = 15.0  # Miles
RISK_PREMIUM_PER_MILE = 1.00  # $1 per mile over the threshold


def is_manhattan(zone_id: int) -> bool:
    return zone_id in MANHATTAN_ZONES

def is_airport(zone_id: int) -> bool:
    return zone_id in AIRPORT_ZONES

def get_flat_rate(airport_zone: int) -> float:
    if airport_zone == 132: return FLAT_RATE_JFK
    if airport_zone == 138: return FLAT_RATE_LGA
    if airport_zone == 237: return FLAT_RATE_EWR
    return 0.0


def production_pricing_engine(distance, duration, pickup_zone, dropoff_zone):
    # 1. Flat-Rate Override (Airport <-> Manhattan ONLY)
    if (is_airport(pickup_zone) and is_manhattan(dropoff_zone)) or \
       (is_airport(dropoff_zone) and is_manhattan(pickup_zone)):
        rate = get_flat_rate(pickup_zone) if is_airport(pickup_zone) else get_flat_rate(dropoff_zone)
        return rate, "flat_rate"

    # 2. Core Engine
    fare = CORE_INTERCEPT + (CORE_DISTANCE_RATE * distance) + (CORE_TIME_RATE * duration)

    # 3. Option C: Scaled risk premium for long-haul trips
    if distance > LONG_HAUL_THRESHOLD:
        risk_premium = (distance - LONG_HAUL_THRESHOLD) * RISK_PREMIUM_PER_MILE
        fare += risk_premium
        return max(fare, MINIMUM_FARE), "risk_premium"

    # 4. Low-End Guardrail
    if fare < MINIMUM_FARE:
        return MINIMUM_FARE, "minimum_fare"

    return fare, "linear"


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_sample()

    valid = df[INDEPENDENT_NUMERIC_FEATURES + [TARGET, "pickup_location_id", "dropoff_location_id"]].dropna()
    X = sm.add_constant(valid[INDEPENDENT_NUMERIC_FEATURES])
    y = valid[TARGET]

    model = sm.OLS(y, X).fit()
    linear_preds = model.predict(X)

    engine_results = valid.apply(
        lambda row: production_pricing_engine(
            row["trip_distance"], row["trip_duration_minutes"],
            int(row["pickup_location_id"]), int(row["dropoff_location_id"])
        ), axis=1, result_type="expand"
    )
    engine_preds = engine_results[0]
    engine_routes = engine_results[1]

    eval_df = pd.DataFrame({
        "distance": valid["trip_distance"], "actual": y,
        "linear_pred": linear_preds, "engine_pred": engine_preds, "engine_route": engine_routes,
    })

    eval_df["linear_abs_error"] = np.abs(eval_df["actual"] - eval_df["linear_pred"])
    eval_df["engine_abs_error"] = np.abs(eval_df["actual"] - eval_df["engine_pred"])

    print("--- Production Engine Performance (Scaled Risk Premium) ---")
    print(f"Linear Model MAE:        ${eval_df['linear_abs_error'].mean():.2f}")
    print(f"Engine MAE:              ${eval_df['engine_abs_error'].mean():.2f}")
    print(f"Linear 90th Pct Error:   ${eval_df['linear_abs_error'].quantile(0.90):.2f}")
    print(f"Engine 90th Pct Error:   ${eval_df['engine_abs_error'].quantile(0.90):.2f}")

    route_stats = eval_df.groupby("engine_route").agg(
        count=("actual", "size"), linear_mae=("linear_abs_error", "mean"),
        engine_mae=("engine_abs_error", "mean"), mean_actual=("actual", "mean")
    ).reset_index()
    print("\n--- Performance by Engine Route ---")
    print(route_stats.round(2).to_string(index=False))

    linear_trips = eval_df[eval_df["engine_route"].isin(["linear", "risk_premium"])].copy()
    dist_bins = [0, 3, 5, 10, 15, 20, np.inf]
    dist_labels = ['0-3mi', '3-5mi', '5-10mi', '10-15mi', '15-20mi', '20mi+']
    linear_trips["dist_bin"] = pd.cut(linear_trips["distance"], bins=dist_bins, labels=dist_labels)

    comparison = linear_trips.groupby("dist_bin", observed=False).agg(
        count=("actual", "size"), linear_mae=("linear_abs_error", "mean"),
        engine_mae=("engine_abs_error", "mean")
    ).reset_index()
    
    print("\n--- MAE by Distance Bin (Linear vs Scaled Risk Premium) ---")
    print(comparison.round(2).to_string(index=False))
    comparison.to_csv(CSV_OUT, index=False)

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)

    # Top: Scatter (downsampled)
    sample = eval_df.sample(n=min(30000, len(eval_df)), random_state=42)
    ax1.scatter(sample["distance"], sample["actual"], s=3, alpha=0.15, color="#4c72b0", label="Actual Fare")
    ax1.scatter(sample["distance"], sample["engine_pred"], s=3, alpha=0.3, color="#d62728", label="Engine Prediction")
    ax1.axhline(MINIMUM_FARE, color="green", linestyle="--", linewidth=1.5, label=f"Min Fare (${MINIMUM_FARE})")
    ax1.axhline(70, color="orange", linestyle="--", linewidth=1.5, label="JFK Flat Rate ($70)")
    ax1.axhline(52, color="purple", linestyle="--", linewidth=1.5, label="LGA Flat Rate ($52)")
    
    ax1.set_xlabel("Trip Distance (miles)")
    ax1.set_ylabel("Fare ($)")
    ax1.set_title("Production Pricing Engine: Actual vs Predicted (Scaled Risk Premium)")
    ax1.legend(markerscale=3, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Bottom: Grouped Bar Chart
    x = np.arange(len(comparison))
    width = 0.35
    ax2.bar(x - width/2, comparison["linear_mae"], width, label="Linear Model (Baseline)", color="#4c72b0", alpha=0.8)
    ax2.bar(x + width/2, comparison["engine_mae"], width, label="Engine (Scaled Risk Premium)", color="#d62728", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(comparison["dist_bin"])
    ax2.set_xlabel("Trip Distance (miles)")
    ax2.set_ylabel("Mean Absolute Error ($)")
    ax2.set_title("MAE Comparison: Grouped Bars")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    fig.savefig(PNG_OUT, dpi=150)
    plt.close(fig)

    md_lines = [
        "# 16: Production Pricing Engine (Scaled Risk Premium)", "",
        "The engine now applies a **scaled** risk premium for long-haul trips instead of a flat surcharge.", "",
        "## Routing Logic", "",
        "| Route | Trigger | Action |", "| --- | --- | --- |",
        "| `flat_rate` | Airport ↔ Manhattan ONLY | Fixed price ($70 JFK, $52 LGA) |",
        f"| `risk_premium` | Distance > {LONG_HAUL_THRESHOLD} miles | Add ${RISK_PREMIUM_PER_MILE:.2f} per mile over threshold |",
        "| `minimum_fare` | Predicted fare < $5.00 | Snap to $5.00 |",
        "| `linear` | Everything else | `$4.13 + $3.03×dist + $0.29×min` |", "",
        "## Performance by Route", "", route_stats.round(2).to_markdown(index=False), "",
        "## MAE by Distance Bin", "", comparison.round(2).to_markdown(index=False), "",
        "## Key Insight", "",
        f"The scaled premium (+${RISK_PREMIUM_PER_MILE:.2f}/mile over {LONG_HAUL_THRESHOLD}mi) provides a gentler buffer for long trips.",
        "A 17-mile trip adds $2; a 25-mile trip adds $10. This reduces over-correction compared to the flat $15 surcharge."
    ]
    MD_OUT.write_text("\n".join(md_lines))
    print(f"wrote {CSV_OUT}, {PNG_OUT}, {MD_OUT}")

if __name__ == "__main__":
    main()
