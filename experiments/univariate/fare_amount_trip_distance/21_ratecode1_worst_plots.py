"""21: spotlight plots for the worst log-fare predictions (step 20).

Two figures, following the step-20 analysis:
1. fare vs distance — all trips in gray, the worst-prediction trips in red,
   zoomed to the main cluster (xlim 0-20, ylim 0-60).
2. log-residuals vs predicted log-fare — all residuals in gray, worst in red.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import RESULTS, load_metered
from _ols_bp import add_log_predictions

THRESHOLD = -2.0

OUT_FARE_DISTANCE = RESULTS / f"{Path(__file__).stem}_fare_distance.png"
OUT_RESIDUALS = RESULTS / f"{Path(__file__).stem}_residuals.png"


def plot_fare_vs_distance(df, worst, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["trip_distance"], df["fare_amount"], s=5, alpha=0.1, color="gray", label="Normal Trips")
    ax.scatter(worst["trip_distance"], worst["fare_amount"], s=40, color="red", edgecolor="black", label="Worst Predictions")
    ax.set_title(f"Spotlight on Worst Predictions (Fare vs Distance, N={len(df)})")
    ax.set_xlabel("Trip Distance (miles)")
    ax.set_ylabel("Fare Amount ($)")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 60)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_residuals(df, worst, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["predicted_log_fare"], df["log_residuals"], s=5, alpha=0.1, color="gray")
    ax.scatter(worst["predicted_log_fare"], worst["log_residuals"], s=40, color="red", edgecolor="black")
    ax.axhline(0, color="black", linestyle="--")
    ax.set_title(f"Residuals vs Fitted (Outliers Highlighted, N={len(df)})")
    ax.set_xlabel("Predicted Log Fare")
    ax.set_ylabel("Log Residual")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    df = add_log_predictions(load_metered())
    worst = df[df["log_residuals"] < THRESHOLD]
    print(f"trips with log_residuals < {THRESHOLD}: {len(worst)} of {len(df)}")

    plot_fare_vs_distance(df, worst, OUT_FARE_DISTANCE)
    print(f"wrote {OUT_FARE_DISTANCE}")
    plot_residuals(df, worst, OUT_RESIDUALS)
    print(f"wrote {OUT_RESIDUALS}")


if __name__ == "__main__":
    main()
