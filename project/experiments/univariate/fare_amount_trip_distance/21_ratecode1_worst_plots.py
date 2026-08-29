"""21: spotlight plots for the worst log-fare predictions (step 20).

One figure, two panels: fare vs distance (all trips gray, worst in red,
zoomed to the main cluster) and log-residuals vs predicted log-fare
(worst highlighted), following the step-20 analysis.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import RESULTS, load_metered
from _ols_bp import add_log_predictions

THRESHOLD = -2.0

OUT = RESULTS / f"{Path(__file__).stem}.png"


def draw_fare_vs_distance(ax, df, worst) -> None:
    ax.scatter(df["trip_distance"], df["fare_amount"], s=5, alpha=0.1, color="gray", label="Normal Trips")
    ax.scatter(worst["trip_distance"], worst["fare_amount"], s=40, color="red", edgecolor="black", label="Worst Predictions")
    ax.set_title(f"Spotlight on Worst Predictions (Fare vs Distance, N={len(df)})")
    ax.set_xlabel("Trip Distance (miles)")
    ax.set_ylabel("Fare Amount ($)")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 60)
    ax.legend()


def draw_residuals(ax, df, worst) -> None:
    ax.scatter(df["predicted_log_fare"], df["log_residuals"], s=5, alpha=0.1, color="gray")
    ax.scatter(worst["predicted_log_fare"], worst["log_residuals"], s=40, color="red", edgecolor="black")
    ax.axhline(0, color="black", linestyle="--")
    ax.set_title(f"Residuals vs Fitted (Outliers Highlighted, N={len(df)})")
    ax.set_xlabel("Predicted Log Fare")
    ax.set_ylabel("Log Residual")


def main() -> None:
    df = add_log_predictions(load_metered())
    worst = df[df["log_residuals"] < THRESHOLD]
    print(f"trips with log_residuals < {THRESHOLD}: {len(worst)} of {len(df)}")

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 6.5))
    draw_fare_vs_distance(ax_left, df, worst)
    draw_residuals(ax_right, df, worst)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
