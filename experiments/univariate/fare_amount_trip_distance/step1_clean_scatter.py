"""Step 1: clean the raw fare_amount vs trip_distance relationship and plot it.

Draws a random 50k sample (no boroughs, no stratification) via the generic
read_sample loader, filters out garbage, saves the cleaned sample, and renders a
density scatter of trip_distance (x) vs fare_amount (y). Results are written to
experiments/results/<category>/<name>/ (gitignored).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from project.data import read_training_sample

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[2] / "experiments" / "results" / HERE.parents[0].name / HERE.name

SAMPLE_SIZE = 50_000
MIN_FARE = 2.50
MIN_DISTANCE = 0.0
MAX_DISTANCE = 50.0


def clean(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["fare_amount"] > MIN_FARE)
        & (df["trip_distance"] > MIN_DISTANCE)
        & (df["trip_distance"] <= MAX_DISTANCE)
    ]


def scatter(df: pd.DataFrame, out_path: Path) -> None:
    x = df["trip_distance"].to_numpy()
    y = df["fare_amount"].to_numpy()
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(x, y, s=1.5, alpha=0.15, edgecolors="none")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("trip_distance (miles)")
    ax.set_ylabel("fare_amount ($)")
    ax.set_title("fare_amount vs trip_distance (raw)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"axis limits: trip_distance [{x_min:.2f}, {x_max:.2f}], fare_amount [{y_min:.2f}, {y_max:.2f}]")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw = read_training_sample(
        sample=SAMPLE_SIZE, columns=["fare_amount", "trip_distance"]
    )
    cleaned = clean(raw)
    cleaned.to_parquet(RESULTS / "sample_clean.parquet")
    scatter(cleaned, RESULTS / "scatter.png")
    print(f"random sample rows: {len(raw)}")
    print(f"rows after filter: {len(cleaned)}")
    print(f"rows removed: {len(raw) - len(cleaned)}")
    print(f"wrote {RESULTS / 'sample_clean.parquet'}")
    print(f"wrote {RESULTS / 'scatter.png'}")


if __name__ == "__main__":
    main()
