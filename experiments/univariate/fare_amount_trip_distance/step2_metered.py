"""Step 2: split metered vs flat fares and compare.

Reads the cleaned sample from step 1, splits trips into metered
(fare_amount < $55) vs flat/airport (>= $55), reports counts, and renders a
scatter distinguishing the two regimes.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[2] / "experiments" / "results" / HERE.parents[0].name / HERE.name

METERED_CUTOFF = 55.0


def load_cleaned() -> pd.DataFrame:
    path = RESULTS / "sample_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run step1_clean_scatter.py first")
    return pd.read_parquet(path)


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metered = df[df["fare_amount"] < METERED_CUTOFF]
    flat = df[df["fare_amount"] >= METERED_CUTOFF]
    return metered, flat


def plot_split(metered: pd.DataFrame, flat: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        metered["trip_distance"], metered["fare_amount"],
        s=1.5, alpha=0.15, edgecolors="none", label="metered (< $55)",
    )
    ax.scatter(
        flat["trip_distance"], flat["fare_amount"],
        s=1.5, alpha=0.3, edgecolors="none", color="#d62728", label="flat (>= $55)",
    )
    ax.axhline(METERED_CUTOFF, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("trip_distance (miles)")
    ax.set_ylabel("fare_amount ($)")
    ax.set_title("fare_amount vs trip_distance — metered vs flat")
    ax.grid(True, alpha=0.3)
    ax.legend(markerscale=5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_cleaned()
    metered, flat = split(df)
    plot_split(metered, flat, RESULTS / "metered_vs_flat_scatter.png")
    print(f"cleaned rows: {len(df)}")
    print(f"metered (< ${METERED_CUTOFF:.0f}): {len(metered)}")
    print(f"flat (>= ${METERED_CUTOFF:.0f}): {len(flat)}")
    print(f"wrote {RESULTS / 'metered_vs_flat_scatter.png'}")


if __name__ == "__main__":
    main()
