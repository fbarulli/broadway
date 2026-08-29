"""03: histogram of filtered fare_amount (all fares, no cutoff)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from _common import CLEAN_PARQUET, RESULTS

OUT = RESULTS / f"{Path(__file__).stem}.png"


def load_filtered() -> pd.DataFrame:
    if not CLEAN_PARQUET.exists():
        raise FileNotFoundError(f"{CLEAN_PARQUET} not found — run 01_filtered_min_max_scatter.py first")
    return pd.read_parquet(CLEAN_PARQUET)


def plot_histogram(df: pd.DataFrame, out_path: Path) -> None:
    values = df["fare_amount"]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(values, bins=60, edgecolor="white", linewidth=0.5)
    ax.set_xlim(float(values.min()), float(values.max()))
    ax.set_xlabel("fare_amount ($)")
    ax.set_ylabel("count")
    ax.set_title(f"fare_amount histogram (filtered, N={len(df)})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_filtered()
    plot_histogram(df, OUT)
    print(f"filtered rows: {len(df)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
