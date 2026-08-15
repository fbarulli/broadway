"""Step 3: quantify the metered fare/distance "fan out".

Loads the cleaned sample, keeps metered fares (fare_amount < $55), plots a
histogram of fare_amount, then fits fare_amount ~ trip_distance and runs the
Breusch-Pagan test on the residuals to check for heteroskedasticity.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from broadway.stats.regression import bp_jb, fit_ols

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[2] / "experiments" / "results" / HERE.parents[0].name / HERE.name

METERED_CUTOFF = 55.0


def load_metered() -> pd.DataFrame:
    path = RESULTS / "sample_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run step1_clean_scatter.py first")
    df = pd.read_parquet(path)
    return df[df["fare_amount"] < METERED_CUTOFF]


def plot_histogram(metered: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(metered["fare_amount"], bins=60, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("fare_amount ($)")
    ax.set_ylabel("count")
    ax.set_title("metered fare_amount (fare < $55)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    metered = load_metered()
    plot_histogram(metered, RESULTS / "hist_fare_metered.png")

    model = fit_ols(metered, "fare_amount ~ trip_distance")
    result = bp_jb(model)
    print(f"metered rows: {len(metered)}")
    print(f"Breusch-Pagan p-value: {result['bp_pval']:.4f}")
    print(f"wrote {RESULTS / 'hist_fare_metered.png'}")


if __name__ == "__main__":
    main()
