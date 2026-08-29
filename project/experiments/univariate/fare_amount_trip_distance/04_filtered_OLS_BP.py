"""04: fit OLS on filtered fares, show residuals, then run Breusch-Pagan.

Fits fare_amount ~ trip_distance on the filtered 50k sample, plots the
residuals-vs-fitted, and runs the Breusch-Pagan test, visualizing the verdict
against the standard normal rejection region (|z| > 1.96) with the observed
statistic annotated.
"""

from pathlib import Path

import pandas as pd
from _common import CLEAN_PARQUET, RESULTS
from _ols_bp import fit_and_plot_ols_bp

OUT = RESULTS / f"{Path(__file__).stem}.png"


def load_filtered() -> pd.DataFrame:
    if not CLEAN_PARQUET.exists():
        raise FileNotFoundError(f"{CLEAN_PARQUET} not found — run 01_filtered_min_max_scatter.py first")
    return pd.read_parquet(CLEAN_PARQUET)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_filtered()

    result = fit_and_plot_ols_bp(df, "fare_amount ~ trip_distance", OUT)

    print(f"filtered rows: {len(df)}")
    print(f"Breusch-Pagan p-value: {result['bp_pval']:.4f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
