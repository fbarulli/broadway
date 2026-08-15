"""05: downsample to N=300 and run Breusch-Pagan (small-sample contrast)."""

import pandas as pd

from _common import CLEAN_PARQUET
from broadway.stats.regression import bp_jb, fit_ols

SAMPLE_N = 300
SEED = 42


def main() -> None:
    if not CLEAN_PARQUET.exists():
        raise FileNotFoundError(f"{CLEAN_PARQUET} not found — run 01_filtered_min_max_scatter.py first")
    df = pd.read_parquet(CLEAN_PARQUET)

    small = df.sample(n=SAMPLE_N, random_state=SEED)
    small_result = bp_jb(fit_ols(small, "fare_amount ~ trip_distance"))

    full_result = bp_jb(fit_ols(df, "fare_amount ~ trip_distance"))

    print(f"Breusch-Pagan p-value (N={len(small)}): {small_result['bp_pval']:.4f}")
    print(f"Breusch-Pagan p-value (N={len(df)}): {full_result['bp_pval']:.4f}")


if __name__ == "__main__":
    main()
