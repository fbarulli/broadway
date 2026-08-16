"""05: downsample to N=300 and run Breusch-Pagan (small-sample contrast).

Results are persisted per-dataset to tests_sample_50k.json (tied to
sample_50k.parquet) with provenance metadata.
"""

import pandas as pd

from _common import CLEAN_PARQUET
from _tests import write_tests_json
from broadway.stats.regression import bp_jb, fit_ols

SAMPLE_N = 300
SEED = 42


def main() -> None:
    if not CLEAN_PARQUET.exists():
        raise FileNotFoundError(f"{CLEAN_PARQUET} not found — run 01_filtered_min_max_scatter.py first")
    df = pd.read_parquet(CLEAN_PARQUET)

    small = df.sample(n=SAMPLE_N, random_state=SEED)
    small_p = bp_jb(fit_ols(small, "fare_amount ~ trip_distance"))["bp_pval"]
    full_p = bp_jb(fit_ols(df, "fare_amount ~ trip_distance"))["bp_pval"]

    results = {
        "breusch_pagan_n300": {"n": len(small), "seed": SEED, "p_value": small_p},
        "breusch_pagan_full": {"n": len(df), "p_value": full_p},
    }
    out = write_tests_json(CLEAN_PARQUET, results, "05_downsample_bp.py", n_rows=len(df))

    print(f"Breusch-Pagan p-value (N={len(small)}): {small_p:.4f}")
    print(f"Breusch-Pagan p-value (N={len(df)}): {full_p:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
