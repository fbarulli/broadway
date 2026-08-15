"""05: downsample to N=300 and run Breusch-Pagan (small-sample contrast)."""

import json

import pandas as pd

from _common import CLEAN_PARQUET, RESULTS, TESTS_JSON
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

    results = {}
    if TESTS_JSON.exists():
        results = json.loads(TESTS_JSON.read_text())
    results["breusch_pagan_n300"] = {"n": len(small), "p_value": small_p}
    results["breusch_pagan_full"] = {"n": len(df), "p_value": full_p}
    RESULTS.mkdir(parents=True, exist_ok=True)
    TESTS_JSON.write_text(json.dumps(results, indent=2))

    print(f"Breusch-Pagan p-value (N={len(small)}): {small_p:.4f}")
    print(f"Breusch-Pagan p-value (N={len(df)}): {full_p:.4f}")
    print(f"wrote {TESTS_JSON}")


if __name__ == "__main__":
    main()
