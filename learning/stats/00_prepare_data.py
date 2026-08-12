"""
00_prepare_data.py

Deterministic preparation step — loads raw data, joins zones, takes a stratified
sample, and caches the result. All downstream diagnostic scripts (08-12) read
from this cache, guaranteeing they use the same sample for comparability.

Run: python learning/stats/00_prepare_data.py
"""

from _config import (
    RESULTS_DIR, SAMPLE_CACHE, SAMPLE_META,
    SAMPLE_SIZE, RANDOM_STATE, PICKUP_BOROUGH_COL,
    load_boroughs_pandas, _params_hash,
)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_boroughs_pandas()
    frac = min(1.0, SAMPLE_SIZE / len(df))
    sample = (
        df.groupby(PICKUP_BOROUGH_COL)
        .sample(frac=frac, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    sample.to_parquet(SAMPLE_CACHE)
    SAMPLE_META.write_text(
        '{{"params_hash": "{}"}}'.format(_params_hash())
    )
    print(f"wrote {len(sample)} rows to {SAMPLE_CACHE}")


if __name__ == "__main__":
    main()
