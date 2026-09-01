"""06: Tiered exact-duplicate dedupe — deliberate representatives, not blind drops.

05b found 13,519 rows in retailer+title duplicate groups, but only 4,439 also
match on price — so ~9,080 same-title rows at one retailer are DISTINCT
marketplace offers (Gittigidiyor-style sellers), not scrape glitches.
Collapsing them is PRICE-AGGREGATION, not noise-removal, so the representative
is chosen deliberately: has barcode (trusted identity) > most complete >
lowest price.

Tiers (each operates on the rows surviving the previous tier):
  T1 retailer+barcode     same product at one retailer -> 1 row (barcode is
                          ground truth, highest confidence).
  T2 retailer+title+price identical everything -> 1 row (lossless).
  T3 retailer+title       varying price -> 1 deliberate representative; the
                          group is FLAGGED as an ambiguous offer (pv) so the
                          price-collapse is auditable, never silent.

The deduped dataset is the MATCHING-stage input (step 03+); the raw export
stays the source of truth (load_dataset unchanged).

Writes:
  project/data/euromonitor/dataset_deduped.csv   deduped dataset (pipeline input)
  results/06_dedupe_summary.csv                  per-tier counts (display table)
  results/06_ambiguous_offer_groups.csv          retailer+title with >1 price
"""

import pandas as pd
from _common import DATA_PATH, RESULTS, load_dataset

CSV_SUMMARY = RESULTS / "06_dedupe_summary.csv"
CSV_OFFERS = RESULTS / "06_ambiguous_offer_groups.csv"
DEDUPED_PATH = DATA_PATH.with_name("dataset_deduped.csv")

HELPERS = ["_price", "_nonnull", "_has_bc"]


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    n0 = len(df)
    work = df.assign(
        _price=pd.to_numeric(df["price"], errors="coerce"),
        _nonnull=df.notna().sum(axis=1),
        _has_bc=(df["barcode"].fillna("").str.len() > 0).astype(int),
    )

    def keep_best(groups: list[str], sort_cols: list[str],
                  ascending: list[bool]) -> pd.DataFrame:
        return (work.sort_values(sort_cols, ascending=ascending,
                                 na_position="last")
                    .drop_duplicates(groups, keep="first"))

    summary = []

    # T1: retailer+barcode -> one row (ground-truth identity)
    t1 = keep_best(["retailer", "barcode"], ["_nonnull", "_price"], [False, True])
    dropped1 = work.index.difference(t1.index)
    work = work.drop(dropped1)
    summary.append({"tier": "T1 retailer+barcode",
                    "dropped_rows": len(dropped1)})

    # T2: retailer+title+price -> one row (lossless)
    t2 = keep_best(["retailer", "title", "price"], ["_nonnull", "_price"],
                   [False, True])
    dropped2 = work.index.difference(t2.index)
    work = work.drop(dropped2)
    summary.append({"tier": "T2 retailer+title+price",
                    "dropped_rows": len(dropped2)})

    # T3: retailer+title with varying price -> deliberate representative + flag
    pv = work.groupby(["retailer", "title"])["_price"].agg(
        ["count", "nunique", "min", "max"])
    ambiguous = pv[(pv["count"] > 1) & (pv["nunique"] > 1)]
    t3 = keep_best(["retailer", "title"], ["_has_bc", "_nonnull", "_price"],
                   [False, False, True])
    dropped3 = work.index.difference(t3.index)
    work = work.drop(dropped3)
    summary.append({"tier": "T3 retailer+title (price-aggregation)",
                    "dropped_rows": len(dropped3)})

    # ---- outputs --------------------------------------------------------------
    deduped = work.drop(columns=HELPERS).reset_index(drop=True)
    deduped.to_csv(DEDUPED_PATH, index=False)
    print(f"wrote {DEDUPED_PATH} ({len(deduped):,} rows)")

    # sanity: no (retailer,title) duplicates may remain
    dups = int(deduped.duplicated(subset=["retailer", "title"]).sum())
    if dups:
        raise AssertionError(f"sanity FAILED: {dups} retailer+title dupes remain")
    print("  [PASS] no retailer+title duplicates remain")

    summary.append({"tier": "TOTAL dropped", "dropped_rows": n0 - len(deduped)})
    summary.append({"tier": "TOTAL remaining", "dropped_rows": len(deduped)})
    pd.DataFrame(summary).to_csv(CSV_SUMMARY, index=False)
    print(f"wrote {CSV_SUMMARY} (display table, {len(summary)} rows)")

    ambiguous_out = ambiguous.rename(
        columns={"count": "rows", "nunique": "distinct_prices"}).reset_index()
    ambiguous_out.to_csv(CSV_OFFERS, index=False)
    print(f"wrote {CSV_OFFERS} (display table, {len(ambiguous_out)} rows) "
          f"— {len(ambiguous_out):,} ambiguous-offer groups flagged")

    print(f"\n{len(df):,} rows -> {len(deduped):,} after tiered dedupe "
          f"(dropped {n0 - len(deduped):,}); "
          f"{len(ambiguous_out):,} retailer+title groups were "
          "price-varying offers (flagged, not silently merged)")


if __name__ == "__main__":
    main()
