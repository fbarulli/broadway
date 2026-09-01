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
  project/data/euromonitor/sku_to_rep.csv        raw SKU (product_id) -> rep_id
  results/06_dedupe_summary.csv                  per-tier counts (display table)
  results/06_ambiguous_offer_groups.csv          retailer+title with >1 price
"""

import pandas as pd
from _common import DATA_PATH, RESULTS, load_dataset

CSV_SUMMARY = RESULTS / "06_dedupe_summary.csv"
CSV_OFFERS = RESULTS / "06_ambiguous_offer_groups.csv"
DEDUPED_PATH = DATA_PATH.with_name("dataset_deduped.csv")
SKU_TO_REP_PATH = DATA_PATH.with_name("sku_to_rep.csv")

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

    # parent[i] = original row index of the surviving representative for row i.
    # Updated per tier and resolved transitively at the end (a T1 survivor may
    # itself be collapsed by T2/T3).
    parent = {i: i for i in work.index}

    def collapse(frame: pd.DataFrame, groups: list[str], sort_cols: list[str],
                 ascending: list[bool]) -> tuple[pd.DataFrame, pd.Index]:
        """Collapse each group to one representative; record the mapping.

        Uses the SAME sort + first-of-group semantics as drop_duplicates
        (groupby dropna=False so NaN==NaN grouping matches drop_duplicates).
        """
        ordered = frame.sort_values(sort_cols, ascending=ascending,
                                    na_position="last")
        survivors = ordered.drop_duplicates(groups, keep="first")
        for _, grp in ordered.groupby(groups, sort=False, dropna=False):
            rep = grp.index[0]
            for idx in grp.index:
                parent[idx] = rep
        dropped = frame.index.difference(survivors.index)
        return survivors, dropped

    summary = []

    # T1: retailer+barcode -> one row (ground-truth identity), ONLY for rows
    # that actually have a barcode. Rows with a MISSING barcode are NOT
    # collapsed ("no barcode" is not "same barcode").
    with_bc = work[work["_has_bc"] == 1]
    no_bc = work[work["_has_bc"] == 0]
    t1, dropped1 = collapse(with_bc, ["retailer", "barcode"],
                            ["_nonnull", "_price"], [False, True])
    work = pd.concat([t1, no_bc])
    summary.append({"tier": "T1 retailer+barcode",
                    "dropped_rows": len(dropped1)})

    # T2: retailer+title+price -> one row (lossless), ONLY for rows that HAVE
    # a price. NaN != NaN in the real world, so two missing-price rows are not
    # "identical everything" and must flow to T3's auditable price-aggregation.
    with_price = work[work["_price"].notna()]
    no_price = work[work["_price"].isna()]
    t2, dropped2 = collapse(with_price, ["retailer", "title", "price"],
                            ["_nonnull", "_price"], [False, True])
    work = pd.concat([t2, no_price])
    summary.append({"tier": "T2 retailer+title+price",
                    "dropped_rows": len(dropped2)})

    # T3: retailer+title with varying price -> deliberate representative + flag
    pv = work.groupby(["retailer", "title"])["_price"].agg(
        ["count", "nunique", "min", "max"])
    ambiguous = pv[(pv["count"] > 1) & (pv["nunique"] > 1)]
    t3, dropped3 = collapse(work, ["retailer", "title"],
                            ["_has_bc", "_nonnull", "_price"], [False, False, True])
    work = t3
    summary.append({"tier": "T3 retailer+title (price-aggregation)",
                    "dropped_rows": len(dropped3)})

    # ---- outputs --------------------------------------------------------------
    # Resolve representative pointers transitively (T1/T2 survivors may be
    # dropped by a later tier), then emit the raw-SKU -> rep mapping so the
    # deliverable notebook derives ITEM_ID from this step instead of re-deduping.
    for i in df.index:
        while parent[parent[i]] != parent[i]:
            parent[i] = parent[parent[i]]

    deduped = work.drop(columns=HELPERS).reset_index(drop=True)
    deduped.to_csv(DEDUPED_PATH, index=False)
    print(f"wrote {DEDUPED_PATH} ({len(deduped):,} rows)")

    rep_pos = {idx: pos for pos, idx in enumerate(work.index)}
    sku_to_rep = pd.DataFrame({
        "product_id": df["product_id"].to_numpy(),
        "rep_id": [rep_pos[parent[i]] for i in df.index],
    })
    sku_to_rep.to_csv(SKU_TO_REP_PATH, index=False)
    print(f"wrote {SKU_TO_REP_PATH} ({len(sku_to_rep):,} rows)")

    # sanity: no (retailer,title) duplicates may remain, and every raw SKU
    # resolves to a valid representative.
    dups = int(deduped.duplicated(subset=["retailer", "title"]).sum())
    if dups:
        raise AssertionError(f"sanity FAILED: {dups} retailer+title dupes remain")
    if sku_to_rep["rep_id"].isna().any():
        raise AssertionError("sanity FAILED: some SKUs map to no representative")
    if set(sku_to_rep["rep_id"].unique()) != set(range(len(deduped))):
        raise AssertionError("sanity FAILED: rep_id coverage is not 0..n-1")
    print("  [PASS] no retailer+title duplicates remain; SKU->rep mapping complete")

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
