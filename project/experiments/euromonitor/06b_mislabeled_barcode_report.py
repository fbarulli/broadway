"""06b: Mislabeled-barcode report — identical title+brand, conflicting barcodes.

Same exact product title at 2+ retailers should carry ONE barcode. Groups
where it carries several are either barcode errors or the same product under
two GTINs. They are FLAGGED for human review, never merged — barcode is the
ground-truth key and we don't silently override it. This is also the direct
explanation of the 04 finding (negative-pair max TF-IDF similarity 0.704):
near-identical titles that the barcode layer says are different products.

Writes:
  results/06b_conflicting_barcode_groups.csv   the reviewer artifact (display)
  results/06b_conflicting_summary.csv          one-row summary (display table)
"""

import pandas as pd
from _common import RESULTS, load_dataset

CSV_CONFLICTS = RESULTS / "06b_conflicting_barcode_groups.csv"
CSV_SUMMARY = RESULTS / "06b_conflicting_summary.csv"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    has_bc = df["barcode"].fillna("").astype(str).str.len() > 0

    cb = (
        df.groupby(["title", "brand"], dropna=False)
          .agg(
              n_retailers=("retailer", "nunique"),
              n_barcodes=("barcode", lambda s: int(s[has_bc.reindex(s.index)].nunique())),
              barcodes=("barcode", lambda s: sorted({
                  str(x) for x in s.dropna() if str(x)})),
              retailers=("retailer", lambda s: sorted(set(s.dropna()))),
          )
          .reset_index()
    )
    cb["n_barcodes"] = cb["barcodes"].apply(len)
    conflicts = cb[(cb["n_retailers"] > 1) & (cb["n_barcodes"] > 1)]

    # sanity: same definition as 05b's corrected probe (cross-retailer, >1
    # unique real barcode) — both must agree on 339.
    if len(conflicts) != 339:
        print(f"NOTE: expected 339 conflicting groups (05b agrees); "
              f"this run finds {len(conflicts)} — definitions diverged")
    conflicts.to_csv(CSV_CONFLICTS, index=False)
    print(f"wrote {CSV_CONFLICTS} (reviewer artifact, {len(conflicts)} rows)")

    summary = pd.DataFrame({
        "metric": ["exact_title_cross_retailer_groups",
                   "conflicting_barcode_groups",
                   "clean_calibration_positives",
                   "conflict_rate"],
        "value": [int(((cb["n_retailers"] > 1)).sum()),
                  len(conflicts),
                  int(((cb["n_retailers"] > 1) & (cb["n_barcodes"] <= 1)).sum()),
                  round(len(conflicts) / max(int((cb["n_retailers"] > 1).sum()), 1), 4)],
    })
    summary.to_csv(CSV_SUMMARY, index=False)
    print(f"wrote {CSV_SUMMARY} (display table, {len(summary)} rows)")

    print(f"\nexact-title cross-retailer groups: {int((cb['n_retailers'] > 1).sum()):,}")
    print(f"  with CONFLICTING barcodes (flagged): {len(conflicts):,}  "
          f"({len(conflicts) / (cb['n_retailers'] > 1).sum():.1%})")
    print(f"  CLEAN calibration positives: "
          f"{int(((cb['n_retailers'] > 1) & (cb['n_barcodes'] <= 1)).sum()):,}")
    if len(conflicts):
        print("\ntop conflicting groups by retailer spread:")
        for _, r in conflicts.nlargest(5, "n_retailers").iterrows():
            print(f"  {r['title'][:48]:<48} retailers={r['n_retailers']} "
                  f"barcodes={r['barcodes'][:3]}")


if __name__ == "__main__":
    main()
