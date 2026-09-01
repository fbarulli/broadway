"""06b: Mislabeled-barcode report — identical title+brand, conflicting barcodes.

Same exact product title at 2+ retailers should carry ONE barcode. Groups
where it carries several are either barcode errors or the same product under
two GTINs. They are FLAGGED for human review, never merged — barcode is the
ground-truth key and we don't silently override it. This is also the direct
explanation of the 04 finding (negative-pair max TF-IDF similarity 0.796):
near-identical titles that the barcode layer says are different products.

Writes:
  results/06b_conflicting_barcode_groups.csv   the reviewer artifact (display)
  results/06b_conflicting_summary.csv          one-row summary (display table)
"""

import pandas as pd
from _common import RESULTS, load_dataset
from _hard_negatives import exact_title_groups

CSV_CONFLICTS = RESULTS / "06b_conflicting_barcode_groups.csv"
CSV_SUMMARY = RESULTS / "06b_conflicting_summary.csv"

N_CONFLICTING = 339  # snapshot expectation shared with 05b's corrected probe


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_dataset()

    cb = exact_title_groups(df)
    conflicts = cb[(cb["n_retailers"] > 1) & (cb["n_barcodes"] > 1)]

    # sanity: same definition as 05b's corrected probe (cross-retailer, >1
    # unique real barcode) — both must agree on 339. Fail loudly, don't just
    # note, so a definitional drift is never silent.
    if len(conflicts) != N_CONFLICTING:
        raise AssertionError(
            f"expected {N_CONFLICTING} conflicting groups (05b agrees); "
            f"this run finds {len(conflicts)} — definitions diverged")
    conflicts.to_csv(CSV_CONFLICTS, index=False)
    print(f"wrote {CSV_CONFLICTS} (reviewer artifact, {len(conflicts)} rows)")

    n_cross = int((cb["n_retailers"] > 1).sum())
    summary = pd.DataFrame({
        "metric": ["exact_title_cross_retailer_groups",
                   "conflicting_barcode_groups",
                   "clean_calibration_positives",
                   "conflict_rate"],
        "value": [n_cross,
                  len(conflicts),
                  int(((cb["n_retailers"] > 1) & (cb["n_barcodes"] <= 1)).sum()),
                  round(len(conflicts) / max(n_cross, 1), 4)],
    })
    summary.to_csv(CSV_SUMMARY, index=False)
    print(f"wrote {CSV_SUMMARY} (display table, {len(summary)} rows)")

    print(f"\nexact-title cross-retailer groups: {n_cross:,}")
    print(f"  with CONFLICTING barcodes (flagged): {len(conflicts):,}  "
          f"({len(conflicts) / max(n_cross, 1):.1%})")
    print(f"  CLEAN calibration positives: "
          f"{int(((cb['n_retailers'] > 1) & (cb['n_barcodes'] <= 1)).sum()):,}")
    if len(conflicts):
        print("\ntop conflicting groups by retailer spread:")
        for _, r in conflicts.nlargest(5, "n_retailers").iterrows():
            print(f"  {r['title'][:48]:<48} retailers={r['n_retailers']} "
                  f"barcodes={r['barcodes'][:3]}")


if __name__ == "__main__":
    main()
