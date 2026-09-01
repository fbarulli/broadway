"""03: Reconcile BARCODE volume-mismatch flags against pack counts in the titles.

A group flagged 'likely_variant_or_error' with vol_ratio ~= N is very likely
just: one retailer listed the single-unit volume, another listed the N-pack
total volume. That is NOT a data error. This step reclassifies such groups to
'likely_pack_variant' when the ratio matches a pack count actually printed on
one of the group's own listings.

Independent of VOLUME_RE — works on raw listing text (full names per BARCODE).
"""

from __future__ import annotations

import pandas as pd
from _common import RESULTS, parse_list_cell
from _text import is_pack_multiple

CSV_IN = RESULTS / "02b_volume_disagreement_split.csv"
CSV_OUT = RESULTS / "03_pack_reconcile.csv"


def reclassify_bucket(bucket: str, vol_ratio: float, sample_names: list[str]) -> tuple[str, int | None]:
    """Return (new_bucket, matched_pack_count). Only touches 'likely_variant_or_error'."""
    if bucket != "likely_variant_or_error":
        return bucket, None
    n = is_pack_multiple(vol_ratio, sample_names)
    if n is not None:
        return "likely_pack_variant", n
    return bucket, None


def main() -> None:
    df = pd.read_csv(CSV_IN)
    reclassified = 0
    out_rows = []
    for _, r in df.iterrows():
        names = parse_list_cell(r["sample_names"])
        new_bucket, matched_n = reclassify_bucket(
            r["bucket"], r["vol_ratio"], names)
        if new_bucket != r["bucket"]:
            reclassified += 1
            print(
                f"BARCODE {r['barcode']}: {r['bucket']} -> {new_bucket} "
                f"(matched {matched_n}-pack) vols={r['canonical_volumes']} "
                f"ratio={r['vol_ratio']:.3f}")
        out_rows.append({**r.to_dict(), "bucket": new_bucket,
                         "matched_pack_count": matched_n})
    out = pd.DataFrame(out_rows)
    out.to_csv(CSV_OUT, index=False)
    print(f"\nwrote {CSV_OUT}")
    print(f"{reclassified} / {len(df)} groups reclassified as pack variants, not errors")
    print(f"remaining likely_variant_or_error: "
          f"{int((out['bucket'] == 'likely_variant_or_error').sum())}")


if __name__ == "__main__":
    main()
