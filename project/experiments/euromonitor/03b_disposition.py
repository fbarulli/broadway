"""03b: Fine-grained classification of within-BARCODE volume disagreements.

Extends 03's binary split (near_miss / likely_variant_or_error / pack_variant)
into the 7-category disposition taxonomy:

  pack_variant          ratio matches a pack count printed in the titles
  size_variant_real     non-round ratio, both volumes are plausible real sizes
                        (1L vs 1.5L, 8oz vs 10.5oz resize) — genuinely two
                        SKUs sharing one BARCODE; NOT an error, but ambiguous
                        ground truth -> exclude from training, tag distinctly
  typo                  ratio ~10/100/1000 or dropped digit/decimal
                        (33ml should be 330ml; 350l should be 350ml)
  ambiguous_unit        dry mix/powder where oz is weight, not fluid volume
  mixed_product         listings describe genuinely different products/flavors
                        under one BARCODE -> upstream data-quality issue
  unresolved_check_needed  round ratio (2/4/20) with no pack count visible
                        in the FULL listing text
  extraction_bug        extractor produced an implausible value (e.g. literal 0)

The `evidence` column records WHY (printed pack counts, full listing count,
ratio, volume list) so every classification is traceable.
"""

from __future__ import annotations

import ast
import re

import pandas as pd
from _common import RESULTS, load_euromonitor

CSV_IN = RESULTS / "03_pack_reconcile.csv"
CSV_OUT = RESULTS / "03b_disposition.csv"

from _text import DRY_MIX_HINTS, SUSPECT_ROUND, extract_pack_counts


def _list(value) -> list:
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [value]
    return list(value or [])


def classify_row(row: pd.Series, full_names: list[str]) -> tuple[str, str]:
    """Return (category, evidence) for one flagged group."""
    bucket = row["bucket"]
    if bucket == "near_miss":
        return "near_miss", "ratio < 1.05 (bucket rounding noise)"
    if bucket == "likely_pack_variant":
        return "pack_variant", f"ratio matches printed {row['matched_pack_count']}-pack"

    ratio = row["vol_ratio"]
    vols = _list(row["canonical_volumes"])
    if pd.isna(ratio):  # NaN or missing
        if any(v == 0 for v in vols):
            return "extraction_bug", f"vols={vols} include literal 0 (extractor artifact)"
        return "unresolved_check_needed", f"vols={vols} ratio=NaN"

    # typo: ratio ~10/100/1000 (dropped digit/decimal) or a single bad listing
    for magnitude in (1000, 100, 10):
        if abs(ratio - magnitude) <= 0.05 * magnitude:
            return "typo", f"ratio={ratio:.2f} ~ {magnitude}x (dropped digit/decimal)"

    # pack count present but not matching ratio? check full-text for ANY count
    counts = set()
    for nm in full_names:
        counts |= extract_pack_counts(nm)
    if counts and any(abs(ratio - n) <= 0.08 * n for n in counts):
        n = next(n for n in sorted(counts) if abs(ratio - n) <= 0.08 * n)
        return "pack_variant", f"ratio={ratio:.2f} matches {n}-pack in FULL titles"

    # dry mix / powder + bare-oz listings => ambiguous weight vs volume
    joined = " ".join(full_names).lower()
    if DRY_MIX_HINTS.search(joined) and "oz" in joined:
        return "ambiguous_unit", f"dry-mix product with oz (weight not volume); vols={vols}"

    # mixed products: names differ in flavor/product beyond size
    if len(full_names) >= 2:
        distinct_flavors = {re.sub(r"\d+[.\d]*\s*(ml|l|oz|cl|dl|gal|qt|pt)\b", "", n).strip()[:30]
                            for n in full_names}
        if len(distinct_flavors) >= 2 and len(full_names) >= 4:
            return "mixed_product", f"{len(distinct_flavors)} distinct product names under one BARCODE"

    # unresolved: round ratio with no pack count anywhere in full text
    for n in SUSPECT_ROUND:
        if abs(ratio - n) <= 0.12 * n:
            return "unresolved_check_needed", f"ratio={ratio:.2f} near {n}-pack but no count in FULL titles"

    # size_variant_real: non-round, both plausible real package sizes
    return "size_variant_real", f"vols={vols} ratio={ratio:.2f} (plausible real sizes)"


def main() -> None:
    df = pd.read_csv(CSV_IN)
    # full listing text per BARCODE from the raw dataset (not the 5-name sample)
    raw = load_euromonitor().reset_index(drop=True)
    raw["barcode_s"] = raw["barcode"].fillna("").astype(str)

    out_rows = []
    for _, r in df.iterrows():
        barcode = str(r["barcode"])
        full_names = raw.loc[raw["barcode_s"] == barcode, "title"].dropna().tolist()
        category, evidence = classify_row(r, full_names)
        out_rows.append({**r.to_dict(), "category": category, "evidence": evidence})
    out = pd.DataFrame(out_rows)
    out.to_csv(CSV_OUT, index=False)
    print(f"wrote {CSV_OUT}")
    print("\n=== disposition ===")
    for cat, n in out["category"].value_counts().items():
        print(f"  {cat}: {n}")
    print(f"\ntotal: {len(out)}")


if __name__ == "__main__":
    main()
