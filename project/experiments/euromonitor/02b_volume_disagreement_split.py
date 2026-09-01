"""02b: Split within-BARCODE volume disagreements into near-miss vs likely-variant/error.

Takes the flagged groups from 02 (within_barcode_volume_agreement, 57 groups)
and buckets them by max/min volume ratio:
  - near-miss (ratio < NEAR_MISS_THRESHOLD): likely rounding/bucket noise,
    extractor artifact, or a genuine near-duplicate — safe to treat as
    "agrees" for matching purposes after inspection.
  - likely_variant_or_error (ratio >= threshold): either a genuine pack-size
    line extension sharing a BARCODE (rare, non-standard per GS1 rules) or a
    mislabelled/miscoded BARCODE in source data. Flagged for exclusion from
    training pairs unless manually confirmed as a real variant.

Writes:
  02b_volume_disagreement_split.csv   full detail per flagged BARCODE group
  02b_disagreement_summary.csv        counts + threshold used

NOTE on canonical volume source: canonical_volume_ml comes from title
ONLY (the validated v2 extractor). The description field is NOT used as a
fallback — its nutrition/serving/dilution prose ("per 12 fl oz", "0.2 l glass",
"dilute in 9 volumes") injects systematic false volumes (proven: fallback
agreement 96.5% + 198 flagged vs name-only 99.0% + 57 flagged). Description
stays an explicit low-confidence feature, never folded in.
"""

import pandas as pd
from _common import RESULTS, barcode_agreement_table, canonical_volume, load_euromonitor

NEAR_MISS_THRESHOLD = 1.05  # <5% relative diff treated as noise, not a real gap

CSV_SPLIT = RESULTS / "02b_volume_disagreement_split.csv"
CSV_SUMMARY = RESULTS / "02b_disagreement_summary.csv"


def build_barcode_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Per-BARCODE agreement table via the shared helper (same logic as 02)."""
    agree = barcode_agreement_table(df, [("canonical_volume_ml", "canonical")], sample=True)
    return agree.dropna(subset=["canonical_agree"])  # honest denominator, exclude empty groups


def classify_disagreements(agree: pd.DataFrame, threshold: float = NEAR_MISS_THRESHOLD) -> pd.DataFrame:
    """Bucket flagged (disagreeing) groups by volume ratio."""
    flagged = agree[agree["canonical_agree"] == False].copy()
    flagged["vol_ratio"] = flagged["canonical_volumes"].apply(
        lambda v: max(v) / min(v) if len(v) > 1 and min(v) > 0 else float("nan")
    )
    flagged["vol_gap_ml"] = flagged["canonical_volumes"].apply(
        lambda v: max(v) - min(v) if len(v) > 1 else 0
    )
    flagged["bucket"] = flagged["vol_ratio"].apply(
        lambda r: "near_miss" if pd.notna(r) and r < threshold else "likely_variant_or_error"
    )
    return flagged.sort_values("vol_ratio", ascending=False)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_euromonitor()

    # canonical volume from title ONLY (validated v2; see module docstring)
    vol = canonical_volume(df["title"])
    df["canonical_volume_ml"] = vol["canonical_volume_ml"]
    df["canonical_volume_ambiguous"] = vol["canonical_volume_ambiguous"]

    # Description-derived volume kept as a SEPARATE low-confidence signal only
    # (e.g. for future tie-breaking or coverage backfill review) — never
    # merged into canonical_volume_ml, and not used in this split.
    desc = canonical_volume(df["description"])
    df["description_volume_ml"] = desc["canonical_volume_ml"]
    df["description_volume_ambiguous"] = desc["canonical_volume_ambiguous"]

    agree = build_barcode_agreement(df)
    flagged = classify_disagreements(agree)

    flagged.to_csv(CSV_SPLIT, index=False)
    print(f"wrote {CSV_SPLIT} ({len(flagged)} flagged groups)")

    summary = pd.DataFrame({
        "metric": [
            "total_flagged_groups",
            "near_miss_count",
            "likely_variant_or_error_count",
            "near_miss_threshold_ratio",
        ],
        "value": [
            len(flagged),
            int((flagged["bucket"] == "near_miss").sum()),
            int((flagged["bucket"] == "likely_variant_or_error").sum()),
            NEAR_MISS_THRESHOLD,
        ],
    })
    summary.to_csv(CSV_SUMMARY, index=False)
    print(f"wrote {CSV_SUMMARY}")

    print("\n=== near_miss (likely rounding noise) ===")
    for _, r in flagged[flagged["bucket"] == "near_miss"].iterrows():
        print(f"BARCODE {r['barcode']}: vols={r['canonical_volumes']} ratio={r['vol_ratio']:.3f} gap={r['vol_gap_ml']}ml")

    print("\n=== likely_variant_or_error (inspect manually) ===")
    for _, r in flagged[flagged["bucket"] == "likely_variant_or_error"].iterrows():
        print(f"BARCODE {r['barcode']}: vols={r['canonical_volumes']} ratio={r['vol_ratio']:.3f} "
              f"names={r['sample_names']} retailers={r['sample_retailers']}")


if __name__ == "__main__":
    main()
