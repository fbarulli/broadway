"""06c: Validation-set carving — calibration positives + the HARD band.

Exact-title cross-retailer matches score TF-IDF cosine ~1.0 BY CONSTRUCTION —
validating only on them inflates perceived performance. Two honest sets:

  calibration   exact-title cross-retailer groups with a SINGLE barcode
                (clean labels; the threshold must accept ~100% of these).
  hard set      pairs whose TF-IDF cosine falls in the 0.3-0.8 band — where
                the fuzzy classifier actually earns its keep. Split into
                hard positives (same barcode) and hard negatives (cross
                barcode, sampled): this is the set for real precision/recall.

Writes:
  results/06c_validation_summary.csv   counts of both sets (display table)
  results/06c_hard_validation_pairs.csv  (title_a, title_b, cosine, label)
"""

import numpy as np
import pandas as pd
from _blocking import build_true_pairs
from _common import RESULTS, SEED, load_dataset_deduped
from _hard_negatives import exact_title_groups
from _matching import build_vectorizer, score_pairs

CSV_SUMMARY = RESULTS / "06c_validation_summary.csv"
CSV_HARD = RESULTS / "06c_hard_validation_pairs.csv"

LOW, HIGH = 0.3, 0.8  # the hard band: moderate similarity
N_NEG = 20_000
HARD_NEG_CAP = 3_000  # requested hard-negative sample size


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_dataset_deduped()
    titles = df["title"].fillna("").str.strip().tolist()
    rng = np.random.default_rng(SEED)

    # ---- calibration set: clean exact-title cross-retailer groups -------------
    # clean = cross-retailer AND <=1 UNIQUE real barcode (several rows sharing
    # one barcode are still clean; only genuinely conflicting barcodes aren't).
    # Shared definition with 06b (exact_title_groups) so the calibration census
    # can never drift from the conflicting-barcode report.
    cb = exact_title_groups(df)
    clean = cb[(cb["n_retailers"] > 1) & (cb["n_barcodes"] <= 1)]
    n_calib = len(clean)
    print(f"calibration positives (exact-title, single barcode): {n_calib:,}")

    # ---- hard band --------------------------------------------------------------
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(titles)
    print(f"vectorizer: {X.shape[1]:,} features over {X.shape[0]:,} titles")

    # hard positives: same-barcode pairs with cosine in [0.3, 0.8)
    pairs = build_true_pairs(df)
    pos_a = [a for a, _ in pairs]
    pos_b = [b for _, b in pairs]
    pos_cos = np.array(score_pairs(X, np.array(pos_a), np.array(pos_b)))
    band = (pos_cos >= LOW) & (pos_cos < HIGH)
    hard_pos = [(a, b, c) for (a, b), c in zip(pairs, pos_cos) if (c >= LOW and c < HIGH)]
    print(f"true pairs: {len(pairs):,}  in hard band: {int(band.sum()):,}")

    # hard negatives: sampled cross-barcode pairs in the same band, scored in
    # ONE vectorized bulk pass (no per-attempt Python loop). Both rows must
    # carry a non-empty (different) barcode, matching _blocking.py and
    # _hard_negatives.py: a GTIN-missing row has unknown ground truth.
    bc = df["barcode"].fillna("").astype(str).to_numpy()
    tt = np.array(titles)
    n = len(df)
    a = rng.integers(0, n, size=N_NEG * 8)
    b = rng.integers(0, n, size=N_NEG * 8)
    mask = (a != b) & (bc[a] != "") & (bc[b] != "") & (bc[a] != bc[b]) & (tt[a] != tt[b])
    cand_a, cand_b = a[mask], b[mask]
    cos = np.array(score_pairs(X, cand_a, cand_b))
    band_mask = (cos >= LOW) & (cos < HIGH)
    hard_neg = [(int(cand_a[k]), int(cand_b[k]), float(cos[k]))
                for k in np.flatnonzero(band_mask)[:HARD_NEG_CAP]]

    # The TF-IDF hard band is sparse on the negative side: fail loudly when it
    # is empty, and warn when it falls short of the requested sample so the
    # shortfall is never a silent pass.
    if not hard_neg:
        raise RuntimeError(
            f"no hard negatives found in cosine band [{LOW}, {HIGH}); "
            "validation set is empty")
    if len(hard_neg) < HARD_NEG_CAP:
        print(f"WARNING: hard band produced {len(hard_neg):,} negatives, below the "
              f"{HARD_NEG_CAP:,} target — the TF-IDF negative band is nearly empty "
              f"({len(hard_pos):,} hard positives vs {len(hard_neg):,} hard negatives)")
    print(f"hard negatives sampled in band: {len(hard_neg):,}")

    rows = ([{"title_a": titles[a], "title_b": titles[b], "cosine": round(c, 4),
              "label": "positive"} for a, b, c in hard_pos] +
            [{"title_a": titles[a], "title_b": titles[b], "cosine": round(c, 4),
              "label": "negative"} for a, b, c in hard_neg])
    hard_frame = pd.DataFrame(rows)
    hard_frame.to_csv(CSV_HARD, index=False)
    print(f"wrote {CSV_HARD} ({len(hard_frame):,} pairs)")

    summary = pd.DataFrame({
        "metric": ["calibration_positives", "hard_positive_pairs",
                   "hard_negative_pairs", "band"],
        "value": [n_calib, len(hard_pos), len(hard_neg),
                  f"{LOW}-{HIGH} (TF-IDF cosine)"],
    })
    summary.to_csv(CSV_SUMMARY, index=False)
    print(f"wrote {CSV_SUMMARY} (display table, {len(summary)} rows)")

    print("\nvalidation sets carved:")
    print(f"  calibration (exact-title, single barcode): {n_calib:,} — "
          f"threshold must accept ~100%")
    print(f"  hard positives (same barcode, cosine {LOW}-{HIGH}): {len(hard_pos):,}")
    print(f"  hard negatives (cross barcode, cosine {LOW}-{HIGH}): {len(hard_neg):,}")


if __name__ == "__main__":
    main()
