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
from _matching import build_vectorizer, score_pairs

CSV_SUMMARY = RESULTS / "06c_validation_summary.csv"
CSV_HARD = RESULTS / "06c_hard_validation_pairs.csv"

LOW, HIGH = 0.3, 0.8  # the hard band: moderate similarity
N_NEG = 20_000


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_dataset_deduped()
    titles = df["title"].fillna("").str.strip().tolist()
    rng = np.random.default_rng(SEED)

    # ---- calibration set: clean exact-title cross-retailer groups -------------
    # clean = cross-retailer AND <=1 UNIQUE real barcode (several rows sharing
    # one barcode are still clean; only genuinely conflicting barcodes aren't).
    cb = (
        df.groupby(["title", "brand"], dropna=False)
          .agg(n_retailers=("retailer", "nunique"),
               n_barcodes=("barcode", lambda s: len({
                   str(x) for x in s.dropna() if str(x)})))
          .reset_index()
    )
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
    # ONE vectorized bulk pass (no per-attempt Python loop).
    bc = df["barcode"].fillna("").astype(str).to_numpy()
    tt = np.array(titles)
    n = len(df)
    a = rng.integers(0, n, size=N_NEG * 8)
    b = rng.integers(0, n, size=N_NEG * 8)
    mask = (a != b) & (bc[a] != "") & (bc[a] != bc[b]) & (tt[a] != tt[b])
    cand_a, cand_b = a[mask], b[mask]
    cos = np.array(score_pairs(X, cand_a, cand_b))
    band_mask = (cos >= LOW) & (cos < HIGH)
    hard_neg = [(int(cand_a[k]), int(cand_b[k]), float(cos[k]))
                for k in np.flatnonzero(band_mask)[:3000]]
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
