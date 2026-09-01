"""04: TF-IDF cosine title matching — prototype + ground-truth validation.

The matching primitive for entity resolution: titles are TF-IDF vectorized
((1,2) word ngrams, sublinear TF) and similarity is cosine. Validated on the
7,040 multi-retailer barcode groups: title pairs INSIDE the same barcode are
TRUE matches (same product, different retailers), pairs ACROSS barcodes are
non-matches. This measures how separable the two populations are — the
blocking/threshold trade-off for step 03.

The user's demo pair is real ground truth: "clearspring Organic King coconut
100% coconut water 350 ml" (El Corte Ingles) vs "coconut water ECO
clearspring, 350 ml" (Greenweez) share barcode 5021554989646 — the step
scores both the literal strings and the actual same-barcode row pair.

Metrics:
  P(pos > neg)  probability a random true pair outscores a random non-pair
                (1.0 = perfectly separable, 0.5 = coin flip).
  threshold sweep recall@t vs fpr@t — the blocking trade-off.

Writes (RESULTS = project/experiments/results/euromonitor/):
  04_tfidf_stats.csv        pos/neg score stats (display table)
  04_tfidf_threshold.csv    threshold sweep recall/fpr (display table)
  04_tfidf_scores.png       pos vs neg cosine histograms
  04_tfidf_threshold.png    recall/fpr sweep curve
"""

from itertools import combinations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import RESULTS, load_dataset
from _matching import build_vectorizer, score_pairs
from sklearn.metrics.pairwise import cosine_similarity

CSV_STATS = RESULTS / "04_tfidf_stats.csv"
CSV_THRESHOLD = RESULTS / "04_tfidf_threshold.csv"
PNG_SCORES = RESULTS / "04_tfidf_scores.png"
PNG_THRESHOLD = RESULTS / "04_tfidf_threshold.png"

SEED = 42
MAX_POS_PAIRS_PER_GROUP = 4
N_NEG_PAIRS = 10_000

# The user's demo strings — both real rows under barcode 5021554989646.
DEMO_A = "clearspring Organic King coconut 100% coconut water 350 ml"
DEMO_B = "coconut water ECO clearspring, 350 ml"
DEMO_BARCODE = "5021554989646"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    titles = df["title"].fillna("").str.strip().tolist()
    title_to_idx = {t: i for i, t in enumerate(titles)}
    barcodes = df["barcode"].fillna("").astype(str)
    rng = np.random.default_rng(SEED)

    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(titles)
    print(f"vectorizer: {X.shape[1]:,} features over {X.shape[0]:,} titles")

    # ---- positive pairs: titles inside the same multi-retailer barcode -------
    known = df[barcodes.str.len() > 0]
    multi = known[known.groupby("barcode")["retailer"].transform("nunique") > 1]
    pos_i, pos_j = [], []
    for _, g in multi.groupby("barcode"):
        uniq = [t for t in g["title"].dropna().str.strip().unique().tolist() if t]
        combos = list(combinations(range(len(uniq)), 2))
        if len(combos) > MAX_POS_PAIRS_PER_GROUP:
            chosen = rng.choice(len(combos), MAX_POS_PAIRS_PER_GROUP, replace=False)
            combos = [combos[k] for k in chosen]
        for a, b in combos:
            pos_i.append(title_to_idx[uniq[a]])
            pos_j.append(title_to_idx[uniq[b]])

    # ---- negative pairs: different barcodes AND different title text ----------
    neg_i, neg_j = [], []
    attempts = 0
    while len(neg_i) < N_NEG_PAIRS and attempts < N_NEG_PAIRS * 60:
        attempts += 1
        a, b = rng.choice(len(df), 2, replace=False)
        if barcodes.iloc[a] != barcodes.iloc[b] and titles[a] != titles[b]:
            neg_i.append(a)
            neg_j.append(b)

    pos_scores = np.array(score_pairs(X, np.array(pos_i), np.array(pos_j)))
    neg_scores = np.array(score_pairs(X, np.array(neg_i), np.array(neg_j)))

    # ---- separation + threshold sweep -----------------------------------------
    sub, subn = min(5000, len(pos_scores)), min(5000, len(neg_scores))
    p_pos_gt_neg = float(np.mean(pos_scores[:sub, None] > neg_scores[:subn][None, :]))
    ts = np.round(np.linspace(0.0, 1.0, 41), 3)
    recall = np.array([(pos_scores >= t).mean() for t in ts])
    fpr = np.array([(neg_scores >= t).mean() for t in ts])
    # The meaningful operating point: the LARGEST threshold that still keeps
    # 95% recall (t=0 is degenerate — every pair passes, fpr=1).
    recall_ok = recall >= 0.95
    t95 = float(ts[recall_ok][-1]) if recall_ok.any() else float("nan")
    fpr_t95 = float(fpr[recall_ok][-1]) if recall_ok.any() else float("nan")

    # ---- the user's demo pair + the real same-barcode row pair ----------------
    demo_X = vectorizer.transform([DEMO_A, DEMO_B])
    demo = float(cosine_similarity(demo_X[0:1], demo_X[1:2])[0][0])
    demo_rows = [t for t in
                 df[df["barcode"] == DEMO_BARCODE]["title"].dropna().str.strip().unique().tolist()
                 if t]
    real_score = float("nan")
    if len(demo_rows) >= 2:
        real_score = float(score_pairs(
            X, np.array([title_to_idx[demo_rows[0]]]),
            np.array([title_to_idx[demo_rows[1]]]))[0])

    # ---- sanity checks (fail loudly) -------------------------------------------
    checks = [
        ("S1 positive mean > negative mean", pos_scores.mean() > neg_scores.mean()),
        ("S2 P(pos > neg) > 0.85", p_pos_gt_neg > 0.85),
        ("S3 demo pair outscores negative median", demo > np.median(neg_scores)),
        ("S4 real same-barcode pair outscores negative median",
         (not np.isnan(real_score)) and real_score > np.median(neg_scores)),
        ("S5 95%-recall threshold exists", not np.isnan(t95)),
    ]
    for name, ok in checks:
        if not ok:
            raise AssertionError(f"sanity check FAILED: {name}")

    # ---- display tables ----------------------------------------------------------
    def _stats(s: np.ndarray) -> pd.Series:
        return pd.Series({
            "n": len(s), "mean": round(float(s.mean()), 4),
            "median": round(float(np.median(s)), 4),
            "p10": round(float(np.percentile(s, 10)), 4),
            "p90": round(float(np.percentile(s, 90)), 4),
            "min": round(float(s.min()), 4), "max": round(float(s.max()), 4),
        })

    stat_frame = pd.DataFrame({
        "population": ["positive (same barcode)", "negative (cross barcode)"],
    })
    stat_frame = pd.concat(
        [stat_frame, pd.concat([_stats(pos_scores), _stats(neg_scores)], axis=1).T.reset_index(drop=True)],
        axis=1)
    stat_frame.to_csv(CSV_STATS, index=False)
    print(f"wrote {CSV_STATS} (display table, {len(stat_frame)} rows)")
    thr_frame = pd.DataFrame({
        "threshold": ts, "recall": recall.round(4), "fpr": fpr.round(4)})
    thr_frame.to_csv(CSV_THRESHOLD, index=False)
    print(f"wrote {CSV_THRESHOLD} (display table, {len(thr_frame)} rows)")

    # ---- figures ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    ax.hist(pos_scores, bins=50, alpha=0.65, color="#4C72B0",
            label=f"positive same-barcode (n={len(pos_scores):,})")
    ax.hist(neg_scores, bins=50, alpha=0.65, color="#C44E52",
            label=f"negative cross-barcode (n={len(neg_scores):,})")
    pos_counts, _ = np.histogram(pos_scores, bins=50)
    neg_counts, _ = np.histogram(neg_scores, bins=50)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(pos_counts.max(), neg_counts.max()) * 1.1)  # data-derived
    ax.set_xlabel("TF-IDF cosine similarity")
    ax.set_ylabel("pair count")
    ax.set_title(f"Title similarity: same-barcode vs cross-barcode "
                 f"(P(pos>neg)={p_pos_gt_neg:.3f})")
    ax.legend(fontsize=8)
    fig.savefig(PNG_SCORES, dpi=150)
    plt.close(fig)
    print(f"wrote {PNG_SCORES}")

    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    ax.plot(ts, recall, color="#4C72B0", label="recall (true pairs above t)")
    ax.plot(ts, fpr, color="#C44E52", label="fpr (non-pairs above t)")
    if not np.isnan(t95):
        ax.axvline(t95, color="grey", linestyle="--", linewidth=1,
                   label=f"recall>=95% at t={t95:.2f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("cosine threshold t")
    ax.set_ylabel("fraction of pairs")
    ax.set_title("Blocking trade-off: recall vs false-positive rate")
    ax.legend(fontsize=8)
    fig.savefig(PNG_THRESHOLD, dpi=150)
    plt.close(fig)
    print(f"wrote {PNG_THRESHOLD}")

    # ---- printed report -----------------------------------------------------------
    print(f"\npositive pairs: {len(pos_scores):,}   negative pairs: {len(neg_scores):,}")
    print(f"  positive  mean={pos_scores.mean():.3f}  median={np.median(pos_scores):.3f}")
    print(f"  negative  mean={neg_scores.mean():.3f}  median={np.median(neg_scores):.3f}")
    print(f"  separation P(pos>neg) = {p_pos_gt_neg:.3f}")
    if not np.isnan(t95):
        print(f"  95%-recall threshold t = {t95:.2f}  (fpr at that t = {fpr_t95:.4f})")
    print(f"\ndemo pair ({DEMO_A[:44]}... vs {DEMO_B[:30]}...):  score = {demo:.3f}")
    if not np.isnan(real_score):
        print(f"real same-barcode pair {DEMO_BARCODE} ({demo_rows[0][:44]}... vs "
              f"{demo_rows[1][:30]}...):  score = {real_score:.3f}")
    print("\nsanity checks:")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


if __name__ == "__main__":
    main()
