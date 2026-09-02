"""07b: TripletLoss fine-tune on mined hard negatives (round 3, K-fold CV).

Trains a bi-encoder to separate same-product positives from the hard negatives
the zero-shot champion confuses (different brand, same macro category, widened
mid-cosine band 0.45-0.80). Evaluation is barcode-level K-fold CV: each fold
fine-tunes one epoch on the other K-1 folds and scores the held-out fold, then
the held-out hard-negative scores are POOLED and reported with a bootstrap 95%
CI on precision@90%recall — not a bare point estimate over a ~100-pair slice.

Round 3 adds auditable diagnostics on top of the round-2 K-fold pooling:
  - raw TP/FP/threshold counts per fold and pooled (precision_at_recall_breakdown),
  - band-crossing counts (how many held-out hard negatives fine-tuning pushed
    above 0.80 or below 0.45),
  - precision@90R stratified by in-country / cross-country / unknown-country positives,
  - per-fold AUC with population mean ± std,
  - a four-population score-distribution CSV (in/cross-country pos, hard neg,
    random neg) for the report step,
  - fine-tuned encode wall-time, and optional MLflow logging.

Why three fixes matter (round 1 reported precision@90%recall = 1.0000 on 108
held-out hard negatives):
  1. widened confusion band (0.45-0.80 vs 0.5-0.75) -> more test volume,
  2. bootstrap CI -> the precision@90%recall is a sample statistic, not a
     constant, so report its spread instead of a naked point estimate,
  3. group-aware K-fold pooling -> ~5x the eval volume and no single lucky fold.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from _blocking import build_pairs
from _common import PATHS, RESULTS, SEED, load_dataset_deduped
from _hard_negatives import build_triplets, mine_hard_negatives, pairs_in_set
from sklearn.metrics import roc_auc_score

from broadway.training.nlp import (
    _cosine,
    encode_corpus,
    log_nlp_eval,
    precision_at_recall_breakdown,
    precision_ci,
    split_pos_by_country,
)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOSS = "triplet"
CACHE = str(PATHS.experiments.parent / "data" / "euromonitor" / "embeddings_cache")
EPOCHS = 1
BATCH_SIZE = 32
LR = 2e-5
MAX_TRIPLES = 5_000
CV_FOLDS = 5
# widened confusion band (round 1 used 0.5-0.75) to grow the hard-negative pool
COSINE_LO = 0.45
COSINE_HI = 0.80
N_TARGET = 20_000
BOOTSTRAP = 2_000


def _auc(pos_s: np.ndarray, neg_s: np.ndarray) -> float:
    """ROC AUC of a held-out pos/neg pair score population (NaN if a side is empty)."""
    if len(pos_s) == 0 or len(neg_s) == 0:
        return float("nan")
    return float(roc_auc_score(
        np.r_[np.ones(len(pos_s)), np.zeros(len(neg_s))],
        np.r_[pos_s, neg_s]))


def kfold_barcodes(df, k: int, seed: int = SEED) -> list[set[str]]:
    """K barcode sets over multi-retailer barcodes, shuffled and split ~evenly.

    Splits on the barcode (entity) so no product's rows straddle a fold — the
    same group-aware guard as split_barcodes, generalised to K folds.
    """
    barcodes = df["barcode"].fillna("").astype(str)
    known = df[barcodes.str.len() > 0]
    multi = known[known.groupby("barcode")["retailer"].transform("nunique") > 1]
    bcs = np.array(sorted(multi["barcode"].unique()))
    perm = np.random.default_rng(seed).permutation(len(bcs))
    return [set(bcs[perm[i::k]]) for i in range(k)]


def main() -> None:
    t0 = time.perf_counter()
    df = load_dataset_deduped()
    payload = (df["title"].fillna("") + " | " + df["brand"].fillna("") + " | " + df["category"].fillna("")).tolist()
    row_bc = df["barcode"].fillna("").astype(str).to_numpy()
    # country per row (for the in-country vs cross-country positive stratification)
    country = df["country"].fillna("").astype(str).to_numpy()

    folds = kfold_barcodes(df, CV_FOLDS, SEED)
    print(f"barcode K-fold: {CV_FOLDS} folds "
          f"({', '.join(str(len(f)) for f in folds)} entities)", flush=True)

    pos, neg = build_pairs(df, SEED, 4, 10_000)

    # ---- zero-shot embeddings (cached) for mining, encoded ONCE ----
    emb0, encode_s0 = encode_corpus(MODEL, payload, batch_size=256, max_seq_length=128, cache_dir=CACHE)
    print(f"zero-shot corpus encode_s = {encode_s0:.1f}s", flush=True)

    # ---- mine hard negatives ONCE over the full corpus (widened band), then
    # each fold keeps only pairs whose BOTH endpoints live in that fold ----
    hard_pairs, _ = mine_hard_negatives(
        df, emb0, n_target=N_TARGET, cosine_lo=COSINE_LO, cosine_hi=COSINE_HI)
    print(f"hard negatives mined (band {COSINE_LO}-{COSINE_HI}, target {N_TARGET:,}): "
          f"{len(hard_pairs):,}", flush=True)

    from sentence_transformers import SentenceTransformer

    from broadway.training.nlp import _finetune

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "euromonitor-07b-finetune")
    base_params = {
        "n_boot": BOOTSTRAP, "seed": SEED, "model": MODEL, "loss": LOSS,
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
    }

    pooled_pos: list[np.ndarray] = []
    pooled_hard: list[np.ndarray] = []
    pooled_rand: list[np.ndarray] = []
    pooled_test_pos: list[np.ndarray] = []
    fold_aucs: list[float] = []
    encode_s_folds: list[float] = []
    four_pop_rows: list[tuple[str, float]] = []
    for f in range(CV_FOLDS):
        test_bc = folds[f]
        train_bc = set().union(*(folds[i] for i in range(CV_FOLDS) if i != f))
        train_pos = pos[pairs_in_set(pos, row_bc, train_bc)]
        hard_train = hard_pairs[pairs_in_set(hard_pairs, row_bc, train_bc)]
        triples = build_triplets(train_pos, hard_train, payload, seed=SEED, max_triples=MAX_TRIPLES)
        print(f"fold {f}: train_pos {len(train_pos):,} | hard_train {len(hard_train):,} "
              f"| triples {len(triples):,}", flush=True)
        if not triples:
            print(f"fold {f}: no triples, skipping", flush=True)
            continue

        model = SentenceTransformer(MODEL, device="cpu")
        model.max_seq_length = 128
        params = {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LR, "warmup_steps": 0}
        _finetune(model, params, triples, 128, loss=LOSS, log_steps=True)

        t_enc = time.perf_counter()
        emb = model.encode(payload, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        encode_s = time.perf_counter() - t_enc
        test_pos = pos[pairs_in_set(pos, row_bc, test_bc)]
        hard_test = hard_pairs[pairs_in_set(hard_pairs, row_bc, test_bc)]
        rand_test = neg[pairs_in_set(neg, row_bc, test_bc)]
        pos_s = _cosine(emb, test_pos)
        hard_s = _cosine(emb, hard_test)
        rand_s = _cosine(emb, rand_test)
        fold_auc = _auc(pos_s, rand_s)
        fold_aucs.append(fold_auc)
        encode_s_folds.append(encode_s)
        p, lo, hi = precision_ci(pos_s, hard_s, n_boot=BOOTSTRAP, seed=SEED)
        _, tp, fp, thr, rec = precision_at_recall_breakdown(pos_s, hard_s)
        above = int((hard_s > COSINE_HI).sum())   # > 0.80 — looks like a match
        below = int((hard_s < COSINE_LO).sum())   # < 0.45 — clearly not a match
        same, cross, unknown = split_pos_by_country(test_pos, country)
        n_unknown = int(unknown.sum())
        pooled_pos.append(pos_s)
        pooled_hard.append(hard_s)
        pooled_rand.append(rand_s)
        pooled_test_pos.append(test_pos)
        # four-population rows (fine-tuned, per fold)
        four_pop_rows += [("cross_country_pos", s) for s in pos_s[cross]]
        four_pop_rows += [("in_country_pos", s) for s in pos_s[same]]
        four_pop_rows += [("hard_neg", s) for s in hard_s]
        four_pop_rows += [("random_neg", s) for s in rand_s]
        # per-stratum precision@90R (never pooled across strata)
        for label, mask in (("in_country", same), ("cross_country", cross),
                            ("unknown_country", unknown)):
            n_pos = int(mask.sum())
            if n_pos == 0:
                sp, stp, sfp, sthr, srec = float("nan"), 0, 0, float("nan"), float("nan")
            else:
                sp, stp, sfp, sthr, srec = precision_at_recall_breakdown(pos_s[mask], hard_s, target_recall=0.90)
            print(f"fold {f} {label}: n_pos={n_pos} P@90R={sp:.4f} TP={stp} FP={sfp} "
                  f"thr={sthr:.4f} recall={srec:.4f}", flush=True)
            log_nlp_eval(
                {f"precision_at_90pct_recall_{label}": round(sp, 4),
                 f"tp_at_90pct_recall_{label}": float(stp),
                 f"fp_at_90pct_recall_{label}": float(sfp),
                 f"threshold_at_90pct_recall_{label}": round(sthr, 4),
                 f"{label}_n_pos": float(n_pos)},
                {**base_params, "fold": f, "stratum": label},
                tracking_uri, experiment_name,
            )
        print(f"fold {f}: test_pos {len(test_pos):,} | hard_test {len(hard_test):,} "
              f"| unknown {n_unknown} | P@90R {p:.4f} [{lo:.4f}, {hi:.4f}] "
              f"| TP {tp} FP {fp} thr {thr:.4f} recall {rec:.4f} "
              f"| band above {above} below {below} "
              f"| encode_s {encode_s:.1f}s | fold AUC {fold_auc:.4f} "
              f"| hard med {np.median(hard_s):.3f} p90 {np.quantile(hard_s, 0.9):.3f} max {hard_s.max():.3f}",
              flush=True)
        log_nlp_eval(
            {"precision_at_90pct_recall": round(p, 4),
             "tp_at_90pct_recall": float(tp),
             "fp_at_90pct_recall": float(fp),
             "threshold_at_90pct_recall": round(thr, 4),
             "recall_at_90pct": round(rec, 4),
             "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
             "hard_above_band": float(above), "hard_below_band": float(below),
             "encode_s": round(encode_s, 3)},
            {**base_params, "fold": f},
            tracking_uri, experiment_name,
        )

    pos_all = np.concatenate(pooled_pos)
    hard_all = np.concatenate(pooled_hard)
    rand_all = np.concatenate(pooled_rand)
    test_pos_all = np.concatenate(pooled_test_pos)
    p, lo, hi = precision_ci(pos_all, hard_all, n_boot=BOOTSTRAP, seed=SEED)
    _, tp_all, fp_all, thr_all, rec_all = precision_at_recall_breakdown(pos_all, hard_all)
    above_all = int((hard_all > COSINE_HI).sum())
    below_all = int((hard_all < COSINE_LO).sum())
    print(f"\npooled across {len(pooled_pos)} folds: test_pos {len(pos_all):,} "
          f"| hard_test {len(hard_all):,}", flush=True)
    print(f"bootstrap: n_boot={BOOTSTRAP} seed={SEED}", flush=True)
    print(f"precision@90pct-recall (hard band)  {p:.4f}  [95% CI {lo:.4f}, {hi:.4f}]",
          flush=True)
    print(f"  TP {tp_all} | FP {fp_all} | threshold {thr_all:.4f} | recall {rec_all:.4f}",
          flush=True)
    print(f"  band crossing: above {above_all} | below {below_all}", flush=True)
    print(f"pooled AUC (held-out pos vs random neg)  {_auc(pos_all, rand_all):.4f}",
          flush=True)
    # population std (ddof=0) to match _aggregate_fold_metrics in nlp.py
    auc_mean = float(np.mean(fold_aucs))
    auc_std = float(np.std(fold_aucs))
    print(f"per-fold AUC: {', '.join(f'{a:.4f}' for a in fold_aucs)}", flush=True)
    print(f"AUC mean ± std: {auc_mean:.4f} ± {auc_std:.4f}", flush=True)
    enc_mean = float(np.mean(encode_s_folds))
    enc_std = float(np.std(encode_s_folds))
    print(f"encode_s (fine-tuned, per fold): {', '.join(f'{s:.1f}' for s in encode_s_folds)} "
          f"| mean ± std: {enc_mean:.1f} ± {enc_std:.1f}s", flush=True)
    # pooled country strata (stratify the concatenated held-out positives)
    same_all, cross_all, unknown_all = split_pos_by_country(test_pos_all, country)
    n_unknown_all = int(unknown_all.sum())
    print("pooled country strata:", flush=True)
    for label, mask in (("in_country", same_all), ("cross_country", cross_all),
                        ("unknown_country", unknown_all)):
        n_pos = int(mask.sum())
        if n_pos == 0:
            sp, stp, sfp, sthr, srec = float("nan"), 0, 0, float("nan"), float("nan")
        else:
            sp, stp, sfp, sthr, srec = precision_at_recall_breakdown(pos_all[mask], hard_all, target_recall=0.90)
        print(f"  pooled {label}: n_pos={n_pos} P@90R={sp:.4f} TP={stp} FP={sfp} "
              f"thr={sthr:.4f} recall={srec:.4f}", flush=True)
        log_nlp_eval(
            {f"precision_at_90pct_recall_{label}": round(sp, 4),
             f"tp_at_90pct_recall_{label}": float(stp),
             f"fp_at_90pct_recall_{label}": float(sfp),
             f"threshold_at_90pct_recall_{label}": round(sthr, 4),
             f"{label}_n_pos": float(n_pos)},
            {**base_params, "fold": "pooled", "stratum": label},
            tracking_uri, experiment_name,
        )
    print(f"  pooled unknown: {n_unknown_all}", flush=True)
    pooled_metrics = {
        "precision_at_90pct_recall": round(p, 4),
        "tp_at_90pct_recall": float(tp_all),
        "fp_at_90pct_recall": float(fp_all),
        "threshold_at_90pct_recall": round(thr_all, 4),
        "recall_at_90pct": round(rec_all, 4),
        "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
        "hard_above_band": float(above_all), "hard_below_band": float(below_all),
        "auc": round(_auc(pos_all, rand_all), 4),
        "auc_mean": round(auc_mean, 4),
        "auc_std": round(auc_std, 4),
        "encode_s_mean": round(enc_mean, 3),
        "encode_s_std": round(enc_std, 3),
    }
    for i, a in enumerate(fold_aucs):
        pooled_metrics[f"fold_{i}_auc"] = round(a, 4)
    log_nlp_eval(pooled_metrics, {**base_params, "fold": "pooled"}, tracking_uri, experiment_name)
    four_df = pd.DataFrame(four_pop_rows, columns=["population", "cosine"])
    four_df.to_csv(RESULTS / "07b_four_pop_scores.csv", index=False)
    print(f"wrote {RESULTS / '07b_four_pop_scores.csv'} ({len(four_df):,} rows)", flush=True)
    print(f"TOTAL {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
