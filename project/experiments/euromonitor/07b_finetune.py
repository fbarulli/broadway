"""07b: TripletLoss fine-tune on mined hard negatives (round 2, K-fold CV).

Trains a bi-encoder to separate same-product positives from the hard negatives
the zero-shot champion confuses (different brand, same macro category, widened
mid-cosine band 0.45-0.80). Evaluation is barcode-level K-fold CV: each fold
fine-tunes one epoch on the other K-1 folds and scores the held-out fold, then
the held-out hard-negative scores are POOLED and reported with a bootstrap 95%
CI on precision@90%recall — not a bare point estimate over a ~100-pair slice.

Why three fixes matter (round 1 reported precision@90%recall = 1.0000 on 108
held-out hard negatives):
  1. widened confusion band (0.45-0.80 vs 0.5-0.75) -> more test volume,
  2. bootstrap CI -> the precision@90%recall is a sample statistic, not a
     constant, so report its spread instead of a naked point estimate,
  3. group-aware K-fold pooling -> ~5x the eval volume and no single lucky fold.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")

import numpy as np
from _blocking import build_pairs
from _common import PATHS, SEED, load_dataset_deduped
from _hard_negatives import build_triplets, mine_hard_negatives, pairs_in_set
from sklearn.metrics import roc_auc_score

from broadway.training.nlp import _cosine, encode_corpus, precision_at_recall

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


def _precision_ci(
    pos_s: np.ndarray,
    neg_s: np.ndarray,
    target_recall: float = 0.90,
    n_boot: int = BOOTSTRAP,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """precision@target-recall point estimate + bootstrap 95% CI (resample both sides)."""
    point = float(precision_at_recall(pos_s, neg_s, target_recall=target_recall))
    if not np.isfinite(point):
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(neg_s), len(neg_s))
        j = rng.integers(0, len(pos_s), len(pos_s))
        boots[b] = precision_at_recall(pos_s[j], neg_s[i], target_recall=target_recall)
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _precision_breakdown(
    pos_s: np.ndarray,
    neg_s: np.ndarray,
    target_recall: float = 0.90,
) -> tuple[float, int, int, float, float]:
    """(precision, TP, FP, threshold, recall) at the target-recall operating point.

    Mirrors ``precision_at_recall`` (same 10th-percentile-positive threshold) but
    also returns the raw confusion counts, so the per-fold result is auditable
    instead of deriving FP back out of a 4-decimal rounded precision.
    """
    if len(pos_s) == 0 or len(neg_s) == 0:
        return float("nan"), 0, 0, float("nan"), float("nan")
    threshold = float(np.quantile(pos_s, 1 - target_recall))
    tp = int((pos_s >= threshold).sum())
    fp = int((neg_s >= threshold).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / len(pos_s)
    return precision, tp, fp, threshold, recall


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

    folds = kfold_barcodes(df, CV_FOLDS, SEED)
    print(f"barcode K-fold: {CV_FOLDS} folds "
          f"({', '.join(str(len(f)) for f in folds)} entities)", flush=True)

    pos, neg = build_pairs(df, SEED, 4, 10_000)

    # ---- zero-shot embeddings (cached) for mining, encoded ONCE ----
    emb0, _ = encode_corpus(MODEL, payload, batch_size=256, max_seq_length=128, cache_dir=CACHE)

    # ---- mine hard negatives ONCE over the full corpus (widened band), then
    # each fold keeps only pairs whose BOTH endpoints live in that fold ----
    hard_pairs, _ = mine_hard_negatives(
        df, emb0, n_target=N_TARGET, cosine_lo=COSINE_LO, cosine_hi=COSINE_HI)
    print(f"hard negatives mined (band {COSINE_LO}-{COSINE_HI}, target {N_TARGET:,}): "
          f"{len(hard_pairs):,}", flush=True)

    from sentence_transformers import SentenceTransformer

    from broadway.training.nlp import _finetune

    pooled_pos: list[np.ndarray] = []
    pooled_hard: list[np.ndarray] = []
    pooled_rand: list[np.ndarray] = []
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

        emb = model.encode(payload, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        test_pos = pos[pairs_in_set(pos, row_bc, test_bc)]
        hard_test = hard_pairs[pairs_in_set(hard_pairs, row_bc, test_bc)]
        rand_test = neg[pairs_in_set(neg, row_bc, test_bc)]
        pos_s = _cosine(emb, test_pos)
        hard_s = _cosine(emb, hard_test)
        p, lo, hi = _precision_ci(pos_s, hard_s)
        _, tp, fp, thr, rec = _precision_breakdown(pos_s, hard_s)
        pooled_pos.append(pos_s)
        pooled_hard.append(hard_s)
        pooled_rand.append(_cosine(emb, rand_test))
        print(f"fold {f}: test_pos {len(test_pos):,} | hard_test {len(hard_test):,} "
              f"| P@90R {p:.4f} [{lo:.4f}, {hi:.4f}] "
              f"| TP {tp} FP {fp} thr {thr:.4f} recall {rec:.4f} "
              f"| hard med {np.median(hard_s):.3f} p90 {np.quantile(hard_s, 0.9):.3f} max {hard_s.max():.3f}",
              flush=True)

    pos_all = np.concatenate(pooled_pos)
    hard_all = np.concatenate(pooled_hard)
    rand_all = np.concatenate(pooled_rand)
    p, lo, hi = _precision_ci(pos_all, hard_all)
    _, tp_all, fp_all, thr_all, rec_all = _precision_breakdown(pos_all, hard_all)
    print(f"\npooled across {len(pooled_pos)} folds: test_pos {len(pos_all):,} "
          f"| hard_test {len(hard_all):,}", flush=True)
    print(f"precision@90pct-recall (hard band)  {p:.4f}  [95% CI {lo:.4f}, {hi:.4f}]",
          flush=True)
    print(f"  TP {tp_all} | FP {fp_all} | threshold {thr_all:.4f} | recall {rec_all:.4f}",
          flush=True)
    print(f"pooled AUC (held-out pos vs random neg)  {_auc(pos_all, rand_all):.4f}",
          flush=True)
    print(f"TOTAL {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
