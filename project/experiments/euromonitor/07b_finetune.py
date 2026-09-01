"""07b: TripletLoss fine-tune on mined hard negatives (round 1).

Trains a bi-encoder to separate same-product positives from the hard negatives
the zero-shot champion confuses (different brand, same macro category, mid
cosine). Mines hard negatives with the conflicting-barcode exclusion, builds
anchor/positive/hard-negative triples on a barcode-level train split, fine-tunes
one epoch with TripletLoss, then evaluates on the held-out test split — reporting
AUC AND precision@target-recall on the hard band (not just random negatives).
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
from _hard_negatives import build_triplets, mine_hard_negatives, pairs_in_set, split_barcodes
from sklearn.metrics import roc_auc_score

from broadway.training.nlp import _cosine, encode_corpus, precision_at_recall

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOSS = "triplet"
CACHE = str(PATHS.experiments.parent / "data" / "euromonitor" / "embeddings_cache")
EPOCHS = 1
BATCH_SIZE = 32
LR = 2e-5
MAX_TRIPLES = 5_000


def _auc(pos_s: np.ndarray, neg_s: np.ndarray) -> float:
    """ROC AUC of a held-out pos/neg pair score population (NaN if a side is empty)."""
    if len(pos_s) == 0 or len(neg_s) == 0:
        return float("nan")
    return float(roc_auc_score(
        np.r_[np.ones(len(pos_s)), np.zeros(len(neg_s))],
        np.r_[pos_s, neg_s]))


def main() -> None:
    t0 = time.perf_counter()
    df = load_dataset_deduped()
    payload = (df["title"].fillna("") + " | " + df["brand"].fillna("") + " | " + df["category"].fillna("")).tolist()
    row_bc = df["barcode"].fillna("").astype(str).to_numpy()

    # ---- barcode-level 70/15/15 split (no same-product leakage) ----
    split_bc = split_barcodes(df, SEED)
    train_bc, val_bc, test_bc = split_bc["train"], split_bc["val"], split_bc["test"]
    print(f"entities: train {len(train_bc):,} | val {len(val_bc):,} | test {len(test_bc):,}",
          flush=True)

    pos, neg = build_pairs(df, SEED, 4, 10_000)
    train_pos = pos[pairs_in_set(pos, row_bc, train_bc)]
    val_pos = pos[pairs_in_set(pos, row_bc, val_bc)]
    test_pos = pos[pairs_in_set(pos, row_bc, test_bc)]
    # Held-out negatives: both endpoints must live in the SAME split, so the
    # eval population shares no entity with training.
    val_neg = neg[pairs_in_set(neg, row_bc, val_bc)]
    test_neg = neg[pairs_in_set(neg, row_bc, test_bc)]
    print(f"positives  train {len(train_pos):,} | val {len(val_pos):,} | test {len(test_pos):,}",
          flush=True)
    print(f"held-out negatives  val {len(val_neg):,} | test {len(test_neg):,}", flush=True)

    # ---- zero-shot embeddings (cached) for mining + baseline ----
    emb0, _ = encode_corpus(MODEL, payload, batch_size=256, max_seq_length=128, cache_dir=CACHE)

    # ---- mine hard negatives, then split so BOTH endpoints are held in the
    # same split (no train entity leaks into val/test, and vice versa) ----
    hard_pairs, _ = mine_hard_negatives(df, emb0, n_target=10_000)
    hard_train = hard_pairs[pairs_in_set(hard_pairs, row_bc, train_bc)]
    hard_val = hard_pairs[pairs_in_set(hard_pairs, row_bc, val_bc)]
    hard_test = hard_pairs[pairs_in_set(hard_pairs, row_bc, test_bc)]
    print(f"hard negatives  train {len(hard_train):,} | val {len(hard_val):,} | "
          f"test {len(hard_test):,}", flush=True)

    triples = build_triplets(train_pos, hard_train, payload, seed=SEED, max_triples=MAX_TRIPLES)
    if not triples:
        raise RuntimeError("no training triples built (empty train hard-negative set)")
    print(f"training triples {len(triples):,} (loss={LOSS})", flush=True)

    # ---- fine-tune ----
    from sentence_transformers import SentenceTransformer

    from broadway.training.nlp import _finetune
    model = SentenceTransformer(MODEL, device="cpu")
    model.max_seq_length = 128
    params = {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LR, "warmup_steps": 0}
    _finetune(model, params, triples, 128, loss=LOSS)
    print(f"fine-tuned in {time.perf_counter()-t0:.0f}s", flush=True)

    # ---- encode full corpus with fine-tuned weights, score the held-out sets ----
    emb = model.encode(payload, batch_size=256, normalize_embeddings=True, show_progress_bar=False)

    val_auc = _auc(_cosine(emb, val_pos), _cosine(emb, val_neg))
    test_auc = _auc(_cosine(emb, test_pos), _cosine(emb, test_neg))
    print(f"val AUC {val_auc:.4f} | test AUC {test_auc:.4f} (held-out pos vs random neg)",
          flush=True)

    val_p = precision_at_recall(_cosine(emb, val_pos), _cosine(emb, hard_val), target_recall=0.90)
    test_p = precision_at_recall(_cosine(emb, test_pos), _cosine(emb, hard_test), target_recall=0.90)
    print(f"precision@90pct-recall on hard band  val {val_p:.4f} | test {test_p:.4f}",
          flush=True)
    print(f"TOTAL {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
