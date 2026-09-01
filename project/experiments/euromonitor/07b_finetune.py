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
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from _blocking import build_pairs
from _common import PATHS, SEED, load_dataset_deduped
from _hard_negatives import mine_hard_negatives
from broadway.training.nlp import encode_corpus, precision_at_recall

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOSS = "triplet"
CACHE = str(PATHS.experiments.parent / "data" / "euromonitor" / "embeddings_cache")
RESULTS = PATHS.experiments / "results" / "euromonitor"
EPOCHS = 1
BATCH_SIZE = 32
LR = 2e-5
MAX_TRIPLES = 5_000


def main() -> None:
    t0 = time.perf_counter()
    df = load_dataset_deduped()
    payload = (df["title"].fillna("") + " | " + df["brand"].fillna("") + " | " + df["category"].fillna("")).tolist()
    barcodes = df["barcode"].fillna("").astype(str)
    row_bc = barcodes.to_numpy()

    # ---- barcode-level 70/15/15 split (no same-product leakage) ----
    known = df[barcodes.str.len() > 0]
    multi = known[known.groupby("barcode")["retailer"].transform("nunique") > 1]
    bcs = np.array(sorted(multi["barcode"].unique()))
    perm = np.random.default_rng(SEED).permutation(len(bcs))
    tr, va = int(0.70 * len(bcs)), int(0.85 * len(bcs))
    split_bc = {
        "train": set(bcs[perm[:tr]]),
        "val": set(bcs[perm[tr:va]]),
        "test": set(bcs[perm[va:]]),
    }
    print(f"entities: train {len(split_bc['train']):,} | val {len(split_bc['val']):,} | test {len(split_bc['test']):,}", flush=True)

    pos, neg = build_pairs(df, SEED, 4, 10_000)
    train_mask = np.isin(row_bc[pos[:, 0]], list(split_bc["train"])) & np.isin(row_bc[pos[:, 1]], list(split_bc["train"]))
    test_mask = np.isin(row_bc[pos[:, 0]], list(split_bc["test"])) & np.isin(row_bc[pos[:, 1]], list(split_bc["test"]))
    train_pos = pos[train_mask]
    test_pos = pos[test_mask]
    print(f"train positives {len(train_pos):,} | test positives {len(test_pos):,}", flush=True)

    # ---- zero-shot embeddings (cached) for mining + baseline ----
    emb0, _ = encode_corpus(MODEL, payload, batch_size=256, max_seq_length=128, cache_dir=CACHE)

    # ---- mine hard negatives, then keep only train-barcode anchors ----
    hard_pairs, hard_cos = mine_hard_negatives(df, emb0, n_target=10_000)
    hn_train_mask = np.isin(row_bc[hard_pairs[:, 0]], list(split_bc["train"]))
    hard_pairs = hard_pairs[hn_train_mask]
    hard_cos = hard_cos[hn_train_mask]
    print(f"hard negatives (train anchors) {len(hard_pairs):,}", flush=True)

    # anchor -> list of hard-negative partners
    hn_map: dict[int, list[int]] = defaultdict(list)
    for (a, b), s in zip(hard_pairs, hard_cos):
        hn_map[int(a)].append(int(b))

    # ---- build triples: (anchor, positive, hard-negative) ----
    from sentence_transformers import InputExample
    rng = np.random.default_rng(SEED)
    triples: list[InputExample] = []
    for a, b in train_pos:
        partners = hn_map.get(int(a)) or hn_map.get(int(b))
        if not partners:
            continue
        c = int(partners[rng.integers(len(partners))])
        triples.append(InputExample(texts=[payload[a], payload[b], payload[c]]))
        if len(triples) >= MAX_TRIPLES:
            break
    print(f"training triples {len(triples):,} (loss={LOSS})", flush=True)

    # ---- fine-tune ----
    from broadway.training.nlp import _finetune
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL, device="cpu")
    model.max_seq_length = 128
    params = {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LR, "warmup_steps": 0}
    _finetune(model, params, triples, 128, loss=LOSS)
    print(f"fine-tuned in {time.perf_counter()-t0:.0f}s", flush=True)

    # ---- encode full corpus with fine-tuned weights, score test pairs ----
    emb = model.encode(payload, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    def cos(pairs):
        return (emb[pairs[:, 0]] * emb[pairs[:, 1]]).sum(axis=1)
    test_neg = neg  # random negatives for AUC; hard band scored separately
    pos_s = cos(test_pos)
    neg_s = cos(test_neg)
    auc = roc_auc_score(np.r_[np.ones(len(pos_s)), np.zeros(len(neg_s))], np.r_[pos_s, neg_s])
    print(f"fine-tuned AUC (test pos vs random neg): {auc:.4f}", flush=True)

    # hard-band precision@recall (test pos vs hard negatives)
    hn_test_mask = np.isin(row_bc[hard_pairs[:, 0]], list(split_bc["test"])) | np.isin(row_bc[hard_pairs[:, 1]], list(split_bc["test"]))
    htp = hard_pairs[hn_test_mask]
    hn_s = cos(htp) if len(htp) else np.array([])
    y = np.r_[np.ones(len(pos_s)), np.zeros(len(hn_s))]
    scores = np.r_[pos_s, hn_s]
    p_at_r = precision_at_recall(y, scores, target_recall=0.99)
    print(f"hard-band pairs scored: {len(hn_s):,}", flush=True)
    print(f"precision@recall=0.99 on hard band: {p_at_r:.4f}", flush=True)
    print(f"TOTAL {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
