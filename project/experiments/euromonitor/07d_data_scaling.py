"""07d: Data-scaling — how much fine-tune data before the model breaks?

Halves the TripletLoss training triples each round (full, 1/2, 1/4, 1/8, 1/16)
on the SAME eval pairs, to locate the data cliff where matching degrades.
Reuses the barcode-level split + hard-negative miner; each fraction fine-tunes
one epoch and re-scores the held-out test split against the hard negatives
(encoding only the rows the eval pairs touch). Writes a learning-curve CSV.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from _blocking import build_pairs
from _common import PATHS, SEED, load_dataset_deduped
from _hard_negatives import mine_hard_negatives

from broadway.training.nlp import encode_corpus, entity_resolution_metrics

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOSS = "triplet"
CACHE = str(PATHS.experiments.parent / "data" / "euromonitor" / "embeddings_cache")
RESULTS = PATHS.experiments / "results" / "euromonitor"
EPOCHS = 1
BATCH_SIZE = 32
LR = 2e-5
FRACTIONS = (1.0, 0.5, 0.25, 0.125, 0.0625)


def main() -> None:
    t0 = time.perf_counter()
    df = load_dataset_deduped()
    payload = (df["title"].fillna("") + " | " + df["brand"].fillna("") + " | " + df["category"].fillna("")).tolist()
    barcodes = df["barcode"].fillna("").astype(str)
    row_bc = barcodes.to_numpy()

    # barcode-level 70/15/15 split (no same-product leakage)
    known = df[barcodes.str.len() > 0]
    multi = known[known.groupby("barcode")["retailer"].transform("nunique") > 1]
    bcs = np.array(sorted(multi["barcode"].unique()))
    perm = np.random.default_rng(SEED).permutation(len(bcs))
    tr, va = int(0.70 * len(bcs)), int(0.85 * len(bcs))
    train_bc = set(bcs[perm[:tr]])
    test_bc = set(bcs[perm[va:]])

    pos, _ = build_pairs(df, SEED, 4, 10_000)
    train_mask = np.isin(row_bc[pos[:, 0]], list(train_bc)) & np.isin(row_bc[pos[:, 1]], list(train_bc))
    test_mask = np.isin(row_bc[pos[:, 0]], list(test_bc)) & np.isin(row_bc[pos[:, 1]], list(test_bc))
    train_pos = pos[train_mask]
    test_pos = pos[test_mask]
    print(f"train positives {len(train_pos):,} | test positives {len(test_pos):,}", flush=True)

    # zero-shot embeddings for hard-negative mining
    emb0, _ = encode_corpus(MODEL, payload, batch_size=256, max_seq_length=128, cache_dir=CACHE)

    # mine hard negatives; train-anchor ones build the triple negatives,
    # test-anchor ones are the held-out hard eval set.
    hard_pairs, _ = mine_hard_negatives(df, emb0, n_target=10_000)
    hn_train = hard_pairs[np.isin(row_bc[hard_pairs[:, 0]], list(train_bc))]
    hn_test = hard_pairs[np.isin(row_bc[hard_pairs[:, 0]], list(test_bc))]

    hn_map: dict[int, list[int]] = defaultdict(list)
    for a, b in hn_train:
        hn_map[int(a)].append(int(b))

    from sentence_transformers import InputExample
    rng = np.random.default_rng(SEED)
    triples: list[InputExample] = []
    for a, b in train_pos:
        partners = hn_map.get(int(a)) or hn_map.get(int(b))
        if not partners:
            continue
        c = int(partners[rng.integers(len(partners))])
        triples.append(InputExample(texts=[payload[a], payload[b], payload[c]]))
    print(f"full triple set: {len(triples):,} | hard eval negatives (test anchors): {len(hn_test):,}", flush=True)

    # fixed eval set: test positives vs hard negatives; encode only their rows
    eval_rows = np.unique(np.r_[test_pos.ravel(), hn_test.ravel()])
    row_to_idx = {int(r): i for i, r in enumerate(eval_rows)}
    test_idx = np.vectorize(lambda r: row_to_idx[int(r)])(test_pos)
    hn_idx = np.vectorize(lambda r: row_to_idx[int(r)])(hn_test)
    eval_payload = [payload[r] for r in eval_rows]
    print(f"unique eval rows: {len(eval_rows):,}", flush=True)

    from sentence_transformers import SentenceTransformer

    from broadway.training.nlp import _finetune

    records = []
    for frac in FRACTIONS:
        n = max(1, int(len(triples) * frac))
        subset = list(triples) if frac == 1.0 else rng.choice(triples, n, replace=False).tolist()
        model = SentenceTransformer(MODEL, device="cpu")
        model.max_seq_length = 128
        _finetune(model, {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LR, "warmup_steps": 0},
                  subset, 128, loss=LOSS)
        emb = model.encode(eval_payload, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        pos_s = (emb[test_idx[:, 0]] * emb[test_idx[:, 1]]).sum(axis=1)
        neg_s = (emb[hn_idx[:, 0]] * emb[hn_idx[:, 1]]).sum(axis=1)
        m = entity_resolution_metrics(pos_s, neg_s)
        records.append({"fraction": frac, "n_triples": n, **m})
        print(f"frac={frac:<6} n={n:<5} AUC={m['auc']:.4f} AP={m['average_precision']:.4f} "
              f"P@90R={m['precision_at_90pct_recall']:.4f} F1={m['f1_at_5pct_fpr']:.4f}", flush=True)

    out = pd.DataFrame(records)
    out.to_csv(RESULTS / "07d_data_scaling.csv", index=False)
    print(f"\nwrote {RESULTS / '07d_data_scaling.csv'}", flush=True)
    print(f"TOTAL {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
