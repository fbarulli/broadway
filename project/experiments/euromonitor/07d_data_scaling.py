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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from _blocking import build_pairs
from _common import PATHS, RESULTS, SEED, load_dataset_deduped
from _hard_negatives import build_triplets, mine_hard_negatives, pairs_in_set, split_barcodes

from broadway.training.nlp import _cosine, encode_corpus, entity_resolution_metrics

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOSS = "triplet"
CACHE = str(PATHS.experiments.parent / "data" / "euromonitor" / "embeddings_cache")
EPOCHS = 1
BATCH_SIZE = 32
LR = 2e-5
FRACTIONS = (1.0, 0.5, 0.25, 0.125, 0.0625)


def main() -> None:
    t0 = time.perf_counter()
    df = load_dataset_deduped()
    payload = (df["title"].fillna("") + " | " + df["brand"].fillna("") + " | " + df["category"].fillna("")).tolist()
    row_bc = df["barcode"].fillna("").astype(str).to_numpy()

    # barcode-level 70/15/15 split (no same-product leakage); the data-scaling
    # curve only needs train/test (val is a 07b concern), both held out cleanly.
    split_bc = split_barcodes(df, SEED)
    train_bc, test_bc = split_bc["train"], split_bc["test"]

    pos, _ = build_pairs(df, SEED, 4, 10_000)
    train_pos = pos[pairs_in_set(pos, row_bc, train_bc)]
    test_pos = pos[pairs_in_set(pos, row_bc, test_bc)]
    print(f"train positives {len(train_pos):,} | test positives {len(test_pos):,}", flush=True)

    # zero-shot embeddings for hard-negative mining
    emb0, _ = encode_corpus(MODEL, payload, batch_size=256, max_seq_length=128, cache_dir=CACHE)

    # mine hard negatives; both endpoints of every pair are held in the SAME
    # split (train pairs build triples, test pairs are the held-out hard eval).
    hard_pairs, _ = mine_hard_negatives(df, emb0, n_target=20_000, cosine_lo=0.45, cosine_hi=0.80)
    hard_train = hard_pairs[pairs_in_set(hard_pairs, row_bc, train_bc)]
    hard_test = hard_pairs[pairs_in_set(hard_pairs, row_bc, test_bc)]

    # full (uncapped) triple set — the learning curve halves this per fraction.
    triples = build_triplets(train_pos, hard_train, payload, seed=SEED, max_triples=len(train_pos))
    if not triples:
        raise RuntimeError("no training triples built (empty train hard-negative set)")
    print(f"full triple set: {len(triples):,} | hard eval negatives (test): {len(hard_test):,}",
          flush=True)

    # fixed eval set: test positives vs hard negatives; encode only their rows
    eval_rows = np.unique(np.r_[test_pos.ravel(), hard_test.ravel()])
    row_to_idx = {int(r): i for i, r in enumerate(eval_rows)}
    test_idx = np.vectorize(lambda r: row_to_idx[int(r)])(test_pos)
    hn_idx = np.vectorize(lambda r: row_to_idx[int(r)])(hard_test)
    eval_payload = [payload[r] for r in eval_rows]
    print(f"unique eval rows: {len(eval_rows):,}", flush=True)

    if len(test_pos) == 0 or len(hard_test) == 0:
        raise RuntimeError(
            f"held-out eval set empty: test_pos={len(test_pos)}, hard_test={len(hard_test)}")

    from sentence_transformers import SentenceTransformer

    from broadway.training.nlp import _finetune

    rng = np.random.default_rng(SEED)
    records = []
    for frac in FRACTIONS:
        n = max(1, int(len(triples) * frac))
        subset = list(triples) if frac == 1.0 else rng.choice(triples, n, replace=False).tolist()
        model = SentenceTransformer(MODEL, device="cpu")
        model.max_seq_length = 128
        _finetune(model, {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LR, "warmup_steps": 0},
                  subset, 128, loss=LOSS)
        emb = model.encode(eval_payload, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        pos_s = _cosine(emb, test_idx)
        neg_s = _cosine(emb, hn_idx)
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
