"""07c: Field ablation — how much does each payload field contribute?

Runs the SAME encoder and the SAME eval pairs with each field knocked out one at
a time (title / brand / category), plus single-field payloads, to measure each
field's marginal contribution to matching. Encodes only the rows that appear in
the eval pairs (the full-corpus encode is unnecessary for scoring), so it stays
CPU-cheap. Reports AUC, PR-AUC, precision@90%recall, and F1 per variant.
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

from broadway.training.nlp import _cosine, encode_corpus, entity_resolution_metrics

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE = str(PATHS.experiments.parent / "data" / "euromonitor" / "embeddings_cache")
SEP = " | "


def main() -> None:
    t0 = time.perf_counter()
    df = load_dataset_deduped()
    pos, neg = build_pairs(df, SEED, 4, 10_000)
    print(f"eval pairs: {len(pos):,} pos / {len(neg):,} neg", flush=True)

    # only the rows the eval pairs touch need embedding
    rows = np.unique(np.r_[pos.ravel(), neg.ravel()])
    idx_map = {int(r): i for i, r in enumerate(rows)}
    print(f"unique rows to encode: {len(rows):,}", flush=True)

    title = df["title"].fillna("").astype(str)
    brand = df["brand"].fillna("").astype(str)
    category = df["category"].fillna("").astype(str)

    variants = {
        "title|brand|category": title + SEP + brand + SEP + category,
        "title|brand":         title + SEP + brand,
        "title|category":      title + SEP + category,
        "brand|category":      brand + SEP + category,
        "title":               title,
        "brand":               brand,
        "category":            category,
    }

    pos_idx = np.vectorize(idx_map.__getitem__)(pos)
    neg_idx = np.vectorize(idx_map.__getitem__)(neg)

    records = []
    for name, series in variants.items():
        payload = series.iloc[rows].tolist()
        emb, encode_s = encode_corpus(MODEL, payload, batch_size=256, max_seq_length=128, cache_dir=CACHE)
        pos_s = _cosine(emb, pos_idx)
        neg_s = _cosine(emb, neg_idx)
        m = entity_resolution_metrics(pos_s, neg_s)
        records.append({"variant": name, "encode_s": round(encode_s, 1), **m})
        print(f"{name:<22} AUC={m['auc']:.4f}  AP={m['average_precision']:.4f}  "
              f"P@90R={m['precision_at_90pct_recall']:.4f}  F1@5%FPR={m['f1_at_5pct_fpr']:.4f}", flush=True)

    out = pd.DataFrame(records)
    out.to_csv(RESULTS / "07c_field_ablation.csv", index=False)
    print(f"\nwrote {RESULTS / '07c_field_ablation.csv'}", flush=True)
    print(f"TOTAL {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
