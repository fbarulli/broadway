"""07c: Field ablation — how much does each payload field contribute?

Runs the SAME encoder and the SAME eval pairs with each field knocked out one at
a time (title / brand / category), plus single-field payloads, to measure each
field's marginal contribution to matching. Encodes only the rows that appear in
the eval pairs (the full-corpus encode is unnecessary for scoring), so it stays
CPU-cheap. Reports AUC, PR-AUC, precision@90%recall, and F1 per variant.

Also runs the SAME field variants against the cross-country silver-label proxy
pairs (same brand+category, different country — see 01h) to check whether the
field a model relies on most cross-country is the same field it relies on most
in general. This is the direct test of whether the "translation tax" traces to
a specific field (e.g. title, which is the least stable field across markets).
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
N_CROSS_PAIRS = 3000  # cap, matches 01h's cross-country proxy construction


def build_cross_country_pairs(df: pd.DataFrame) -> np.ndarray:
    """Same brand+category, different country, different barcode — silver proxy from 01h."""
    cross_pairs = []
    for (_b, _c), g in df[df["brand"].fillna("") != ""].groupby(["brand", "category"], sort=False):
        by_country = {ct: grp.index[0] for ct, grp in g.groupby("country")}
        if len(by_country) < 2:
            continue
        cts = list(by_country)
        a = by_country[cts[0]]
        bb = by_country[cts[1]]
        cross_pairs.append((a, bb))
        if len(cross_pairs) >= N_CROSS_PAIRS:
            break
    return np.array(cross_pairs)


def main() -> None:
    t0 = time.perf_counter()
    df = load_dataset_deduped().reset_index(drop=True)
    pos, neg = build_pairs(df, SEED, 4, 10_000)
    cross_pairs = build_cross_country_pairs(df)
    print(f"eval pairs: {len(pos):,} pos / {len(neg):,} neg", flush=True)
    print(f"cross-country proxy pairs: {len(cross_pairs):,}", flush=True)

    # only the rows any pair set touches need embedding
    rows = np.unique(np.r_[pos.ravel(), neg.ravel(), cross_pairs.ravel()])
    idx_map = {int(r): i for i, r in enumerate(rows)}
    print(f"unique rows to encode: {len(rows):,}", flush=True)

    title = df["title"].fillna("").astype(str)
    brand = df["brand"].fillna("").astype(str)
    category = df["category"].fillna("").astype(str)

    variants = {
        "title|brand|category": title + SEP + brand + SEP + category,
        "title|brand":          title + SEP + brand,
        "title|category":       title + SEP + category,
        "brand|category":       brand + SEP + category,
        "title":                title,
        "brand":                brand,
        "category":             category,
    }

    pos_idx = np.vectorize(idx_map.__getitem__)(pos)
    neg_idx = np.vectorize(idx_map.__getitem__)(neg)
    cross_idx = np.vectorize(idx_map.__getitem__)(cross_pairs)

    records = []
    for name, series in variants.items():
        payload = series.iloc[rows].tolist()
        emb, encode_s = encode_corpus(MODEL, payload, batch_size=256, max_seq_length=128, cache_dir=CACHE)

        pos_s = _cosine(emb, pos_idx)
        neg_s = _cosine(emb, neg_idx)
        m = entity_resolution_metrics(pos_s, neg_s)

        cross_s = _cosine(emb, cross_idx)
        cross_median = float(np.median(cross_s))
        cross_ge_055 = float((cross_s >= 0.55).mean())
        # gap vs this variant's own in-country positive median — the
        # per-field "translation tax", not just the full-payload one
        translation_tax = float(np.median(pos_s) - cross_median)

        records.append({
            "variant": name,
            "encode_s": round(encode_s, 1),
            **m,
            "cross_country_median": round(cross_median, 4),
            "cross_country_pct_ge_055": round(cross_ge_055, 4),
            "translation_tax": round(translation_tax, 4),
        })
        print(
            f"{name:<22} AUC={m['auc']:.4f}  AP={m['average_precision']:.4f}  "
            f"P@90R={m['precision_at_90pct_recall']:.4f}  F1@5%FPR={m['f1_at_5pct_fpr']:.4f}  "
            f"TP@90R={m['tp_at_90pct_recall']:.0f} FP@90R={m['fp_at_90pct_recall']:.0f} "
            f"thr={m['threshold_at_90pct_recall']:.4f}  |  "
            f"cross_median={cross_median:.4f}  cross>=0.55={cross_ge_055:.1%}  "
            f"tax={translation_tax:+.4f}",
            flush=True,
        )

    out = pd.DataFrame(records)
    out.to_csv(RESULTS / "07c_field_ablation.csv", index=False)
    print(f"\nwrote {RESULTS / '07c_field_ablation.csv'}", flush=True)

    # which field drives cross-country matching most: the variant whose removal
    # increases translation_tax the most (from the full-payload baseline) is
    # the field the model is leaning on that's least stable across borders
    baseline_tax = out.loc[out["variant"] == "title|brand|category", "translation_tax"].iloc[0]
    print("\ntranslation tax by variant (full-payload baseline = "
          f"{baseline_tax:+.4f}):")
    print(out[["variant", "translation_tax"]].sort_values("translation_tax").to_string(index=False))

    print(f"\nTOTAL {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()