"""08: consolidated prep + encode + link pipeline — one re-runnable command.

Production path end-to-end, writing the deliverable `sku_to_item.csv`:
  1. prep   — 06_dedupe.main()      (dataset_deduped.csv + sku_to_rep.csv)
  2. encode — zero-shot bi-encoder   (cached: title | brand | category)
  3. link   — _link.resolve_items    (hard barcode + soft cosine + closure)
  4. map    — raw SKU -> ITEM_ID via sku_to_rep.csv
  5. write  — results/euromonitor/sku_to_item.csv
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from _common import DATA_PATH, RESULTS, load_dataset, load_dataset_deduped
from _link import resolve_items

from broadway.training.nlp import encode_corpus

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE = str(DATA_PATH.parent / "embeddings_cache")
SKU_TO_REP_PATH = DATA_PATH.with_name("sku_to_rep.csv")
OUT_PATH = RESULTS / "sku_to_item.csv"


def main() -> None:
    t0 = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)

    # 1. prep (tiered dedupe -> dataset_deduped.csv + sku_to_rep.csv)
    dedupe = importlib.import_module("06_dedupe")
    dedupe.main()

    # 2. encode the deduped representatives (cached; writes a fresh .npz on miss)
    reps = load_dataset_deduped()
    payload = (
        reps["title"].fillna("") + " | " + reps["brand"].fillna("") + " | " + reps["category"].fillna("")
    ).tolist()
    emb, encode_s = encode_corpus(MODEL, payload, batch_size=256, max_seq_length=128, cache_dir=CACHE)
    print(f"embeddings: {emb.shape}  (encode_s={encode_s:.0f})", flush=True)

    # 3. link
    res = resolve_items(reps, emb)
    rep_item = res["rep_item"]

    # 4. map every raw SKU to its ITEM_ID via 06's representative
    sku_to_rep = pd.read_csv(SKU_TO_REP_PATH, dtype=str)
    raw = load_dataset()
    rep_of = dict(zip(sku_to_rep["product_id"], sku_to_rep["rep_id"].astype(int)))
    raw["ITEM_ID"] = [rep_item[rep_of[pid]] for pid in raw["product_id"]]
    out = raw[["product_id", "ITEM_ID"]].rename(columns={"product_id": "sku_id", "ITEM_ID": "item_id"})
    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}  ({len(out):,} SKUs -> {out["item_id"].nunique():,} ITEMs)", flush=True)

    # 5. headline
    n_sku = len(out)
    n_item = int(out["item_id"].nunique())
    item_ids = np.sort(out["item_id"].unique())
    contiguous = bool((item_ids == np.arange(n_item)).all())
    max_cluster = int(np.bincount(rep_item).max())
    print(f"HEADLINE: {n_sku:,} SKUs -> {n_item:,} ITEMs  |  contiguous 0..{n_item - 1}: {contiguous}  |  max cluster: {max_cluster:,}", flush=True)
    print(f"barcode edges: {res['n_barcode']:,}  |  soft candidates: {res['n_soft_candidates']:,}  |  "
          f"admitted: {res['n_soft_admitted']:,}  (blocked: {res['n_soft_blocked']:,})  |  "
          f"total edges: {res['n_edges']:,}", flush=True)
    print(f"pipeline total {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
