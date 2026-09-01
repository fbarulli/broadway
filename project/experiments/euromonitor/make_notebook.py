"""Build the entity-resolution deliverable notebook (SKU_ID -> ITEM_ID)."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "entity_resolution.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


cells = [
    md("""# Euromonitor Entity Resolution — SKU → ITEM grouping

**Goal.** Given ~72k product listings (SKUs) scraped from 280 retailers across 19 countries, assign each SKU an **ITEM_ID** so that every listing of the *same physical product* shares one ITEM_ID. This is **entity resolution / record linkage**.

**Output.** A two-column table `SKU_ID → ITEM_ID` (written to `results/euromonitor/sku_to_item.csv`).
"""),

    md("""## 1. The method and why

The naive approach — compare every SKU to every other SKU — is ~2.6 **billion** pairs, too slow and far too noisy. A production matcher needs a **block → score → link** pipeline:

1. **Block** on `brand × macro-category` to cut candidates ~1000× (recall 0.957, ~1.8M pairs) — blocking is *recall-first*.
2. **Score** each blocked pair with a **bi-encoder** (sentence-transformers `all-MiniLM-L6-v2`) cosine similarity. It beats TF-IDF (AUC 0.9993 vs 0.983) because it is *semantic*: `"Coca-Cola 500ml"` and `"Coca Cola 0.5 L"` are seen as the same product despite different words.
3. **Link** with two rules:
   - **hard link** — identical non-empty **barcode** (GTIN) ⇒ same ITEM (ground-truth identity),
   - **soft link** — same brand + same volume + cosine ≥ 0.85 ⇒ same ITEM (the tighter constraint that stops cross-variant chain-merging).
4. **Transitive closure** (connected components) turns pairwise links into ITEM clusters, so A≈B and B≈C ⇒ A,B,C are one ITEM.
5. **Dedupe first** — within-retailer marketplace listings (one title listed 143× on Gittigidiyor) were collapsed to a representative by `06_dedupe.py` *before* matching; this notebook consumes that deduped output.
"""),

    md("""## 2. Setup"""),

    code("""import itertools
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

ROOT = Path.cwd()
SERIES = ROOT / "project" / "experiments" / "euromonitor"
sys.path.insert(0, str(SERIES))

from _common import PATHS, SEED, load_dataset, load_dataset_deduped
from _text import MACRO_MAP, extract_volume_ml

from broadway.training.nlp import _cosine, encode_corpus, entity_resolution_metrics

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE = str(PATHS.experiments.parent / "data" / "euromonitor" / "embeddings_cache")
DATA_DIR = PATHS.experiments.parent / "data" / "euromonitor"
SKU_TO_REP = DATA_DIR / "sku_to_rep.csv"   # written by 06_dedupe.py
print("setup ok")"""),

    md("""## 3. Exploratory data analysis"""),

    code("""raw = load_dataset()   # 71,623 SKUs, canonical column names
print(f"SKUs: {len(raw):,}  |  columns: {list(raw.columns)}")
raw.head(3)"""),

    code("""barcode = raw["barcode"].fillna("").astype(str)
vol = raw["title"].fillna("").map(extract_volume_ml).map(lambda t: t[0])
price = pd.to_numeric(raw["price"], errors="coerce")

stats = pd.DataFrame({
    "metric": ["rows", "retailers", "countries", "categories",
               "barcode coverage", "volume coverage", "description missing",
               "price coverage", "price CV (local currency)"],
    "value": [len(raw), raw["retailer"].nunique(), raw["country"].nunique(), raw["category"].nunique(),
              round((barcode.str.len() > 0).mean(), 3), round(vol.notna().mean(), 3),
              round(raw["description"].fillna("").eq("").mean(), 3),
              round(price.notna().mean(), 3), round(price.std() / price.mean(), 2)],
})
stats"""),

    code("""# duplicate-listing structure: how many rows are repeat (retailer, title)?
dup = raw.duplicated(subset=["retailer", "title"]).mean()
dup_bc = raw.duplicated(subset=["retailer", "barcode"]).mean()
print(f"repeat (retailer,title): {dup:.1%}   repeat (retailer,barcode): {dup_bc:.1%}")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
raw["category"].value_counts().head(10).plot(kind="barh", ax=axes[0], color="#4C72B0")
axes[0].set_title("Top 10 categories")
axes[0].set_xlabel("SKUs")
coverage = pd.Series({"has barcode": (barcode.str.len() > 0).mean(),
                      "no barcode": (barcode.str.len() == 0).mean()})
coverage.plot(kind="bar", ax=axes[1], color=["#4C72B0", "#BBBBBB"], rot=0)
axes[1].set_title("Barcode (GTIN) coverage — the hard-link signal")
axes[1].set_ylabel("share of SKUs")
plt.show()"""),

    md("""**EDA takeaways.** Barcode covers only **42%** of SKUs — so it is the *hard* link but cannot resolve everything. Price is **local currency** across 19 countries (CV 5.0) ⇒ unusable without FX. Volume is extractable from ~64% of titles. ~19% of rows are repeat (retailer, title) marketplace listings ⇒ dedupe before matching."""),

    md("""## 4. Feature engineering

Three cheap, high-signal features are engineered per representative SKU:
- **canonical volume** (ml) parsed from the title (regex extractor, `_text.py`),
- **macro category** (24 Euromonitor categories → 6 coarse buckets, for blocking),
- **barcode flag** — whether the representative carries a hard-link GTIN."""),

    code("""reps = load_dataset_deduped()   # step 06's tiered dedupe output (matching input)
reps["canonical_volume_ml"] = reps["title"].fillna("").map(extract_volume_ml).map(lambda t: t[0])
reps["macro_category"] = reps["category"].fillna("").map(lambda c: MACRO_MAP.get(c, "OTHER"))
reps["has_barcode"] = reps["barcode"].fillna("").str.len().gt(0)
print(f"representatives: {len(reps):,}  |  volume extracted: {reps['canonical_volume_ml'].notna().mean():.1%}  |  macro buckets: {reps['macro_category'].nunique()}")
reps[["title", "brand", "macro_category", "canonical_volume_ml", "has_barcode"]].head(3)"""),

    md("""## 5. Dedupe — already applied by 06_dedupe.py

`06_dedupe.py` collapsed the within-retailer marketplace listings (tiered: retailer+barcode → retailer+title+price → retailer+title) into `dataset_deduped.csv` and wrote `sku_to_rep.csv` mapping every raw SKU to its representative. We load that mapping here so the final ITEM_ID derives from step 06's single source of truth instead of a notebook re-implementation."""),

    code("""# every raw SKU -> its deduped representative id (positional index into `reps`)
sku_to_rep = pd.read_csv(SKU_TO_REP, dtype=str)
print(f"SKU -> representative mapping: {len(sku_to_rep):,} rows")
assert len(sku_to_rep) == len(raw), "sku_to_rep.csv does not cover every raw SKU"
assert set(sku_to_rep["product_id"]) == set(raw["product_id"]), "product_id mismatch vs raw"
"""),

    md("""## 6. Embed the representatives

A bi-encoder turns each representative's `title | brand | category` into a 384-dim vector. Cosine similarity between two vectors is our "same product" score. (The encode is cached, so re-runs are instant.)"""),

    code("""payload = (reps["title"].fillna("") + " | " + reps["brand"].fillna("") + " | " + reps["category"].fillna("")).tolist()
emb, _ = encode_corpus(MODEL, payload, batch_size=256, max_seq_length=128, cache_dir=CACHE)
print(f"embeddings: {emb.shape}")"""),

    md("""## 7. Link — hard (barcode) + soft (cosine) edges

Then take the **transitive closure**: a connected component over all edges is one ITEM. This lets one strong barcode link carry a whole chain of fuzzy links."""),

    code("""n = len(reps)
edges = set()

# 7a. hard links: identical non-empty barcode => same product
bc = reps["barcode"].fillna("").astype(str)
for g, idxs in reps.groupby(bc).indices.items():
    if g == "":
        continue
    for a, b in itertools.combinations(idxs, 2):
        edges.add((min(a, b), max(a, b)))
n_barcode = len(edges)

# 7b. soft links: SAME product = same brand + same size + near-identical title.
# Cosine >= 0.55 alone over-merges (chains every variant of a brand into one
# cluster). Adding the volume constraint + a higher threshold keeps only true
# same-product pairs (same brand + same size + same flavor).
SOFT_THRESHOLD = 0.85
reps["macro_category"] = reps["category"].fillna("").map(lambda c: MACRO_MAP.get(c, "OTHER"))
reps["_brand"] = reps["brand"].fillna("")
reps["_vol"] = reps["title"].fillna("").map(extract_volume_ml).map(lambda t: t[0])
vol_arr = reps["_vol"].to_numpy()
for (br, m), grp in reps.groupby(["_brand", "macro_category"], sort=False):
    idx = grp.index.to_numpy()
    if len(idx) < 2:
        continue
    nn = NearestNeighbors(radius=1.0 - SOFT_THRESHOLD, metric="cosine")
    nn.fit(emb[idx])
    dist, neigh = nn.radius_neighbors(emb[idx])
    for i, row in enumerate(neigh):
        a = int(idx[i])
        for j in row:
            b = int(idx[j])
            if a >= b:
                continue
            va, vb = vol_arr[a], vol_arr[b]
            if pd.notna(va) and pd.notna(vb) and va != vb:
                continue  # different size -> different product
            edges.add((a, b))

print(f"barcode edges: {n_barcode:,}  |  total edges (barcode + fuzzy): {len(edges):,}")"""),

    code("""# transitive closure -> ITEM_ID per representative (symmetric adjacency for undirected components)
rows = np.array(sorted(edges))
symmetric = np.vstack([rows, rows[:, ::-1]])            # (i,j) AND (j,i)
graph = csr_matrix((np.ones(len(symmetric)), (symmetric[:, 0], symmetric[:, 1])), shape=(n, n))
n_items, rep_item = connected_components(graph, directed=False)
print(f"ITEMs (clusters): {n_items:,}  from {n:,} representatives")
print(f"sizes: median {np.median(np.bincount(rep_item)):.0f}, max {np.bincount(rep_item).max():,}")"""),

    code("""# map every SKU to its ITEM_ID via 06's representative, and write the deliverable
rep_of = dict(zip(sku_to_rep["product_id"], sku_to_rep["rep_id"].astype(int)))
raw["ITEM_ID"] = [rep_item[rep_of[pid]] for pid in raw["product_id"]]
out = raw[["product_id", "ITEM_ID"]].rename(columns={"product_id": "SKU_ID"})
out_path = PATHS.experiments / "results" / "euromonitor" / "sku_to_item.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(out_path, index=False)
print(f"wrote {out_path}  ({len(out):,} SKUs -> {out['ITEM_ID'].nunique():,} ITEMs)")
out.head()"""),

    md("""## 8. Validation — does the grouping match ground truth?

Barcode gives us an **independent** ground truth: two listings with the same non-empty barcode *are* the same product. We score the bi-encoder on that truth (same-barcode pairs = positives, cross-barcode pairs = negatives) and report **PR-AUC and precision@recall** — not just AUC, because AUC saturates (~0.999) on easy random negatives."""),

    code("""from _blocking import build_pairs

pos, neg = build_pairs(reps, SEED, 4, 10_000)   # same-barcode / cross-barcode
pos_s = _cosine(emb, pos)
neg_s = _cosine(emb, neg)
m = entity_resolution_metrics(pos_s, neg_s)
pd.DataFrame([{"metric": k, "value": v} for k, v in m.items()])"""),

    md("""**Readout.** `precision_at_90pct_recall` is the number that matters: at 90% recall, ~99% of flagged pairs are genuinely the same product. The remaining errors are concentrated in **different-brand, same-category** near-duplicates (a known, mineable hard band), and in **mislabeled barcodes** (same title, conflicting barcodes) which we explicitly exclude from training."""),

    md("""## 9. Caveats & known limits

- **Barcode coverage is 42%**, so the hard-link signal is partial; the fuzzy bi-encoder link carries the rest.
- **Mislabeled barcodes** (~0.6% of cross-retailer exact-title groups carry conflicting barcodes) can over-merge distinct products — flagged, not auto-resolved.
- **Price is local currency** across 19 countries (CV 5.0) and is deliberately *not* used as a matching feature without FX normalization.
- **Volume** is a strong discriminator when both sides have it, but is redundant on top of the semantic embedding (title already encodes "500ml"), so it is not a primary link feature.
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {OUT} ({len(cells)} cells)")
