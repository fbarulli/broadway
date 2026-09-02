"""Build the entity-resolution deliverable notebook (SKU_ID -> ITEM_ID)."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "entity_resolution.ipynb"
SERIES_ABS = Path(__file__).resolve().parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


# The notebook must run from any cwd, so the series directory is baked in as an
# absolute path (derived from this file) instead of Path.cwd().
setup_src = """import itertools
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

SERIES = Path(__SERIES_ABS__)   # absolute path baked in by make_notebook.py -> runs from any cwd
sys.path.insert(0, str(SERIES))

from _common import PATHS, SEED, load_dataset, load_dataset_deduped
from _text import MACRO_MAP, extract_volume_ml, flavor_from_name

from broadway.training.nlp import _cosine, load_cached_corpus

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE = str(PATHS.experiments.parent / "data" / "euromonitor" / "embeddings_cache")
DATA_DIR = PATHS.experiments.parent / "data" / "euromonitor"
SKU_TO_REP = DATA_DIR / "sku_to_rep.csv"   # written by 06_dedupe.py
print("setup ok")""".replace("__SERIES_ABS__", repr(str(SERIES_ABS)))


cells = [
    md("""# Euromonitor Entity Resolution — SKU → ITEM grouping

**Goal.** Given ~72k product listings (SKUs) scraped from 280 retailers across 19 countries, assign each SKU an **ITEM_ID** so that every listing of the *same physical product* shares one ITEM_ID. This is **entity resolution / record linkage**.

**Output.** A two-column table `SKU_ID → ITEM_ID` (written to `results/euromonitor/sku_to_item.csv`). The exact SKU → ITEM count is machine-generated in the summary cell below (read back from the written CSV).
"""),

    md("""## 1. The method and why

The naive approach — compare every SKU to every other SKU — is ~2.6 **billion** pairs, too slow and far too noisy. A production matcher needs a **block → score → link** pipeline:

1. **Block** on `brand × macro-category` to cut candidates ~1000× (recall 0.957, ~1.8M pairs) — blocking is *recall-first*.
2. **Score** each blocked pair with a **bi-encoder** (sentence-transformers `all-MiniLM-L6-v2`) cosine similarity. It beats TF-IDF (AUC 0.9993 vs 0.983) because it is *semantic*: `"Coca-Cola 500ml"` and `"Coca Cola 0.5 L"` are seen as the same product despite different words.
3. **Link** with two rules:
   - **hard link** — identical non-empty **barcode** (GTIN) ⇒ same ITEM (ground-truth identity),
   - **soft link** — same brand + same volume + same flavor + cosine ≥ 0.85 ⇒ same ITEM (the tighter constraint that stops cross-variant chain-merging; the buggy 0.55 over-merge value is deliberately not used).
4. **Transitive closure in two stages** — close over barcode edges first, then admit a soft edge only if it does **not** join two clusters that each already carry a distinct non-empty barcode, so a mislabeled barcode or one fuzzy edge cannot chain distinct product clusters.
5. **Dedupe first** — within-retailer marketplace listings (one title listed 143× on Gittigidiyor) were collapsed to a representative by `06_dedupe.py` *before* matching; this notebook consumes that deduped output.
"""),

    md("""## 2. Setup"""),

    code(setup_src),

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

A bi-encoder turns each representative's `title | brand | category` into a 384-dim vector; cosine similarity between two vectors is our "same product" score. The embeddings are **precomputed once by the pipeline** and cached to disk, so this notebook only *loads* them — it never pulls model weights."""),

    code("""payload = (reps["title"].fillna("") + " | " + reps["brand"].fillna("") + " | " + reps["category"].fillna("")).tolist()
emb, _ = load_cached_corpus(MODEL, payload, batch_size=256, max_seq_length=128, cache_dir=CACHE)
print(f"embeddings: {emb.shape}")"""),

    md("""## 7. Link — hard (barcode) + soft (cosine) edges

Soft links are only *candidates* at first. We close over **barcode edges first** (union-find), then admit a soft edge only if it does **not** join two clusters that each already carry a distinct non-empty barcode. Keeping barcode and fuzzy edges separate stops a mislabeled barcode or a single fuzzy edge from chaining distinct product clusters."""),

    code("""n = len(reps)

# 7a. hard links: identical non-empty barcode => same product (ground truth).
bc = reps["barcode"].fillna("").astype(str)
barcode_edges = set()
for g, idxs in reps.groupby(bc).indices.items():
    if g == "":
        continue
    for a, b in itertools.combinations(idxs, 2):
        barcode_edges.add((min(a, b), max(a, b)))
n_barcode = len(barcode_edges)

# 7b. soft-link candidates: same (normalized) brand x macro block, cosine >= 0.85,
# then veto a pair whose volume differs (when both parse) or whose flavor differs
# (when both are detected).
SOFT_THRESHOLD = 0.85
reps["macro_category"] = reps["category"].fillna("").map(lambda c: MACRO_MAP.get(c, "OTHER"))
reps["_brand"] = reps["brand"].fillna("").astype(str).str.strip().str.casefold()
reps["_vol"] = reps["title"].fillna("").map(extract_volume_ml).map(lambda t: t[0])
reps["_flavor"] = reps["title"].fillna("").map(flavor_from_name)
vol_arr = reps["_vol"].to_numpy()
flav_arr = reps["_flavor"].to_numpy()

soft_edges = set()
cand_pairs = []
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
            cand_pairs.append((a, b))   # in-block pair at cosine >= 0.85
            va, vb = vol_arr[a], vol_arr[b]
            if pd.notna(va) and pd.notna(vb) and va != vb:
                continue  # different size -> different product
            fa, fb = flav_arr[a], flav_arr[b]
            if fa is not None and fb is not None and fa != fb:
                continue  # different flavor -> different product
            soft_edges.add((a, b))

# 7c. two-stage closure: union barcode edges first, then admit a soft edge only if
# it does NOT join two clusters that each already carry a distinct non-empty barcode.
parent = list(range(n))
comp_size = [1] * n
comp_has_bc = (bc != "").tolist()

def _find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def _union(a, b):
    ra, rb = _find(a), _find(b)
    if ra == rb:
        return
    if comp_size[ra] < comp_size[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    comp_size[ra] += comp_size[rb]
    comp_has_bc[ra] = comp_has_bc[ra] or comp_has_bc[rb]

for a, b in barcode_edges:
    _union(a, b)

edges = set(barcode_edges)
n_soft_admitted = 0
n_soft_blocked = 0
for a, b in sorted(soft_edges):
    ra, rb = _find(a), _find(b)
    if ra != rb and comp_has_bc[ra] and comp_has_bc[rb]:
        n_soft_blocked += 1   # would chain two barcode-identified clusters
        continue
    edges.add((a, b))
    _union(a, b)
    n_soft_admitted += 1

print(f"barcode edges: {n_barcode:,}  |  soft candidates: {len(soft_edges):,}  |  admitted: {n_soft_admitted:,}  (blocked: {n_soft_blocked:,})  |  total edges: {len(edges):,}")"""),

    code("""# transitive closure -> ITEM_ID per representative (symmetric adjacency for undirected components)
if edges:
    rows = np.array(sorted(edges))
    symmetric = np.vstack([rows, rows[:, ::-1]])            # (i,j) AND (j,i)
    graph = csr_matrix((np.ones(len(symmetric)), (symmetric[:, 0], symmetric[:, 1])), shape=(n, n))
    n_items, rep_item = connected_components(graph, directed=False)
else:
    n_items, rep_item = n, np.arange(n)                     # no edges -> every rep its own ITEM
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

    md("""## 8. Result — machine-generated headline

The deliverable resolves all 71,623 SKUs into ITEMs with dense, contiguous ITEM_IDs. The exact ITEM count and the max cluster size are read back from `sku_to_item.csv` in the code cell immediately below — never a hand-typed number."""),

    code("""# authoritative headline, read back from the deliverable
deliverable = pd.read_csv(out_path)
n_sku = len(deliverable)
n_item = int(deliverable["ITEM_ID"].nunique())
item_ids = np.sort(deliverable["ITEM_ID"].unique())
contiguous = bool((item_ids == np.arange(n_item)).all())
max_cluster = int(np.bincount(rep_item).max())
print(f"HEADLINE: {n_sku:,} SKUs -> {n_item:,} ITEMs  |  ITEM_ID unique & contiguous 0..{n_item - 1}: {contiguous}  |  max cluster size: {max_cluster:,}")"""),

    md("""## 9. Validation — does the grouping match ground truth?

Barcode gives an **independent** ground truth: two listings with the same non-empty barcode *are* the same product. Precision is reported at the **actual 0.85 link threshold** (not a quantile operating point), and the negative population is stated explicitly:

- **soft-link candidates** — the real linking population: same brand × macro pairs at cosine ≥ 0.85, labeled by barcode (same barcode = match, both-known-different barcodes = non-match).
- **random cross-barcode negatives** — the easy population the old metric used.
- **mined hard negatives** — `_hard_negatives.mine_hard_negatives` (different brand × same macro, the model's 0.45–0.80 confusion band).
"""),

    code("""from _blocking import build_pairs
from _hard_negatives import mine_hard_negatives
from sklearn.metrics import average_precision_score, roc_auc_score

SOFT_THRESHOLD = 0.85   # the ACTUAL link threshold

def precision_at(pos_s, neg_s, thr):
    tp = float((pos_s >= thr).sum())
    fp = float((neg_s >= thr).sum())
    return (tp / (tp + fp)) if (tp + fp) > 0 else float("nan")

# (a) the soft-link population itself, labeled by barcode ground truth.
if cand_pairs:
    cand = np.asarray(cand_pairs, dtype=int)
    bca = bc.to_numpy()[cand[:, 0]]
    bcb = bc.to_numpy()[cand[:, 1]]
    n_same = int(((bca != "") & (bca == bcb)).sum())
    n_cross = int(((bca != "") & (bcb != "") & (bca != bcb)).sum())
    precision_candidates = n_same / (n_same + n_cross) if (n_same + n_cross) else float("nan")
else:
    n_same = n_cross = 0
    precision_candidates = float("nan")

# (b) barcode ground-truth pairs (same-barcode positives, cross-barcode negatives).
pos, neg = build_pairs(reps, SEED, 4, 10_000)
pos_s = _cosine(emb, pos)
neg_s = _cosine(emb, neg)
precision_easy = precision_at(pos_s, neg_s, SOFT_THRESHOLD)

# (c) mined hard negatives (different brand x same macro — the champion's FP signature).
hard, hard_s = mine_hard_negatives(reps, emb, seed=SEED, n_target=20_000, cosine_lo=0.45, cosine_hi=0.80)
precision_hard = precision_at(pos_s, hard_s, SOFT_THRESHOLD)

precision = pd.DataFrame([
    {"population": "soft-link candidates (same brand x macro)",
     "positives": n_same, "negatives": n_cross, "precision@0.85": round(precision_candidates, 4)},
    {"population": "random cross-barcode negatives",
     "positives": len(pos_s), "negatives": len(neg_s), "precision@0.85": round(precision_easy, 4)},
    {"population": "mined hard negatives (diff brand)",
     "positives": len(pos_s), "negatives": len(hard_s), "precision@0.85": round(precision_hard, 4)},
])

# ranking context (AUC / PR-AUC) — no quantile-based precision_at_90pct_recall
y = np.r_[np.ones(len(pos_s)), np.zeros(len(neg_s))]
scores = np.r_[pos_s, neg_s]
ranking = pd.DataFrame([
    {"metric": "roc_auc", "value": round(float(roc_auc_score(y, scores)), 4)},
    {"metric": "average_precision", "value": round(float(average_precision_score(y, scores)), 4)},
    {"metric": "pos_median_cosine", "value": round(float(np.median(pos_s)), 4)},
    {"metric": "neg_p90_cosine", "value": round(float(np.quantile(neg_s, 0.9)), 4)},
])
print("precision @ 0.85 (the actual link threshold):")
print(precision.to_string(index=False))
print()
print("ranking context (no quantile precision):")
print(ranking.to_string(index=False))"""),

    md("""**Readout.** `precision@0.85` is computed at the **actual link threshold**, not a quantile operating point. The number that matters is the **soft-link candidate** row: among same-brand × same-macro pairs at cosine ≥ 0.85 (the pairs the soft rule could actually link), the barcode-labeled precision is the fraction that are genuinely the same product. The random cross-barcode negatives are easy, and the mined hard negatives sit in the 0.45–0.80 confusion band — below the link threshold — so they contribute no false positives at 0.85."""),

    md("""## 10. Worked example — a cross-country success and a genuine failure

The bi-encoder pays a **translation tax**: the same physical product listed in two countries often has a differently-worded title, so its cosine drops. Two same-barcode pairs — pulled straight from the zero-shot scores — make that tax concrete:

- **cross-country SUCCESS** — one product whose two listings live in different countries, yet the encoder still scores it high enough to match.
- **genuine FAILURE** — one same-barcode pair the encoder scores just below the operating threshold (the 5% false-positive-rate point on the cross-barcode negatives — the same threshold `07_report_plots.py` uses; *not* the 0.85 soft-link threshold).

Each pair shows the two titles, the brand, the two countries, and the cosine, plus one plain-English *why*. The "why" is **inferred** from the pair's attributes (missing brand, very short title, or wording divergence) — not proven to be the cause. A miss whose two titles are *very* different may in fact be a mislabeled barcode (two products sharing a GTIN, ~0.6% of cross-retailer groups) — a label error the encoder is arguably right about — rather than a wording gap it failed to bridge."""),

    code("""# worked example: one cross-country SUCCESS and one genuine FAILURE from the
# zero-shot bi-encoder's own scores. Same-barcode pairs are ground-truth matches;
# the operating threshold mirrors 07_report_plots.py (5% FPR on cross-barcode
# negatives), NOT the 0.85 soft-link threshold.
thr = float(np.quantile(neg_s, 0.95))
title = reps["title"].fillna("").astype(str).to_numpy()
brand = reps["brand"].fillna("").astype(str).to_numpy()
country = reps["country"].fillna("").astype(str).to_numpy()
a, b = pos[:, 0], pos[:, 1]
cross = (country[a] != "") & (country[b] != "") & (country[a] != country[b])

# 1. cross-country SUCCESS: the highest-scoring same-barcode pair across two
#    countries whose titles genuinely differ once case and whitespace are
#    normalized — a match earned in spite of the wording gap, not a trivially
#    identical title. Falls back to the plain argmax if none exists.
norm = np.array([" ".join(t.split()).lower() for t in title])
cross_diff = cross & (norm[a] != norm[b])
best_pool = cross_diff if cross_diff.any() else cross
if not best_pool.any():
    raise RuntimeError("no cross-country positive pair to show as a success")
best = int(np.flatnonzero(best_pool)[np.argmax(pos_s[best_pool])])

# 2. genuine FAILURE: a same-barcode positive below the operating threshold,
#    prefer cross-country. Take the CLOSEST miss (highest score below thr):
#    the pair the tax most plausibly cost, rather than a near-0.3 mislabeled
#    barcode (a label error, not a matching miss).
fn = pos_s < thr
fn_pool = (fn & cross) if (fn & cross).any() else fn
if not fn_pool.any():
    raise RuntimeError("no same-barcode positive scores below thr — no genuine failure to show")
fail = int(np.flatnonzero(fn_pool)[np.argmax(pos_s[fn_pool])])

def _why(i):
    x, y = pos[i]
    va, vb = vol_arr[x], vol_arr[y]
    no_brand = brand[x] == "" or brand[y] == ""
    short = len(title[x]) < 20 or len(title[y]) < 20
    same_vol = bool(pd.notna(va) and pd.notna(vb) and va == vb)
    if pos_s[i] >= thr:
        reasons = []
        if not no_brand:
            reasons.append("a shared brand")
        if same_vol:
            reasons.append("matching volume")
        tail = (" + " + " + ".join(reasons)) if reasons else ""
        return "near-identical titles" + tail + " carry the match across the language/formatting gap"
    if no_brand:
        return "missing brand leaves the encoder no brand signal to lean on"
    if short:
        return "very short title text gives the encoder too little to latch onto"
    return "heavy wording divergence between the two listings"

def _describe(i):
    x, y = pos[i]
    return {
        "title_a": title[x],
        "title_b": title[y],
        "brand": brand[x] if brand[x] else brand[y],
        "country_a": country[x],
        "country_b": country[y],
        "cosine": round(float(pos_s[i]), 4),
        "why (inferred)": _why(i),
    }

example = pd.DataFrame({label: _describe(i) for i, label in [(best, "cross-country SUCCESS"), (fail, "genuine FAILURE")]})
print(f"operating threshold (5% FPR): {thr:.3f}")
print(example.to_string(max_colwidth=42))"""),

    md("""## 11. Field ablation — which encode fields carry the decision?

Step 07c re-scored the bi-encoder after dropping each text field from the encode payload. AUC is the ranking objective, but the operating metric is **precision @ 90% recall** — how precise the score is at the decision threshold. At 90% recall the near-perfect full precision (0.9992) is a ground-truth-ease artifact, not brand leakage: **brand alone is strong (0.9982), but removing brand only drops the full precision@90%recall 0.9992 → 0.9950**, so no single field is load-bearing."""),

    code("""# field ablation: 07c re-scored the bi-encoder after dropping each encode field.
# Plot the operating metric (precision @ 90% recall), not the ranking objective (AUC).
abl_path = PATHS.experiments / "results" / "euromonitor" / "07c_field_ablation.csv"
abl = pd.read_csv(abl_path).sort_values("precision_at_90pct_recall", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 3.4), constrained_layout=True)
ax.barh(abl["variant"], abl["precision_at_90pct_recall"], color="#4C72B0")
ax.set_xlabel("precision @ 90% recall (same-barcode vs cross-barcode)")
ax.set_title("Bi-encoder field ablation — precision @ 90% recall by encode fields")
ax.invert_yaxis()   # highest precision on top
for y, v in enumerate(abl["precision_at_90pct_recall"]):
    ax.text(v + 0.0005, y, f"{v:.4f}", va="center", fontsize=9)
plt.show()
abl[["variant", "precision_at_90pct_recall"]]"""),

    md("""## 12. Caveats & known limits

- **Barcode coverage is 42%**, so the hard-link signal is partial; the fuzzy bi-encoder link carries the rest.
- **Mislabeled barcodes** (~0.6% of cross-retailer exact-title groups carry conflicting barcodes) can over-merge distinct products — the two-stage closure stops a mislabeled barcode from chaining distinct clusters, but it is flagged, not auto-resolved.
- **Price is local currency** across 19 countries (CV 5.0) and is deliberately *not* used as a matching feature without FX normalization.
- **Volume & flavor** are vetoes on the soft link, not primary features: a soft edge is rejected when both sides parse to different volumes, or when both sides carry different flavors.
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
