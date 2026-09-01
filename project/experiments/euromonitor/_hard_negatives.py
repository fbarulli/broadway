"""Hard-negative mining for euromonitor entity resolution.

Mines cross-barcode pairs the bi-encoder finds confusing (the 0.5-0.75 cosine
band) while EXCLUDING known label errors: same-title+brand rows carrying
conflicting barcodes (the mislabeled-barcode groups). The output is auditable —
a CSV with title/brand/barcode/cosine per pair — so a human can hand-label a
sample and the exclusion is visible, never hidden.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from _text import MACRO_MAP
from sklearn.neighbors import NearestNeighbors


def conflicting_barcode_pairs(df: pd.DataFrame) -> set[tuple[int, int]]:
    """Row-index pairs that are same (title, brand) but different barcode.

    These are label errors (same product, conflicting barcode): a pair like this
    must never enter the negative pool, or training teaches the model to push
    apart titles that are actually the same product.
    """
    barcodes = df["barcode"].fillna("").astype(str)
    pairs: set[tuple[int, int]] = set()
    for _, g in df.groupby(["title", "brand"], sort=False):
        bcs = barcodes.loc[g.index]
        uniq = bcs[bcs.str.len() > 0].unique()
        if len(uniq) <= 1:
            continue
        idx = g.index.tolist()
        for a, b in combinations(idx, 2):
            if barcodes.loc[a] != barcodes.loc[b]:
                pairs.add((int(a), int(b)))
    return pairs


def exact_title_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Group-level census of exact-title (title, brand) groups.

    Single source for the mislabeled-barcode census used by 06b (raw) and 06c
    (deduped): per (title, brand) group, the number of retailers, the number of
    distinct non-empty barcodes, and the barcode/retailer sets for review. A
    group with >1 retailer and >1 real barcode is a conflicting-barcode label
    error (same product, two GTINs); <=1 barcode is a clean calibration positive.
    """
    cb = (
        df.groupby(["title", "brand"], dropna=False)
        .agg(
            n_retailers=("retailer", "nunique"),
            barcodes=("barcode", lambda s: sorted({str(x) for x in s.dropna() if str(x)})),
            retailers=("retailer", lambda s: sorted(set(s.dropna()))),
        )
        .reset_index()
    )
    cb["n_barcodes"] = cb["barcodes"].apply(len)
    return cb


def split_barcodes(df: pd.DataFrame, seed: int = 42) -> dict[str, set[str]]:
    """Barcode-level 70/15/15 train/val/test split over multi-retailer barcodes.

    Splits on the barcode (entity) so no product's rows straddle a split — the
    guard against same-product leakage. Returns sets of barcode strings.
    """
    barcodes = df["barcode"].fillna("").astype(str)
    known = df[barcodes.str.len() > 0]
    multi = known[known.groupby("barcode")["retailer"].transform("nunique") > 1]
    bcs = np.array(sorted(multi["barcode"].unique()))
    perm = np.random.default_rng(seed).permutation(len(bcs))
    tr = int(0.70 * len(bcs))
    va = int(0.85 * len(bcs))
    return {
        "train": set(bcs[perm[:tr]]),
        "val": set(bcs[perm[tr:va]]),
        "test": set(bcs[perm[va:]]),
    }


def pairs_in_set(pairs: np.ndarray, row_barcodes: np.ndarray, bc_set: set[str]) -> np.ndarray:
    """Boolean mask over pairs whose BOTH endpoints' barcode is in bc_set.

    Held-out pair filtering: a pair is only in the split if both rows belong to
    it, so no train/test entity leaks across the boundary.
    """
    members = list(bc_set)
    return (
        np.isin(row_barcodes[pairs[:, 0]], members)
        & np.isin(row_barcodes[pairs[:, 1]], members)
    )


def build_triplets(
    train_pos: np.ndarray,
    hard_train: np.ndarray,
    payload: list[str],
    *,
    seed: int = 42,
    max_triples: int = 5_000,
) -> list:
    """Build (anchor, positive, hard-negative) triples for TripletLoss.

    Each hard-negative partner is drawn from the anchor's mined hard negatives
    (falling back to the positive partner's). Capped at max_triples so the
    fine-tune stays tractable.
    """
    from sentence_transformers import InputExample

    hn_map: dict[int, list[int]] = defaultdict(list)
    for a, b in hard_train:
        hn_map[int(a)].append(int(b))

    rng = np.random.default_rng(seed)
    triples: list[InputExample] = []
    for a, b in train_pos:
        partners = hn_map.get(int(a)) or hn_map.get(int(b))
        if not partners:
            continue
        c = int(partners[rng.integers(len(partners))])
        triples.append(InputExample(texts=[payload[a], payload[b], payload[c]]))
        if len(triples) >= max_triples:
            break
    return triples


def mine_hard_negatives(
    df: pd.DataFrame,
    emb: np.ndarray,
    *,
    seed: int = 42,
    n_target: int = 10_000,
    cosine_lo: float = 0.5,
    cosine_hi: float = 0.75,
    exclude_conflicting: bool = True,
    k: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Mine hard negatives: cross-barcode, different-brand, same-macro, mid-cosine.

    Uses cosine ANN within each macro-category block, then filters to the
    confusion band (cosine_lo..cosine_hi) with a DIFFERENT brand (the signature
    of the champion's false positives), a different non-empty barcode, and no
    conflicting-barcode label error. Returns (pairs, cosine) as an (N,2) int
    array and an (N,) float array, hardest-first.
    """
    barcodes = df["barcode"].fillna("").astype(str).to_numpy()
    brands = df["brand"].fillna("").astype(str).to_numpy()
    macro = df["category"].fillna("").map(lambda c: MACRO_MAP.get(c, "?")).to_numpy()
    excluded = conflicting_barcode_pairs(df) if exclude_conflicting else set()

    found: list[tuple[int, int, float]] = []
    for m in np.unique(macro):
        idx = np.flatnonzero(macro == m)
        if len(idx) < 2:
            continue
        nn = NearestNeighbors(n_neighbors=min(k, len(idx)), metric="cosine")
        nn.fit(emb[idx])
        dist, neigh = nn.kneighbors(emb[idx])
        sim = 1.0 - dist
        for i, row_neigh in enumerate(neigh):
            a = int(idx[i])
            for col, j in enumerate(row_neigh):
                b = int(idx[j])
                if a >= b:
                    continue
                s = float(sim[i, col])
                if not (cosine_lo <= s <= cosine_hi):
                    continue
                if barcodes[a] == "" or barcodes[b] == "" or barcodes[a] == barcodes[b]:
                    continue  # missing or same barcode
                if brands[a] == brands[b]:
                    continue  # want DIFFERENT brand (the champion's FP signature)
                if (a, b) in excluded:
                    continue  # same product, conflicting barcode -> label error
                found.append((a, b, s))

    found.sort(key=lambda t: -t[2])  # hardest (highest cosine) first
    seen: set[tuple[int, int]] = set()
    pairs_out: list[tuple[int, int]] = []
    cos_out: list[float] = []
    for a, b, s in found:
        if (a, b) in seen:
            continue
        seen.add((a, b))
        pairs_out.append((a, b))
        cos_out.append(s)
        if len(pairs_out) >= n_target:
            break

    if not pairs_out:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=float)
    return np.asarray(pairs_out, dtype=int), np.asarray(cos_out, dtype=float)


def write_hard_negative_csv(df: pd.DataFrame, pairs: np.ndarray, cosine: np.ndarray, path) -> pd.DataFrame:
    """Auditable CSV: title/brand/barcode/macro per side + cosine.

    Columns are the raw identity fields (not indices) so a reviewer can
    hand-label a sample without re-joining, and the conflicting-barcode
    exclusion is visible in the barcode_a/barcode_b columns.
    """
    title = df["title"].fillna("").astype(str).to_numpy()
    brand = df["brand"].fillna("").astype(str).to_numpy()
    barcode = df["barcode"].fillna("").astype(str).to_numpy()
    macro = df["category"].fillna("").map(lambda c: MACRO_MAP.get(c, "?")).to_numpy()
    a, b = pairs[:, 0], pairs[:, 1]
    out = pd.DataFrame({
        "title_a": title[a], "title_b": title[b],
        "brand_a": brand[a], "brand_b": brand[b],
        "barcode_a": barcode[a], "barcode_b": barcode[b],
        "macro_a": macro[a], "macro_b": macro[b],
        "cosine": np.round(cosine, 4),
    })
    out.to_csv(path, index=False)
    return out
