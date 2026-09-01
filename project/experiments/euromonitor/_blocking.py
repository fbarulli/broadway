"""_blocking.py: shared blocking evaluation machinery for the euromonitor series.

Single source for the blocking A/B measurement used by 04c (the three-rule
experiment) and 05 (per-feature utility + greedy key selection): true-pair
construction and the recall/candidates evaluator. No regexes here — those
stay in _text.py.
"""

from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd


def build_true_pairs(df: pd.DataFrame,
                     barcode_col: str = "barcode",
                     retailer_col: str = "retailer") -> list[tuple[int, int]]:
    """ALL same-barcode pairs inside multi-retailer groups (the ground truth).

    Keeps the ORIGINAL df index (never reset) so pair indices map straight
    into df.loc / feature Series indexed by df.index.
    """
    barcodes = df[barcode_col].fillna("").astype(str)
    known = df[barcodes.str.len() > 0]
    multi = known[known.groupby(barcode_col)[retailer_col].transform("nunique") > 1]
    pairs: list[tuple[int, int]] = []
    for _, g in multi.groupby(barcode_col):
        pairs.extend(combinations(g.index.tolist(), 2))
    return pairs


def build_pairs(
    df: pd.DataFrame,
    seed: int,
    max_pos_per_group: int,
    n_neg: int,
    barcode_col: str = "barcode",
    retailer_col: str = "retailer",
    title_col: str = "title",
) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth evaluation pairs, 0..n-1 row-indexed.

    positives: title pairs inside the same multi-retailer barcode group, built
    from one row per unique stripped title (capped at max_pos_per_group per
    group) so trivially-identical listings don't inflate agreement. negatives:
    cross-barcode pairs with different title text, sampled in ONE vectorized
    bulk pass (no per-attempt Python loop). Returns (pos_pairs, neg_pairs) as
    (N, 2) int arrays.
    """
    barcodes = df[barcode_col].fillna("").astype(str)
    titles = df[title_col].fillna("").str.strip()
    rng = np.random.default_rng(seed)

    known = df[barcodes.str.len() > 0]
    multi = known[known.groupby(barcode_col)[retailer_col].transform("nunique") > 1]
    pos_i: list[int] = []
    pos_j: list[int] = []
    for _, g in multi.groupby(barcode_col):
        sub = g.assign(_t=titles.loc[g.index])
        rows = sub[sub["_t"] != ""].drop_duplicates("_t").index.tolist()
        combos = list(combinations(rows, 2))
        if len(combos) > max_pos_per_group:
            chosen = rng.choice(len(combos), max_pos_per_group, replace=False)
            combos = [combos[k] for k in chosen]
        for a, b in combos:
            pos_i.append(a)
            pos_j.append(b)

    n = len(df)
    bc = barcodes.to_numpy()
    tt = titles.to_numpy()
    a = rng.integers(0, n, size=n_neg * 60)
    b = rng.integers(0, n, size=n_neg * 60)
    mask = (a != b) & (bc[a] != bc[b]) & (tt[a] != tt[b])
    if int(mask.sum()) < n_neg:
        raise RuntimeError(f"only {int(mask.sum())} valid negative pairs sampled (need {n_neg})")

    pos_pairs = np.column_stack([pos_i, pos_j]).astype(int) if pos_i else np.empty((0, 2), dtype=int)
    return pos_pairs, np.column_stack([a[mask][:n_neg], b[mask][:n_neg]]).astype(int)


def eval_blocking(key_series: pd.Series, true_pairs: list[tuple[int, int]]):
    """Blocking recall + candidate count for one key series (indexed by df row).

    recall     = true pairs sharing a block / all true pairs.
    candidates = sum of n*(n-1)/2 over blocks (pairwise classifier cost).
    n_blocks   = number of non-empty blocks.
    """
    idx2block = dict(key_series)
    blocks = defaultdict(int)
    for k in key_series:
        blocks[k] += 1
    n_cand = sum(n * (n - 1) // 2 for n in blocks.values())
    retained = sum(
        1 for a, b in true_pairs
        if idx2block.get(a) == idx2block.get(b))
    return retained / len(true_pairs), n_cand, len(blocks)
