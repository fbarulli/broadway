"""_blocking.py: shared blocking evaluation machinery for the euromonitor series.

Single source for the blocking A/B measurement used by 04c (the three-rule
experiment) and 05 (per-feature utility + greedy key selection): true-pair
construction and the recall/candidates evaluator. No regexes here — those
stay in _text.py.
"""

from collections import defaultdict
from itertools import combinations

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
