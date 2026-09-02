"""_link.py: block -> score -> link — the production item-resolution stage.

Single source for the linking logic that turns representative embeddings into
contiguous ITEM_IDs: hard barcode edges + soft cosine edges (with a volume and
flavor veto), closed transitively in TWO stages so a mislabeled barcode or a
single fuzzy edge cannot chain distinct product clusters. Extracted verbatim
from the deliverable notebook's section 7 so the notebook and the consolidated
pipeline (08_pipeline.py) share one implementation.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from _text import MACRO_MAP, extract_volume_ml, flavor_from_name
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors


def resolve_items(reps: pd.DataFrame, emb: np.ndarray, *, soft_threshold: float = 0.85) -> dict:
    """Link representatives into ITEMs (contiguous, rep-level IDs).

    ``reps`` must be the deduped frame (0..n-1 RangeIndex, matching ``emb``
    row order). This function does NOT mutate ``reps`` — it works on an
    ``.assign()`` copy. Returns a dict with:
      rep_item          (n,) int array of contiguous ITEM_ID per representative
      n_items           int number of ITEMs
      n_barcode         int hard barcode edges
      n_soft_candidates int soft edges after the volume/flavor veto
      n_soft_admitted   int soft edges admitted by the two-stage closure
      n_soft_blocked    int soft edges rejected (would chain two barcode clusters)
      n_edges           int total edges in the final closure
      cand_pairs        list[(int, int)] in-block cosine>=thr pairs (pre-veto)
    """
    n = len(reps)
    bc = reps["barcode"].fillna("").astype(str)

    # 7a. hard links: identical non-empty barcode => same product (ground truth).
    barcode_edges: set[tuple[int, int]] = set()
    for g, idxs in reps.groupby(bc).indices.items():
        if g == "":
            continue
        for a, b in combinations(idxs, 2):
            barcode_edges.add((min(a, b), max(a, b)))
    n_barcode = len(barcode_edges)

    # 7b. soft-link candidates: same (normalized) brand x macro block at cosine
    # >= threshold, then veto a pair whose volume differs (when both parse) or
    # whose flavor differs (when both are detected).
    work = reps.assign(
        macro_category=reps["category"].fillna("").map(lambda c: MACRO_MAP.get(c, "OTHER")),
        _brand=reps["brand"].fillna("").astype(str).str.strip().str.casefold(),
        _vol=reps["title"].fillna("").map(extract_volume_ml).map(lambda t: t[0]),
        _flavor=reps["title"].fillna("").map(flavor_from_name),
    )
    vol_arr = work["_vol"].to_numpy()
    flav_arr = work["_flavor"].to_numpy()

    soft_edges: set[tuple[int, int]] = set()
    cand_pairs: list[tuple[int, int]] = []
    for (br, m), grp in work.groupby(["_brand", "macro_category"], sort=False):
        idx = grp.index.to_numpy()
        if len(idx) < 2:
            continue
        nn = NearestNeighbors(radius=1.0 - soft_threshold, metric="cosine")
        nn.fit(emb[idx])
        _, neigh = nn.radius_neighbors(emb[idx])
        for i, row in enumerate(neigh):
            a = int(idx[i])
            for j in row:
                b = int(idx[j])
                if a >= b:
                    continue
                cand_pairs.append((a, b))  # in-block pair at cosine >= threshold
                va, vb = vol_arr[a], vol_arr[b]
                if pd.notna(va) and pd.notna(vb) and va != vb:
                    continue  # different size -> different product
                fa, fb = flav_arr[a], flav_arr[b]
                if fa is not None and fb is not None and fa != fb:
                    continue  # different flavor -> different product
                soft_edges.add((a, b))

    # 7c. two-stage closure: union barcode edges first, then admit a soft edge
    # only if it does NOT join two clusters that each already carry a distinct
    # non-empty barcode.
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
            n_soft_blocked += 1  # would chain two barcode-identified clusters
            continue
        edges.add((a, b))
        _union(a, b)
        n_soft_admitted += 1

    # transitive closure -> ITEM_ID per representative (symmetric adjacency for
    # undirected components).
    if edges:
        rows = np.array(sorted(edges))
        symmetric = np.vstack([rows, rows[:, ::-1]])  # (i,j) AND (j,i)
        graph = csr_matrix(
            (np.ones(len(symmetric)), (symmetric[:, 0], symmetric[:, 1])),
            shape=(n, n),
        )
        n_items, rep_item = connected_components(graph, directed=False)
    else:
        n_items, rep_item = n, np.arange(n)  # no edges -> every rep its own ITEM

    return {
        "rep_item": np.asarray(rep_item, dtype=int),
        "n_items": int(n_items),
        "n_barcode": n_barcode,
        "n_soft_candidates": len(soft_edges),
        "n_soft_admitted": n_soft_admitted,
        "n_soft_blocked": n_soft_blocked,
        "n_edges": len(edges),
        "cand_pairs": cand_pairs,
    }
