"""08: Data prep — pre-compute the seven CSVs the deliverable notebook reads.

Deterministic (Act 1 samples with a fixed seed 42; Act 2 uses the series seed
from the euromonitor experiment config). Writes seven CSVs under
results/euromonitor/ that the notebook displays with trivial `pd.read_csv`
cells (nothing computed at runtime):

  Act 1 (data mechanics):
    - 08_act1_regex_examples.csv  5 real titles + the volume the regex recovered
    - 08_act1_dedup_pairs.csv     5 retailer+title groups the dedupe collapsed
    - 08_act1_funnel.csv          pipeline stage -> count (raw -> deduped -> ITEMs)

  Act 2 (matching evidence):
    - 08_act2_ground_truth.csv   ground-truth geography (single- vs cross-country share)
    - 08_act2_proxy.csv          silver cross-country proxy: honesty check + strengthened subset
    - 08_act2_four_pop.csv       four zero-shot score populations for the cross-border tax ECDF
    - 08_act2_fisher.csv         Fisher's exact on entity-level FN rates

Everything is computed here ONCE; the notebook only reads these CSVs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from _blocking import build_pairs
from _common import (
    DATA_PATH,
    PATHS,
    RESULTS,
    SEED,
    canonical_volume,
    load_dataset,
    load_dataset_deduped,
)
from _text import extract_volume_measurement, extract_volume_ml
from scipy.stats import fisher_exact

from broadway.training.nlp import load_cached_corpus

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE = str(PATHS.experiments.parent / "data" / "euromonitor" / "embeddings_cache")
ACT1_SEED = 42  # Act 1's example sampling is pinned independently of the series SEED


def _regex_examples(rng: np.random.Generator) -> pd.DataFrame:
    df = load_dataset()
    rows: list[dict] = []
    for title in df["title"].fillna("").astype(str):
        value, unit, ambiguous = extract_volume_measurement(title)
        if value is None or unit is None or ambiguous:
            continue  # skip titles with no volume, or bare-oz (weight vs fluid) ambiguity
        ml = extract_volume_ml(title)[0]
        if ml is None or not 50 <= ml <= 3000:
            continue  # skip implausible volumes (count-lists, multipacks, prose)
        if len(title) > 80:
            continue  # keep the display readable
        rows.append({"title": title, "volume_text": f"{value:g} {unit}", "canonical_ml": int(ml)})
    chosen = rng.choice(len(rows), size=5, replace=False)
    return pd.DataFrame([rows[i] for i in chosen])


def _dedup_pairs(rng: np.random.Generator) -> pd.DataFrame:
    # Reuse 06's own output: retailer+title groups with >1 distinct price are the
    # marketplace offers the dedupe deliberately collapsed to one representative.
    groups = pd.read_csv(RESULTS / "06_ambiguous_offer_groups.csv")
    chosen = rng.choice(len(groups), size=5, replace=False)
    return groups.iloc[chosen].reset_index(drop=True)


def _funnel() -> pd.DataFrame:
    raw_n = len(pd.read_csv(DATA_PATH, dtype=str, usecols=["sku_id"]))
    dedup_n = len(pd.read_csv(DATA_PATH.with_name("dataset_deduped.csv"), usecols=["product_id"]))
    item_n = int(pd.read_csv(RESULTS / "sku_to_item.csv", usecols=["item_id"])["item_id"].nunique())
    return pd.DataFrame([
        {"stage": "raw SKUs", "count": raw_n},
        {"stage": "deduped representatives", "count": dedup_n},
        {"stage": "ITEMs", "count": item_n},
    ])


def _embed(reps: pd.DataFrame) -> np.ndarray:
    payload = (
        reps["title"].fillna("")
        + " | "
        + reps["brand"].fillna("")
        + " | "
        + reps["category"].fillna("")
    ).tolist()
    emb, _ = load_cached_corpus(MODEL, payload, batch_size=256, max_seq_length=128, cache_dir=CACHE)
    return emb


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    # ---- Act 1: data-mechanics CSVs (regex examples, dedup pairs, funnel) ----
    rng = np.random.default_rng(ACT1_SEED)

    regex = _regex_examples(rng)
    regex.to_csv(RESULTS / "08_act1_regex_examples.csv", index=False)
    print(f"wrote {RESULTS / '08_act1_regex_examples.csv'} ({len(regex)} rows)")

    dedup = _dedup_pairs(rng)
    dedup.to_csv(RESULTS / "08_act1_dedup_pairs.csv", index=False)
    print(f"wrote {RESULTS / '08_act1_dedup_pairs.csv'} ({len(dedup)} rows)")

    funnel = _funnel()
    funnel.to_csv(RESULTS / "08_act1_funnel.csv", index=False)
    print(f"wrote {RESULTS / '08_act1_funnel.csv'} ({len(funnel)} rows)")

    print("\nregex examples:")
    print(regex.to_string(index=False, max_colwidth=60))
    print("\ndedup pairs:")
    print(dedup.to_string(index=False, max_colwidth=48))
    print("\nfunnel:")
    print(funnel.to_string(index=False))

    # ---- Act 2: matching-evidence CSVs (ground truth, proxy, populations, fisher) ----
    reps = load_dataset_deduped().reset_index(drop=True)
    emb = _embed(reps)

    def cos(pairs):
        pairs = np.asarray(pairs)
        return (emb[pairs[:, 0]] * emb[pairs[:, 1]]).sum(axis=1)

    country = reps["country"].fillna("").astype(str).to_numpy()
    brand = reps["brand"].fillna("").astype(str).to_numpy()
    cat = reps["category"].fillna("").astype(str).to_numpy()
    vol = canonical_volume(reps["title"].fillna("").astype(str))["canonical_volume_ml"].to_numpy()

    # ---- 1. ground-truth geography (95% single-country) ----
    raw = load_dataset()
    barcodes = raw["barcode"].fillna("").astype(str)
    known = raw[barcodes.str.len() > 0]
    multi = known[known.groupby("barcode")["retailer"].transform("nunique") > 1]
    grp_countries = multi.groupby("barcode")["country"].nunique()
    n_groups = len(grp_countries)
    n_single = int((grp_countries == 1).sum())
    n_cross = int((grp_countries > 1).sum())
    gt = pd.DataFrame({
        "metric": ["multi_retailer_barcode_groups", "single_country_groups", "cross_country_groups",
                   "single_country_share", "cross_country_share"],
        "value": [n_groups, n_single, n_cross, n_single / n_groups, n_cross / n_groups],
    })
    gt.to_csv(RESULTS / "08_act2_ground_truth.csv", index=False)
    print(
        f"wrote 08_act2_ground_truth.csv  "
        f"({n_single}/{n_groups} = {n_single / n_groups:.1%} single-country)"
    )

    # ---- 2. silver cross-country proxy + honesty check + strengthened subset ----
    pos, neg = build_pairs(reps, SEED, 4, 10_000)
    pos_s = cos(pos)
    neg_s = cos(neg)

    cross_pairs = []
    for (b, c), g in reps[reps["brand"].fillna("") != ""].groupby(
        ["brand", "category"], sort=False
    ):
        by_country = {ct: grp.index[0] for ct, grp in g.groupby("country")}
        if len(by_country) < 2:
            continue
        cts = list(by_country)
        cross_pairs.append((by_country[cts[0]], by_country[cts[1]]))
        if len(cross_pairs) >= 3000:
            break
    cross_pairs = np.array(cross_pairs)
    cross_s = cos(cross_pairs)

    va, vb = vol[cross_pairs[:, 0]], vol[cross_pairs[:, 1]]
    both = ~pd.isna(va) & ~pd.isna(vb)
    vol_match = both & (va == vb)
    honesty = float(vol_match.sum() / max(both.sum(), 1))  # "precision" proxy = volume agreement
    proxy = pd.DataFrame({
        "metric": ["proxy_pairs", "proxy_with_volume", "honesty_precision", "strengthened_pairs",
                   "strengthened_median_cosine"],
        "value": [len(cross_pairs), int(both.sum()), round(honesty, 4), int(vol_match.sum()),
                  round(float(np.median(cross_s[vol_match])), 4)],
    })
    proxy.to_csv(RESULTS / "08_act2_proxy.csv", index=False)
    print(f"wrote 08_act2_proxy.csv  (proxy={len(cross_pairs):,}, honesty={honesty:.3%}, "
          f"strengthened={vol_match.sum():,}, median={np.median(cross_s[vol_match]):.3f})")

    # ---- 3. four-population zero-shot scores (cross-border tax ECDF) ----
    a, b = pos[:, 0], pos[:, 1]
    ca, cb = country[a], country[b]
    has = (ca != "") & (cb != "")
    in_country = has & (ca == cb)
    cross_country = has & (ca != cb)
    hard_neg = [(x, y) for x, y in neg if cat[x] == cat[y] and brand[x] != brand[y]]
    hard_neg = np.array(hard_neg[:3000])
    hard_s = cos(hard_neg)

    rows = []
    rows += [("in_country_pos", s) for s in pos_s[in_country]]
    rows += [("cross_country_pos", s) for s in pos_s[cross_country]]
    rows += [("hard_neg", s) for s in hard_s]
    rows += [("random_neg", s) for s in neg_s]
    four = pd.DataFrame(rows, columns=["population", "cosine"])
    four.to_csv(RESULTS / "08_act2_four_pop.csv", index=False)
    print(
        f"wrote 08_act2_four_pop.csv  ({len(four):,} rows: "
        f"{int(in_country.sum()):,} in-country pos, "
        f"{int(cross_country.sum()):,} cross-country pos)"
    )

    # ---- 4. Fisher's exact on entity-level FN rates (the cross-border-tax caveat) ----
    thr = float(np.quantile(neg_s, 0.95))
    fn_mask = pos_s < thr
    bc_arr = reps["barcode"].fillna("").astype(str).to_numpy()

    def _n_groups(mask):
        return int(np.unique(bc_arr[pos[mask][:, 0]]).size)

    cc_g, cc_fg = _n_groups(cross_country), _n_groups(cross_country & fn_mask)
    ic_g, ic_fg = _n_groups(in_country), _n_groups(in_country & fn_mask)
    _, fisher_p = fisher_exact(
        [[cc_fg, cc_g - cc_fg], [ic_fg, ic_g - ic_fg]], alternative="two-sided",
    )
    fisher = pd.DataFrame({
        "metric": ["cross_country_groups", "cross_country_fn_groups", "in_country_groups",
                   "in_country_fn_groups", "fisher_p"],
        "value": [cc_g, cc_fg, ic_g, ic_fg, round(float(fisher_p), 4)],
    })
    fisher.to_csv(RESULTS / "08_act2_fisher.csv", index=False)
    print(
        f"wrote 08_act2_fisher.csv  "
        f"(cross FN {cc_fg}/{cc_g}, in FN {ic_fg}/{ic_g}, Fisher p={fisher_p:.4f})"
    )


if __name__ == "__main__":
    main()
