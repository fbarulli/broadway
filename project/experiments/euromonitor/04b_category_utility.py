"""04b: Category Feature Utility Audit — strict 24 vs macro buckets for blocking.

Before step 03 blocks on `category`, prove (not assume) whether the strict 24
Euromonitor categories are the right blocking granularity. Three tests on the
ground-truth multi-retailer barcode groups (same barcode = true match):

  T1 Hierarchy/Blocking: among TRUE pairs, what fraction share the strict
     category vs the macro category? If strict > 95% keep it; if strict < 85%
     and macro > 95%, block on macro instead.
  T2 Mutual Information: MI(same_brand | match), MI(same_strict | match),
     MI(same_macro | match) on the labeled pair set — is exact category
     carrying real signal or is it redundant/noisy?
  T3 Retailer leakage: Cramer's V between category and retailer (and macro vs
     retailer). V > 0.4 = retailers use different taxonomies -> macro;
     V < 0.2 = categories are universal -> keep strict.

Macro mapping (auditable, built from the dataset's 24 real categories):
  WATER         all bottled waters (still/carbonated/sparkling/flavoured/functional)
  JUICE         100% juices, juice drinks, nectars, coconut & plant waters
  CARBONATES    cola, orange, other carbonates, tonic/mixers, lemonade/lime
  ENERGY_SPORTS energy drinks, sports drinks
  TEA_COFFEE    still/carbonated RTD tea, kombucha, RTD coffee, Asian speciality
  CONCENTRATES  liquid + powder concentrates (the powder one is weight-sold)

Outputs (RESULTS = project/experiments/results/euromonitor/):
  04b_macro_mapping.csv    24 categories -> macro (display table)
  04b_category_audit.csv   the three tests' metrics + verdicts (display table)
  04b_category_audit.png   match rates + MI bars (data-derived limits)
"""

from itertools import combinations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import RESULTS, load_dataset
from _text import MACRO_MAP
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score

CSV_MAPPING = RESULTS / "04b_macro_mapping.csv"
CSV_AUDIT = RESULTS / "04b_category_audit.csv"
PNG_AUDIT = RESULTS / "04b_category_audit.png"

SEED = 42
MAX_POS_PAIRS_PER_GROUP = 4
N_NEG_PAIRS = 10_000


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramer's V for two nominal series (0 = independent)."""
    table = pd.crosstab(x, y)
    chi2 = chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return float(np.sqrt(phi2corr / min(kcorr - 1, rcorr - 1)))


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    df["macro_category"] = df["category"].map(MACRO_MAP)
    rng = np.random.default_rng(SEED)

    # ---- label pairs: positive = same barcode, negative = cross barcode ------
    # NOTE: keep the ORIGINAL df index (no reset_index!) so group indices map
    # straight back into df.loc for brand/category reads.
    barcodes = df["barcode"].fillna("").astype(str)
    titles = df["title"].fillna("").astype(str)
    known = df[barcodes.str.len() > 0]
    multi = known[known.groupby("barcode")["retailer"].transform("nunique") > 1]
    pos = []
    for _, g in multi.groupby("barcode"):
        idx = g.index.tolist()
        combos = list(combinations(idx, 2))
        if len(combos) > MAX_POS_PAIRS_PER_GROUP:
            chosen = rng.choice(len(combos), MAX_POS_PAIRS_PER_GROUP, replace=False)
            combos = [combos[k] for k in chosen]
        pos.extend(combos)
    pos = [(int(a), int(b)) for a, b in pos]

    neg, attempts = [], 0
    all_idx = df.index.tolist()
    while len(neg) < N_NEG_PAIRS and attempts < N_NEG_PAIRS * 60:
        attempts += 1
        a, b = rng.choice(all_idx, 2, replace=False)
        if (barcodes.iloc[a] and barcodes.iloc[a] != barcodes.iloc[b]
                and titles.iloc[a] != titles.iloc[b]):
            neg.append((int(a), int(b)))

    rows = []
    for a, b in pos:
        rows.append({
            "is_true_match": True,
            "same_brand": df.loc[a, "brand"] == df.loc[b, "brand"],
            "same_strict": df.loc[a, "category"] == df.loc[b, "category"],
            "same_macro": df.loc[a, "macro_category"] == df.loc[b, "macro_category"],
        })
    for a, b in neg:
        rows.append({
            "is_true_match": False,
            "same_brand": df.loc[a, "brand"] == df.loc[b, "brand"],
            "same_strict": df.loc[a, "category"] == df.loc[b, "category"],
            "same_macro": df.loc[a, "macro_category"] == df.loc[b, "macro_category"],
        })
    pairs = pd.DataFrame(rows)

    # ---- T1: hierarchy / blocking test ---------------------------------------
    true_pairs = pairs[pairs["is_true_match"]]
    strict_match_rate = float(true_pairs["same_strict"].mean())
    macro_match_rate = float(true_pairs["same_macro"].mean())
    neg_strict = float(pairs[~pairs["is_true_match"]]["same_strict"].mean())
    neg_macro = float(pairs[~pairs["is_true_match"]]["same_macro"].mean())

    # ---- T2: mutual information ------------------------------------------------
    mi_brand = float(mutual_info_score(pairs["is_true_match"], pairs["same_brand"]))
    mi_strict = float(mutual_info_score(pairs["is_true_match"], pairs["same_strict"]))
    mi_macro = float(mutual_info_score(pairs["is_true_match"], pairs["same_macro"]))

    # ---- T3: retailer leakage ---------------------------------------------------
    v_strict = cramers_v(df["category"], df["retailer"])
    v_macro = cramers_v(df["macro_category"], df["retailer"])

    # ---- decisions (the mentor's thresholds) -------------------------------------
    def verdict_t1() -> str:
        if strict_match_rate > 0.95:
            return "KEEP strict: retailers categorize consistently (>95%)"
        if strict_match_rate < 0.85 and macro_match_rate > 0.95:
            return "IMPROVE: block on macro to recover lost matches"
        return "MIXED: no clear winner — inspect per-group diffs"

    def verdict_t2() -> str:
        if mi_strict >= 0.8 * mi_brand:
            return "KEEP strict: exact category carries real signal"
        if mi_strict < 0.05 and mi_macro > 0.05:
            return "IMPROVE: strict is noise, macro holds the signal"
        return "MIXED: category adds little beyond brand — weigh against leakage"

    def verdict_t3() -> str:
        if v_strict > 0.4:
            return "IMPROVE: high leakage — retailers use different taxonomies"
        if v_strict < 0.2:
            return "KEEP strict: categories are universal across retailers"
        return "MODERATE leakage — consider macro if T1/T2 also favor it"

    audit = pd.DataFrame({
        "metric": [
            "T1_strict_match_rate_true_pairs",
            "T1_macro_match_rate_true_pairs",
            "T1_strict_match_rate_neg_pairs",
            "T1_macro_match_rate_neg_pairs",
            "T2_MI_brand_vs_match",
            "T2_MI_strict_vs_match",
            "T2_MI_macro_vs_match",
            "T3_cramers_v_category_vs_retailer",
            "T3_cramers_v_macro_vs_retailer",
            "n_true_pairs", "n_neg_pairs",
        ],
        "value": [
            round(strict_match_rate, 4), round(macro_match_rate, 4),
            round(neg_strict, 4), round(neg_macro, 4),
            round(mi_brand, 4), round(mi_strict, 4), round(mi_macro, 4),
            round(v_strict, 4), round(v_macro, 4),
            len(pos), len(neg),
        ],
    })
    audit = pd.concat([audit, pd.DataFrame({
        "metric": ["T1_decision", "T2_decision", "T3_decision"],
        "value": [verdict_t1(), verdict_t2(), verdict_t3()],
    })], ignore_index=True)
    audit.to_csv(CSV_AUDIT, index=False)
    print(f"wrote {CSV_AUDIT} (display table, {len(audit)} rows)")

    mapping = pd.DataFrame(
        [{"category": c, "macro_category": m} for c, m in sorted(MACRO_MAP.items())])
    mapping["rows"] = mapping["category"].map(df["category"].value_counts()).astype(int)
    mapping.to_csv(CSV_MAPPING, index=False)
    print(f"wrote {CSV_MAPPING} (display table, {len(mapping)} rows)")

    # ---- figure -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    bars = axes[0].bar(
        ["strict\n(true pairs)", "macro\n(true pairs)", "strict\n(neg pairs)",
         "macro\n(neg pairs)"],
        [strict_match_rate * 100, macro_match_rate * 100,
         neg_strict * 100, neg_macro * 100],
        color=["#4C72B0", "#4C72B0", "#BBBBBB", "#BBBBBB"], width=0.55)
    axes[0].bar_label(bars, fmt="%.1f%%", fontsize=9)
    axes[0].set_ylim(0, min(max(strict_match_rate, macro_match_rate) * 110, 100))
    axes[0].set_title("T1: category match within true pairs")
    axes[0].set_ylabel("% of pairs sharing the value")
    bars = axes[1].bar(
        ["brand", "category\n(strict)", "macro"],
        [mi_brand, mi_strict, mi_macro], color=["#4C72B0", "#C44E52", "#4C72B0"],
        width=0.5)
    axes[1].bar_label(bars, fmt="%.3f", fontsize=9)
    axes[1].set_ylim(0, max(mi_brand, mi_strict, mi_macro) * 1.3)
    axes[1].set_title("T2: MI with true-match label")
    axes[1].set_ylabel("mutual information (bits)")
    fig.suptitle(f"Category utility audit — T3 Cramer's V: strict {v_strict:.3f} "
                 f"vs macro {v_macro:.3f} (vs retailer)", fontsize=11)
    fig.savefig(PNG_AUDIT, dpi=150)
    plt.close(fig)
    print(f"wrote {PNG_AUDIT}")

    # ---- printed report -------------------------------------------------------------
    print(f"\nT1 hierarchy (true pairs, n={len(pos):,}):")
    print(f"  strict category match: {strict_match_rate:.1%}   "
          f"macro match: {macro_match_rate:.1%}")
    print(f"  chance baseline (neg pairs): strict {neg_strict:.1%}  "
          f"macro {neg_macro:.1%}")
    print(f"  -> {verdict_t1()}")
    print("\nT2 mutual information vs match label:")
    print(f"  brand {mi_brand:.4f} | strict {mi_strict:.4f} | macro {mi_macro:.4f}")
    print(f"  -> {verdict_t2()}")
    print("\nT3 Cramer's V with retailer:")
    print(f"  strict {v_strict:.3f} | macro {v_macro:.3f}  -> {verdict_t3()}")


if __name__ == "__main__":
    main()
