"""01f: Quantify barcode-coverage bias — is it really non-random, and how much?

Tests barcode presence (has_bc) against each feature and reports effect size:
chi-square + Cramer's V + normalized mutual information for categoricals
(country/retailer/category), point-biserial + Mann-Whitney + Cohen's d for
price/volume, and the selection ratio (segment coverage / overall coverage) to
show over/under-representation directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _common import canonical_volume, has_barcode, load_dataset
from _text import MACRO_MAP
from scipy.stats import chi2_contingency, mannwhitneyu, pointbiserialr
from sklearn.metrics import normalized_mutual_info_score


def cramers_v(x, y) -> float:
    table = pd.crosstab(x, y)
    chi2 = chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    return float(np.sqrt((chi2 / n) / max(1, min(k - 1, r - 1))))


def cohens_d(a, b) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n_a, n_b = len(a), len(b)
    pooled = np.sqrt(((n_a - 1) * a.var() + (n_b - 1) * b.var()) / (n_a + n_b - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def main() -> None:
    df = load_dataset()
    df["has_bc"] = has_barcode(df).astype(int)
    df["macro"] = df["category"].fillna("").map(lambda c: MACRO_MAP.get(c, "OTHER"))
    df["vol"] = canonical_volume(df["title"])["canonical_volume_ml"]
    df["price_num"] = pd.to_numeric(df["price"], errors="coerce")

    overall = df["has_bc"].mean()
    print(f"overall barcode coverage: {overall:.1%}\n")

    print("=== categorical features: association with has_bc ===")
    print(f"{'feature':<12}{'chi2 p':>12}{'CramersV':>10}{'NMI':>8}")
    for col in ["country", "retailer", "macro", "category"]:
        x = df["has_bc"].astype(str)
        y = df[col].astype(str)
        _chi2, p, *_ = chi2_contingency(pd.crosstab(x, y))
        v = cramers_v(x, y)
        nmi = normalized_mutual_info_score(df["has_bc"], y, average_method="arithmetic")
        print(f"{col:<12}{p:>12.3e}{v:>10.3f}{nmi:>8.3f}")

    print("\n=== numeric features: has_bc vs no_bc ===")
    for col in ["price_num", "vol"]:
        a = df.loc[df["has_bc"] == 1, col].dropna()
        b = df.loc[df["has_bc"] == 0, col].dropna()
        # value-based association: has_bc (0/1) vs the actual value (rows where
        # the value exists) — consistent with the MWU / Cohen's d below.
        vals = df[col].dropna()
        mask = df.loc[vals.index, "has_bc"]
        r_value = pointbiserialr(mask, vals)[0]
        # presence-based association: has_bc (0/1) vs "is the value present?"
        r_presence = pointbiserialr(df["has_bc"], df[col].notna().astype(int))[0]
        u_p = mannwhitneyu(a, b, alternative="two-sided").pvalue
        d = cohens_d(a, b)
        print(f"{col:<12} r_value={r_value:+.3f}  r_presence={r_presence:+.3f}  "
              f"MWU p={u_p:.3e}  Cohen's d={d:+.3f}")

    print("\n=== selection ratio (segment coverage / overall) — most extreme ===")
    for col in ["country", "macro"]:
        ratio = (df.groupby(col)["has_bc"].mean() / overall).round(2)
        ratio = ratio.sort_values()
        print(f"\n{col} (over-represented top, under-represented bottom):")
        print(pd.concat([ratio.tail(3), ratio.head(3)]).to_string())


if __name__ == "__main__":
    main()
