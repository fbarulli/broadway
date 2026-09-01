"""01: Euromonitor dataset EDA — coverage, completeness, noise, attributes.

Follows the repo experiment convention: writes `01_eda_describe.csv` and
`01_eda.png` under RESULTS (project/experiments/results/euromonitor/), plus
the GTIN-coverage breakdown CSV and the attribute-key table CSV. Every figure
is saved as PNG, headless (Agg), dpi 150.
"""

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from _common import RESULTS, load_euromonitor

CSV_DESCRIBE = RESULTS / "01_eda_describe.csv"
CSV_GTIN = RESULTS / "01_gtin_coverage.csv"
CSV_ATTR_KEYS = RESULTS / "01_attribute_keys.csv"
PNG_GTIN = RESULTS / "01_gtin_coverage.png"
PNG_RETAILER = RESULTS / "01_retailer_country.png"
PNG_ATTR = RESULTS / "01_attribute_keys.png"

# Columns with long free text (excluded from describe table).
TEXT_COLS = [
    "sku_name_eng", "description_short_eng", "breadcrumbs_eng",
    "sku_url", "image_url", "attribute",
]


def parse_attribute_keys(series: pd.Series, limit: int = 5000) -> Counter:
    """Extract `Key:` names from the ';'-delimited attribute strings."""
    keys: Counter = Counter()
    for value in series.fillna("").head(limit):
        for part in value.split(";"):
            part = part.strip()
            if ":" in part:
                keys[part.split(":", 1)[0].strip()] += 1
    return keys


def plot_gtin_coverage(df: pd.DataFrame, out_path: Path) -> None:
    """SKU-per-GTIN distribution split by retailer multiplicity (log-y).

    Shows which GTINs give real cross-retailer matching signal (multi-
    retailer) vs single-retailer ones that only help validation.
    """
    known = df.loc[df["gtin"].fillna("").str.len() > 0]
    per_retailer_count = known.groupby("gtin")["retailer"].nunique()
    sku_count = known.groupby("gtin").size()
    multi = per_retailer_count > 1

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.hist(
        sku_count[~multi], bins=range(1, 46), alpha=0.7,
        label=f"single-retailer ({(~multi).sum():,} GTINs)", color="#999",
    )
    ax.hist(
        sku_count[multi], bins=range(1, 46), alpha=0.7,
        label=f"multi-retailer ({multi.sum():,} GTINs)", color="#4C72B0",
    )
    ax.set_yscale("log")
    ax.set_xlabel("SKUs per GTIN")
    ax.set_ylabel("GTIN count (log)")
    ax.set_title(f"GTIN multiplicity — {multi.mean():.1%} cross-retailer")
    ax.legend()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_retailer_country(df: pd.DataFrame, out_path: Path) -> None:
    """Top retailers by SKU count — where the catalog lives."""
    top = df["retailer"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    sns.barplot(x=top.values, y=top.index, ax=ax, palette="viridis")
    ax.set_xlabel("SKU count")
    ax.set_title("Top 15 retailers by SKU count")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_attribute_keys(keys: Counter, out_path: Path) -> None:
    """Most frequent attribute key names across retailers."""
    top = pd.Series(dict(keys.most_common(15)))
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    sns.barplot(x=top.values, y=top.index, ax=ax, palette="crest")
    ax.set_xlabel("occurrences (first 5k rows)")
    ax.set_title("Top attribute keys")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_euromonitor()
    print(f"rows: {len(df):,}  cols: {len(df.columns)}")
    print(f"retailers: {df['retailer'].nunique():,}  countries: {df['country'].nunique()}")

    # ---- describe CSV (numeric-ish columns; raw strings so counts are honest)
    numeric_cols = [c for c in df.columns if c not in TEXT_COLS]
    desc = df[numeric_cols].describe(include="all").T
    desc.to_csv(CSV_DESCRIBE)
    print(f"wrote {CSV_DESCRIBE}")

    # ---- GTIN coverage CSV
    known = df["gtin"].fillna("").str.len() > 0
    per = df.loc[known, "gtin"].value_counts()
    multi_by_gtin = df.loc[known].groupby("gtin")["retailer"].nunique()
    # DISPLAY TABLE ONLY: top-25 GTINs + aggregate summary, never the full 15K list.
    gtin_frame = pd.DataFrame({"gtin": per.index, "skus": per.values}).sort_values("skus", ascending=False)
    top = gtin_frame.head(25).copy()
    top.loc[len(top)] = ["TOTAL_GTINS", len(gtin_frame)]
    top.loc[len(top)] = ["MULTI_RETAILER_GTINS", int((multi_by_gtin > 1).sum())]
    top.loc[len(top)] = ["MEAN_SKUS_PER_GTIN", round(per.mean(), 2)]
    top.to_csv(CSV_GTIN, index=False)
    print(f"wrote {CSV_GTIN} (display table, {len(top)} rows)")

    # ---- attribute keys CSV
    keys = parse_attribute_keys(df["attribute"])
    pd.DataFrame(keys.most_common(50), columns=["key", "occurrences"]).to_csv(
        CSV_ATTR_KEYS, index=False)
    print(f"wrote {CSV_ATTR_KEYS}")

    # ---- figures
    plot_gtin_coverage(df, PNG_GTIN)
    print(f"wrote {PNG_GTIN}")
    plot_retailer_country(df, PNG_RETAILER)
    print(f"wrote {PNG_RETAILER}")
    plot_attribute_keys(keys, PNG_ATTR)
    print(f"wrote {PNG_ATTR}")


if __name__ == "__main__":
    main()
