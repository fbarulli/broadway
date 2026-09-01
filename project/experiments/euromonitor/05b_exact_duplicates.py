"""05b: Exact duplicates — identify them before any fuzzy matching.

Exact duplicates are the cheapest wins in entity resolution (string equality,
no model needed) and must be ID'd first: they are dedupe candidates within a
retailer, and trivially-easy TRUE matches across retailers.

Because product_id is 100% unique, NO two rows are identical on all 13
columns (verified below) — "exact" is therefore defined on practical keys:

  full_row             all 13 columns (expect 0 by construction)
  retailer+title       the same listing text scraped twice at one retailer
  retailer+title+price stricter listing identity (same text AND same price)
  retailer+barcode     the same product listed multiple times at one retailer
  title+brand          identical product title across retailers (the easy
                       positives for entity resolution) — and we check whether
                       those groups AGREE on barcode (mislabeled-barcode probe)

Outputs (RESULTS = project/experiments/results/euromonitor/):
  05b_exact_duplicates.csv    duplicate summary per key (display table)
  05b_duplicate_examples.csv  top duplicate groups with sample titles (display)
  05b_exact_duplicates.png    duplicate rows per key (data-derived limits)
"""


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from _common import RESULTS, load_dataset

CSV_SUMMARY = RESULTS / "05b_exact_duplicates.csv"
CSV_EXAMPLES = RESULTS / "05b_duplicate_examples.csv"
PNG_DUP = RESULTS / "05b_exact_duplicates.png"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    n = len(df)
    summary: list[dict] = []

    def add(name: str, groups: int, dup_rows: int, cross_retailer_groups: int | None) -> None:
        summary.append({
            "key": name, "duplicate_groups": groups, "duplicate_rows": dup_rows,
            "share_of_rows": round(dup_rows / n, 4),
            "cross_retailer_groups": cross_retailer_groups,
        })
        print(f"  {name:<26} groups={groups:>6,}  rows={dup_rows:>7,}  "
              f"({dup_rows / n:.1%})"
              + (f"  cross-retailer={cross_retailer_groups:,}" if cross_retailer_groups is not None else ""))

    print(f"exact-duplicate scan ({n:,} rows):")

    # full row: expect 0 (product_id unique)
    full_groups = int(df.duplicated(keep=False).sum())
    print(f"  full_row (13 cols)            groups=0        rows={full_groups:>7,}  "
          f"({full_groups / n:.1%}) — by construction (product_id unique)")
    summary.append({"key": "full_row (13 cols)", "duplicate_groups": 0,
                    "duplicate_rows": full_groups, "share_of_rows": 0.0,
                    "cross_retailer_groups": None})

    # retailer + title
    sizes = df.groupby(["retailer", "title"], dropna=False).size()
    g = sizes[sizes >= 2]
    add("retailer+title", len(g), int(g.sum()), None)

    # retailer + title + price
    sizes = df.groupby(["retailer", "title", "price"], dropna=False).size()
    g = sizes[sizes >= 2]
    add("retailer+title+price", len(g), int(g.sum()), None)

    # retailer + barcode (same product listed multiple times at one retailer)
    has_bc = df["barcode"].fillna("").str.len() > 0
    sizes = df[has_bc].groupby(["retailer", "barcode"], dropna=False).size()
    g = sizes[sizes >= 2]
    add("retailer+barcode", len(g), int(g.sum()), None)

    # title + brand across retailers + barcode agreement probe
    sizes = df.groupby(["title", "brand"], dropna=False).size()
    g = sizes[sizes >= 2]
    cross = 0
    barcode_conflict = 0
    for (title, brand), sz in g.items():
        sub = df[(df["title"] == title) & (df["brand"] == brand)]
        n_ret = sub["retailer"].nunique()
        if n_ret > 1:
            cross += 1
            # conflict = >1 UNIQUE real barcode among the CROSS-retailer rows
            # (empty strings are NOT a barcode; counting them inflated the old
            # probe to 887 — corrected, matches 06b's definition -> 339).
            real = sub["barcode"].fillna("")
            real = real[real.str.len() > 0]
            if real.nunique() > 1:
                barcode_conflict += 1
    add("title+brand", len(g), int(g.sum()), cross)
    print(f"    ...of which cross-retailer groups: {cross:,}")
    print(f"    ...cross-retailer groups with CONFLICTING barcodes: "
          f"{barcode_conflict:,} (mislabeled-barcode probe)")

    # ---- examples -----------------------------------------------------------
    examples = []
    for key_name, cols in [("retailer+title", ["retailer", "title"]),
                           ("title+brand", ["title", "brand"])]:
        clean = df.dropna(subset=cols)  # NaN keys break MultiIndex nlargest
        sizes = clean.groupby(cols).size()
        top = sizes[sizes >= 2].nlargest(5)
        for k, sz in top.items():
            sub = clean
            for col, val in zip(cols, k):
                sub = sub[sub[col] == val]
            retailers = sorted(sub["retailer"].unique().tolist())[:4]
            barcodes = sorted(sub["barcode"].fillna("").unique().tolist())[:3]
            examples.append({
                "key": key_name, "group_size": int(sz),
                "sample_title": str(k[1])[:70] if len(cols) == 2 else str(k)[:70],
                "retailers": "; ".join(retailers),
                "barcodes": "; ".join(str(b) for b in barcodes),
            })
    pd.DataFrame(examples).to_csv(CSV_EXAMPLES, index=False)
    print(f"\nwrote {CSV_EXAMPLES} (display table, {len(examples)} rows)")

    # ---- write summary + plot ---------------------------------------------------
    frame = pd.DataFrame(summary)
    frame.to_csv(CSV_SUMMARY, index=False)
    print(f"wrote {CSV_SUMMARY} (display table, {len(frame)} rows)")

    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    sub = frame[frame["key"] != "full_row (13 cols)"]
    bars = ax.bar(sub["key"], sub["duplicate_rows"],
                  color="#4C72B0", width=0.55)
    ax.bar_label(bars, fmt="{:,.0f}", fontsize=9)
    ax.set_ylim(0, int(sub["duplicate_rows"].max()) * 1.15)  # data-derived
    ax.set_ylabel("rows in duplicate groups")
    ax.set_title(f"Exact duplicates by key definition ({n:,} rows total)")
    fig.savefig(PNG_DUP, dpi=150)
    plt.close(fig)
    print(f"wrote {PNG_DUP}")

    # ---- verdict ---------------------------------------------------------------
    print(f"\nverdict: {frame.iloc[1]['duplicate_groups']:,.0f} exact "
          f"retailer+title duplicates are dedupe candidates before matching; "
          f"{frame.iloc[4]['cross_retailer_groups']:,.0f} exact-title "
          f"cross-retailer groups are free true matches for step 03.")


if __name__ == "__main__":
    main()
