"""02: Volume & text normalization for euromonitor SKUs.

Ground-truth lesson from 01: the ML-extracted `attributes.Volume` field is
unreliable (200 vs 325, 473 vs 710 observed on the same BARCODE). So volume is
RE-DERIVED from raw text (title + description) via regex and
converted to a canonical ml value, then cross-checked against the attributes
field. Disagreement is kept as a feature, not silently resolved.

Validation: on the 7,040 multi-retailer BARCODE groups, canonical volume must
agree within each true BARCODE group (near-100% expected); disagreeing groups are
flagged for manual inspection (mislabelled BARCODE vs genuine pack-size variant).

Writes (RESULTS = project/experiments/results/euromonitor/):
  02_volume_normalize.csv  display table (metric summary only, per repo convention)
  02_volume_agreement_before_after.csv  the 'wrong' row-level metric, before vs after
  02_volume_disagreement.png  bar of disagreement classes
  02_flavor_vocab.png         flavor coverage from name vs attributes
  02_volume_agreement_before_after.png  before/after of the row-level metric + validation
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from _common import RESULTS, load_euromonitor

CSV_PER_SKU = RESULTS / "02_volume_normalize.csv"
CSV_BEFORE_AFTER = RESULTS / "02_volume_agreement_before_after.csv"
PNG_DISAGREE = RESULTS / "02_volume_disagreement.png"
PNG_FLAVOR = RESULTS / "02_flavor_vocab.png"
PNG_BEFORE_AFTER = RESULTS / "02_volume_agreement_before_after.png"

# --- volume extraction -----------------------------------------------------
# --- volume/pack/flavor text machinery: single source in _text.py -------------
from _text import (
    extract_volume_ml,
    flavor_from_attributes,
    flavor_from_name,
    parse_attributes_volume,
)

# --- BEFORE/AFTER reference: the "wrong" metric measured pre-fixes ----------
# row_level_volume_agreement = regex-derived vs ML-extracted attributes volume,
# SKU-by-SKU across the whole dataset. With the ORIGINAL extractor (v1,
# description-fallback active) this read 0.5787 — noisy because the ML
# attributes field itself is unreliable AND the description fallback injected
# nutrition/serving prose. Fixed extractor (v2+ name-only, thousands-sep,
# pack-strip) re-measures the same metric live below. Kept as an explicit
# before/after so a reviewer can see the improvement, not just the final number.
LEGACY_ROW_LEVEL_AGREEMENT = 0.5787  # measured 2026-09-01, extractor v1+desc fallback
LEGACY_WITHIN_BARCODE_AGREEMENT = 0.9653  # same vintage, within-BARCODE on 7,040 groups
LEGACY_FLAGGED_GROUPS = 198  # same vintage


def plot_disagreement(agree_df: pd.DataFrame, out_path: Path, ratio_threshold: float = 1.05) -> None:
    """Within-BARCODE volume-disagreement ratios (the validated metric, not row-level).

    Plots the group-level max/min volume ratio for the flagged groups, with the
    near-miss / likely-variant threshold marked. Uses the consolidated `agree`
    table so the chart matches the 54/199 number cited in the write-up.
    """
    flagged = agree_df[agree_df["canonical_agree"] == False].copy()
    flagged["vol_ratio"] = flagged["canonical_volumes"].apply(
        lambda v: max(v) / min(v) if len(v) > 1 and min(v) > 0 else 1)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if len(flagged):
        sns.histplot(flagged["vol_ratio"], bins=30, ax=ax, color="#C44E52")
    ax.axvline(ratio_threshold, color="black", linestyle="--",
               label=f"near-miss threshold ({ratio_threshold})")
    n_near = int((flagged["vol_ratio"] < ratio_threshold).sum())
    n_real = int((flagged["vol_ratio"] >= ratio_threshold).sum())
    if len(flagged):
        ax.set_xlim(0, min(flagged["vol_ratio"].max() * 1.05, 100))
    ax.set_xlabel("max(volume) / min(volume) within BARCODE group")
    ax.set_title(
        f"Within-BARCODE volume disagreement (n={len(flagged)}): "
        f"{n_near} near-miss, {n_real} likely variant/error")
    ax.legend()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_flavor(df: pd.DataFrame, out_path: Path) -> None:
    """Flavor detection-source overlap + agreement when both present.

    Answers "should I trust name-derived flavor over attributes-derived flavor"
    (the step-03 feature-weighting decision) via the both/name-only/attr-only/
    neither split and the agreement rate when both fire.
    """
    known = df[df["barcode"].fillna("").str.len() > 0].copy()
    known["has_name"] = known["flavor_from_name"].notna()
    known["has_attr"] = known["flavor_from_attributes"].notna()
    both = known[known["has_name"] & known["has_attr"]]
    agree_rate = (
        (both["flavor_from_name"] == both["flavor_from_attributes"]).mean()
        if len(both) else float("nan")
    )
    categories = {
        "both agree": int((both["flavor_from_name"] == both["flavor_from_attributes"]).sum()),
        "both, disagree": int((both["flavor_from_name"] != both["flavor_from_attributes"]).sum()),
        "name only": int((known["has_name"] & ~known["has_attr"]).sum()),
        "attributes only": int((~known["has_name"] & known["has_attr"]).sum()),
        "neither": int((~known["has_name"] & ~known["has_attr"]).sum()),
    }
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    sns.barplot(
        x=list(categories.values()), y=list(categories.keys()), ax=ax, palette="muted")
    ax.set_xlim(0, max(categories.values()) * 1.1)
    ax.set_xlabel("BARCODE-known SKUs")
    ax.set_title(f"Flavor detection source (agreement when both present: {agree_rate:.1%})")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_before_after(row_level_after: float, within_barcode_after: float,
                      flagged_after: int, out_path: Path) -> None:
    """The 'wrong' metric (row-level regex-vs-attributes agreement), before vs after.

    All our fixes are the 'after' column: extractor v2 (name-only, thousands-
    separator, pack-strip, ambiguous-oz flag) versus the original v1 with
    description fallback. Three panels so the improvement is visible on the
    metric that used to look bad AND on the two numbers that actually validate
    the extractor. Explicit data-derived limits (repo convention): agreement
    axes span [min-0.05, max+0.03] clipped to [0, 1]; flagged axis tops at
    max*1.15.
    """
    labels = ["before\n(v1 + desc fallback)", "after\n(v2+, name-only)"]
    panels = [
        (LEGACY_ROW_LEVEL_AGREEMENT, row_level_after, "{:.1%}",
         "row-level agreement\nthe 'wrong' metric"),
        (LEGACY_WITHIN_BARCODE_AGREEMENT, within_barcode_after, "{:.1%}",
         "within-BARCODE agreement\n(ground-truth validation)"),
        (float(LEGACY_FLAGGED_GROUPS), float(flagged_after), "{:d}",
         "flagged disagreeing groups"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4), constrained_layout=True)
    x = np.arange(2)
    for ax, (before, after, fmt, title) in zip(axes, panels):
        values = [before, after]
        bars = ax.bar(x, values, color=["#C44E52", "#4C72B0"], width=0.55)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_title(title, fontsize=9)
        if fmt == "{:d}":
            ax.bar_label(bars, labels=[f"{int(v):d}" for v in values], fontsize=9)
        else:
            ax.bar_label(bars, fmt=fmt, fontsize=9)
        if fmt == "{:d}":
            ax.set_ylim(0, max(values) * 1.15)
        else:
            lo = max(min(values) - 0.05, 0.0)
            hi = min(max(values) + 0.03, 1.0)
            ax.set_ylim(lo, hi)
    fig.suptitle("Volume-extractor fixes: before/after (02_volume_normalize)",
                 fontsize=11)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_euromonitor()

    # Canonical volume comes from title ONLY (the most normalized text
    # source). Description is NOT used as a fallback: its nutrition/serving/
    # dilution prose ("per 12 fl oz", "0.2 l glass", "dilute in 9 volumes")
    # injects systematic false volumes (proven: fallback agreement 96.5% and
    # 198 flagged vs 99.0% and 54 flagged name-only). Description volume is
    # kept as an explicit low-confidence feature, never silently folded in.
    # extract_volume_ml returns (ml, ambiguous) — split into the canonical
    # column + an ambiguity flag (bare oz/ounce could be weight).
    _vol = df["title"].map(extract_volume_ml)
    df["canonical_volume_ml"] = _vol.map(lambda x: x[0])
    df["canonical_volume_ambiguous"] = _vol.map(lambda x: x[1])
    df["desc_volume_ml"] = df["description"].map(
        lambda s: extract_volume_ml(s)[0])
    df["attributes_volume_ml"] = df["attributes"].map(parse_attributes_volume)
    df["volume_agreement"] = (
        df["canonical_volume_ml"].notna()
        & df["attributes_volume_ml"].notna()
        & (df["canonical_volume_ml"] == df["attributes_volume_ml"])
    )
    df["flavor_from_name"] = df["title"].map(flavor_from_name)
    df["flavor_from_attributes"] = df["attributes"].map(flavor_from_attributes)

    # DISPLAY TABLE ONLY (repo convention): compact describe-style summary.
    # Two DISTINCT agreement metrics, labeled explicitly so they can't be
    # conflated:
    #   A. row_level_volume_agreement — regex vs ML-extracted attributes, ALL
    #      SKUs dataset-wide. Expected lower (attributes field is noisy; a
    #      mismatch may mean the regex is right and the attributes is wrong).
    #   B. within_barcode_volume_agreement — canonical volume consistent within
    #      each multi-retailer BARCODE group (ground-truth validation; the
    #      number that validates the extractor).
    summary = pd.DataFrame({
        "metric": [
            "skus_total", "skus_with_barcode",
            "canonical_volume_detected", "attributes_volume_detected",
            "row_level_volume_agreement",  # A (dataset-wide, regex vs attributes)
            "within_barcode_volume_agreement",  # B (multi-retailer BARCODE groups, ground truth)
            "flavor_from_name_detected", "flavor_from_attributes_detected",
        ],
        "value": [
            len(df),
            int(df["barcode"].fillna("").str.len().gt(0).sum()),
            int(df["canonical_volume_ml"].notna().sum()),
            int(df["attributes_volume_ml"].notna().sum()),
            round(df["volume_agreement"].mean(), 4),
            None,  # filled from consolidated agree table below
            round(df["flavor_from_name"].notna().mean(), 4),
            round(df["flavor_from_attributes"].notna().mean(), 4),
        ],
    })
    summary.to_csv(CSV_PER_SKU, index=False)
    print(f"wrote {CSV_PER_SKU} (display table, {len(summary)} rows)")

    out = df  # working frame; per-SKU columns live here, not persisted to CSV

    # ---- BARCODE-group validation on multi-retailer subset --------------------
    # Computed ONCE; summary stat + printed sample both derive from `agree`.
    # Honest denominators: a group counts toward canonical_agree only if it has
    # >=1 detected canonical volume (empty groups are excluded, not counted as
    # trivially-agreeing). attr_agree mirrors the same rule for the attributes
    # field, giving the ML-vs-regex contrast point.
    known = out[out["barcode"].fillna("").str.len() > 0]
    multi = known[known.groupby("barcode")["retailer"].transform("nunique") > 1]
    agree = multi.groupby("barcode").apply(
        lambda x: pd.Series({
            "retailers": x["retailer"].nunique(),
            "skus": len(x),
            "canonical_volumes": sorted(x["canonical_volume_ml"].dropna().unique().tolist()),
            "attr_volumes": sorted(x["attributes_volume_ml"].dropna().unique().tolist()),
            "canonical_agree": (
                x["canonical_volume_ml"].dropna().nunique() <= 1
                if x["canonical_volume_ml"].notna().any() else None
            ),
            "attr_agree": (
                x["attributes_volume_ml"].dropna().nunique() <= 1
                if x["attributes_volume_ml"].notna().any() else None
            ),
        }),
    ).reset_index()
    with_canon = agree.dropna(subset=["canonical_agree"])
    canonical_rate = with_canon["canonical_agree"].mean()
    with_attr = agree.dropna(subset=["attr_agree"])
    attr_rate = with_attr["attr_agree"].mean()
    flagged = agree[agree["canonical_agree"] == False]

    # ---- backfill the summary table with the consolidated numbers ----------
    summary.loc[summary["metric"] == "within_barcode_volume_agreement", "value"] = (
        round(float(canonical_rate), 4))
    summary = pd.concat([summary, pd.DataFrame({
        "metric": ["within_barcode_attributes_agreement"],
        "value": [round(float(attr_rate), 4)],
    })], ignore_index=True)
    summary.to_csv(CSV_PER_SKU, index=False)
    print(f"wrote {CSV_PER_SKU} (display table, {len(summary)} rows)")

    print(f"\nmulti-retailer BARCODE groups with >=1 canonical volume: {len(with_canon):,}")
    print(f"  canonical volume AGREES within BARCODE: {canonical_rate:.1%}")
    print(f"  attributes volume AGREES within BARCODE: {attr_rate:.1%}")
    print(f"  DISAGREEING groups (canonical): {len(flagged):,}")

    # ---- figures -----------------------------------------------------------
    plot_disagreement(agree, PNG_DISAGREE)
    print(f"wrote {PNG_DISAGREE}")
    plot_flavor(out, PNG_FLAVOR)
    print(f"wrote {PNG_FLAVOR}")

    # ---- the 'wrong' metric kept as an explicit before/after ----------------
    # User directive: keep row_level_volume_agreement as a before/after rather
    # than dropping it — same metric, ORIGINAL extractor vs all our fixes.
    row_level_after = float(df["volume_agreement"].mean())
    before_after = pd.DataFrame({
        "metric": [
            "row_level_volume_agreement", "within_barcode_volume_agreement",
            "flagged_disagreeing_groups",
        ],
        "before": [
            LEGACY_ROW_LEVEL_AGREEMENT, LEGACY_WITHIN_BARCODE_AGREEMENT,
            LEGACY_FLAGGED_GROUPS,
        ],
        "after": [
            round(row_level_after, 4), round(float(canonical_rate), 4), len(flagged),
        ],
        "delta_after_minus_before": [
            round(row_level_after - LEGACY_ROW_LEVEL_AGREEMENT, 4),
            round(float(canonical_rate) - LEGACY_WITHIN_BARCODE_AGREEMENT, 4),
            LEGACY_FLAGGED_GROUPS - len(flagged),
        ],
    })
    before_after.to_csv(CSV_BEFORE_AFTER, index=False)
    print(f"wrote {CSV_BEFORE_AFTER} (display table, {len(before_after)} rows)")
    plot_before_after(
        row_level_after=row_level_after,
        within_barcode_after=float(canonical_rate),
        flagged_after=len(flagged),
        out_path=PNG_BEFORE_AFTER,
    )
    print(f"wrote {PNG_BEFORE_AFTER}")

    # sample of flagged groups
    if len(flagged):
        print("\n=== sample of disagreeing BARCODE groups ===")
        for _, r in flagged.head(8).iterrows():
            print(f"BARCODE {r['barcode']}: canonical={r['canonical_volumes']} attr={r['attr_volumes']}")


if __name__ == "__main__":
    main()
