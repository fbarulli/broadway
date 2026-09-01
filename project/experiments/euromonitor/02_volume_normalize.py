"""02: Volume & text normalization for euromonitor SKUs.

Ground-truth lesson from 01: the ML-extracted `attribute.Volume` field is
unreliable (200 vs 325, 473 vs 710 observed on the same GTIN). So volume is
RE-DERIVED from raw text (sku_name_eng + description_short_eng) via regex and
converted to a canonical ml value, then cross-checked against the attribute
field. Disagreement is kept as a feature, not silently resolved.

Validation: on the 7,040 multi-retailer GTIN groups, canonical volume must
agree within each true GTIN group (near-100% expected); disagreeing groups are
flagged for manual inspection (mislabelled GTIN vs genuine pack-size variant).

Writes (RESULTS = project/experiments/results/euromonitor/):
  02_volume_normalize.csv  display table (metric summary only, per repo convention)
  02_volume_disagreement.png  bar of disagreement classes
  02_flavor_vocab.png         flavor coverage from name vs attribute
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from _common import RESULTS, load_euromonitor

CSV_PER_SKU = RESULTS / "02_volume_normalize.csv"
PNG_DISAGREE = RESULTS / "02_volume_disagreement.png"
PNG_FLAVOR = RESULTS / "02_flavor_vocab.png"

# --- volume extraction -----------------------------------------------------
# --- volume extraction (v2: broader unit coverage + decimal-comma) --------
# Handles: ml/millilitre(s), l/liter(s)/litre(s)/ltr, cl/centilitre(s),
# dl/decilitre(s), fl oz/fl. oz./floz/fl-oz/fluid ounce(s), oz/ounce(s)
# (AMBIGUOUS — flagged), gal/gallon(s), qt/quart(s), pt/pint(s);
# decimal comma ("1,5 L") and punctuation/spacing tolerance ("16.9FlOz").
# A trailing (?![a-zA-Z]) boundary keeps bare units from matching inside
# unrelated words ("2 lb" is NOT 2 liters; "2 large" is not 2l).
_UNIT_ALTERNATION = r"""
    (?:
        milli\s?lit(?:er|re)s?    |  ml
      | centi\s?lit(?:er|re)s?    |  cl
      | deci\s?lit(?:er|re)s?     |  dl
      | lit(?:er|re)s?            |  ltr | lt | l
      | fluid\s?ounces?           |  fl\.?\s?-?\s?oz\.?  | floz
      | gallons?                  |  gal
      | quarts?                   |  qt
      | pints?                    |  pt
      | ounces?                   |  oz
    )
"""

_VOLUME_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?<![a-zA-Z])" + _UNIT_ALTERNATION + r"(?![a-zA-Z])",
    re.IGNORECASE | re.VERBOSE,
)

_PACK_RE = re.compile(
    r"(\d+)\s*(?:-|\s)?(?:pack|pk|count|ct|x)\b", re.IGNORECASE,
)
_NUTRITION_RE = re.compile(
    r"per\s*(?:100|1)\s*(?:ml|g|gram|grams)|kcal\s*per|per\s*serving",
    re.IGNORECASE,
)

_TO_ML = {
    "ml": 1.0, "millilitre": 1.0, "milliliter": 1.0, "millilitres": 1.0, "milliliters": 1.0,
    "cl": 10.0, "centilitre": 10.0, "centiliter": 10.0, "centilitres": 10.0, "centiliters": 10.0,
    "dl": 100.0, "decilitre": 100.0, "deciliter": 100.0, "decilitres": 100.0, "deciliters": 100.0,
    "l": 1000.0, "ltr": 1000.0, "lt": 1000.0, "litre": 1000.0, "liter": 1000.0, "litres": 1000.0, "liters": 1000.0,
    "gal": 3785.41, "gallon": 3785.41, "gallons": 3785.41,
    "qt": 946.353, "quart": 946.353, "quarts": 946.353,
    "pt": 473.176, "pint": 473.176, "pints": 473.176,
    "fl oz": 29.5735, "fl.oz": 29.5735, "fl-oz": 29.5735, "floz": 29.5735,
    "fluid ounce": 29.5735, "fluid ounces": 29.5735,
    "oz": 29.5735, "ounce": 29.5735, "ounces": 29.5735,
}

# Bare oz/ounce could mean weight (chips, protein powder) rather than fluid
# volume. Flag rather than silently trusting it.
_AMBIGUOUS_UNITS = {"oz", "ounce", "ounces"}

_BUCKET = 5  # nearest 5 ml absorbs 355 vs 355.0 vs 354 noise


def _norm_unit(token: str) -> str:
    t = re.sub(r"[.\-\s]", " ", token.lower()).strip()
    t = re.sub(r"\s+", " ", t)
    if t in {"floz", "fl.oz", "fl-oz"}:
        return "fl oz"
    return t


def extract_volume_ml(text: str) -> tuple[float | None, bool]:
    """Return (canonical_ml, is_ambiguous_unit).

    is_ambiguous_unit=True means the match was a bare oz/ounce with no
    'fl'/'fluid' qualifier — could be weight, not volume. Caller should
    decide whether to trust it (e.g. only within a beverage category).
    Pack-count and nutrition-per-100ml phrases are stripped first.
    """
    if not isinstance(text, str):
        return None, False
    cleaned = _PACK_RE.sub("", text)
    cleaned = _NUTRITION_RE.sub("", cleaned)
    match = _VOLUME_RE.search(cleaned)
    if not match:
        return None, False
    value = float(match.group(1).replace(",", "."))
    unit = _norm_unit(match.group(0)[len(match.group(1)):].strip())
    if unit not in _TO_ML:
        return None, False
    ml = value * _TO_ML[unit]
    ambiguous = unit in _AMBIGUOUS_UNITS
    return round(ml / _BUCKET) * _BUCKET, ambiguous


def parse_attribute_volume(attribute: str) -> float | None:
    """Parse the 'Volume: <n>' segment from the ';'-delimited attribute string."""
    if not isinstance(attribute, str):
        return None
    for part in attribute.split(";"):
        part = part.strip()
        if part.lower().startswith("volume:"):
            value = part.split(":", 1)[1].strip()
            m = re.match(r"(\d+(?:\.\d+)?)", value)
            if m:
                return round(float(m.group(1)) / _BUCKET) * _BUCKET
    return None


# --- flavor vocabulary (light touch, per MISSION plan) ---------------------
FLAVOR_VOCAB = [
    "peach", "raspberry", "lemon", "orange", "strawberry", "grape", "apple",
    "cherry", "mango", "lime", "mixed berry", "berry", "cola", "ginger",
    "vanilla", "chocolate", "coffee", "tea", "root beer", "pineapple",
    "cranberry", "blueberry", "watermelon", "coconut", "kiwi", "caramel",
    "original", "unflavored", "plain", "lemonade", "fruit punch",
]
_FLAVOR_RE = re.compile(
    r"(" + "|".join(re.escape(f) for f in sorted(FLAVOR_VOCAB, key=len, reverse=True)) + r")",
    re.IGNORECASE,
)


def flavor_from_name(name: str) -> str | None:
    if not isinstance(name, str):
        return None
    m = _FLAVOR_RE.search(name.lower())
    return m.group(1).lower() if m else None


def flavor_from_attribute(attribute: str) -> str | None:
    if not isinstance(attribute, str):
        return None
    for part in attribute.split(";"):
        part = part.strip()
        if part.lower().startswith("flavour:") or part.lower().startswith("flavor:"):
            value = part.split(":", 1)[1].strip().lower()
            for flavor in sorted(FLAVOR_VOCAB, key=len, reverse=True):
                if flavor in value:
                    return flavor
    return None


def plot_disagreement(agree_df: pd.DataFrame, out_path: Path, ratio_threshold: float = 1.05) -> None:
    """Within-GTIN volume-disagreement ratios (the validated metric, not row-level).

    Plots the group-level max/min volume ratio for the flagged groups, with the
    near-miss / likely-variant threshold marked. Uses the consolidated `agree`
    table so the chart matches the 54/199 number cited in the write-up.
    """
    flagged = agree_df[agree_df["canonical_agree"] == False].copy()  # noqa: E712
    flagged["vol_ratio"] = flagged["canonical_volumes"].apply(
        lambda v: max(v) / min(v) if len(v) > 1 and min(v) > 0 else 1)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if len(flagged):
        sns.histplot(flagged["vol_ratio"], bins=30, ax=ax, color="#C44E52")
    ax.axvline(ratio_threshold, color="black", linestyle="--",
               label=f"near-miss threshold ({ratio_threshold})")
    n_near = int((flagged["vol_ratio"] < ratio_threshold).sum())
    n_real = int((flagged["vol_ratio"] >= ratio_threshold).sum())
    ax.set_xlabel("max(volume) / min(volume) within GTIN group")
    ax.set_title(
        f"Within-GTIN volume disagreement (n={len(flagged)}): "
        f"{n_near} near-miss, {n_real} likely variant/error")
    ax.legend()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_flavor(df: pd.DataFrame, out_path: Path) -> None:
    """Flavor detection-source overlap + agreement when both present.

    Answers "should I trust name-derived flavor over attribute-derived flavor"
    (the step-03 feature-weighting decision) via the both/name-only/attr-only/
    neither split and the agreement rate when both fire.
    """
    known = df[df["gtin"].fillna("").str.len() > 0].copy()
    known["has_name"] = known["flavor_from_name"].notna()
    known["has_attr"] = known["flavor_from_attribute"].notna()
    both = known[known["has_name"] & known["has_attr"]]
    agree_rate = (
        (both["flavor_from_name"] == both["flavor_from_attribute"]).mean()
        if len(both) else float("nan")
    )
    categories = {
        "both agree": int((both["flavor_from_name"] == both["flavor_from_attribute"]).sum()),
        "both, disagree": int((both["flavor_from_name"] != both["flavor_from_attribute"]).sum()),
        "name only": int((known["has_name"] & ~known["has_attr"]).sum()),
        "attribute only": int((~known["has_name"] & known["has_attr"]).sum()),
        "neither": int((~known["has_name"] & ~known["has_attr"]).sum()),
    }
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    sns.barplot(
        x=list(categories.values()), y=list(categories.keys()), ax=ax, palette="muted")
    ax.set_xlabel("GTIN-known SKUs")
    ax.set_title(f"Flavor detection source (agreement when both present: {agree_rate:.1%})")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_euromonitor()

    # Canonical volume comes from sku_name_eng ONLY (the most normalized text
    # source). Description is NOT used as a fallback: its nutrition/serving/
    # dilution prose ("per 12 fl oz", "0.2 l glass", "dilute in 9 volumes")
    # injects systematic false volumes (proven: fallback agreement 96.5% and
    # 198 flagged vs 99.0% and 54 flagged name-only). Description volume is
    # kept as an explicit low-confidence feature, never silently folded in.
    # extract_volume_ml returns (ml, ambiguous) — split into the canonical
    # column + an ambiguity flag (bare oz/ounce could be weight).
    _vol = df["sku_name_eng"].map(extract_volume_ml)
    df["canonical_volume_ml"] = _vol.map(lambda x: x[0])
    df["canonical_volume_ambiguous"] = _vol.map(lambda x: x[1])
    df["desc_volume_ml"] = df["description_short_eng"].map(
        lambda s: extract_volume_ml(s)[0])
    df["attribute_volume_ml"] = df["attribute"].map(parse_attribute_volume)
    df["volume_agreement"] = (
        df["canonical_volume_ml"].notna()
        & df["attribute_volume_ml"].notna()
        & (df["canonical_volume_ml"] == df["attribute_volume_ml"])
    )
    df["flavor_from_name"] = df["sku_name_eng"].map(flavor_from_name)
    df["flavor_from_attribute"] = df["attribute"].map(flavor_from_attribute)

    # DISPLAY TABLE ONLY (repo convention): compact describe-style summary.
    # Two DISTINCT agreement metrics, labeled explicitly so they can't be
    # conflated:
    #   A. row_level_volume_agreement — regex vs ML-extracted attribute, ALL
    #      SKUs dataset-wide. Expected lower (attribute field is noisy; a
    #      mismatch may mean the regex is right and the attribute is wrong).
    #   B. within_gtin_volume_agreement — canonical volume consistent within
    #      each multi-retailer GTIN group (ground-truth validation; the
    #      number that validates the extractor).
    summary = pd.DataFrame({
        "metric": [
            "skus_total", "skus_with_gtin",
            "canonical_volume_detected", "attribute_volume_detected",
            "row_level_volume_agreement",  # A (dataset-wide, regex vs attribute)
            "within_gtin_volume_agreement",  # B (multi-retailer GTIN groups, ground truth)
            "flavor_from_name_detected", "flavor_from_attribute_detected",
        ],
        "value": [
            len(df),
            int(df["gtin"].fillna("").str.len().gt(0).sum()),
            int(df["canonical_volume_ml"].notna().sum()),
            int(df["attribute_volume_ml"].notna().sum()),
            round(df["volume_agreement"].mean(), 4),
            None,  # filled from consolidated agree table below
            round(df["flavor_from_name"].notna().mean(), 4),
            round(df["flavor_from_attribute"].notna().mean(), 4),
        ],
    })
    summary.to_csv(CSV_PER_SKU, index=False)
    print(f"wrote {CSV_PER_SKU} (display table, {len(summary)} rows)")

    out = df  # working frame; per-SKU columns live here, not persisted to CSV

    # ---- GTIN-group validation on multi-retailer subset --------------------
    # Computed ONCE; summary stat + printed sample both derive from `agree`.
    # Honest denominators: a group counts toward canonical_agree only if it has
    # >=1 detected canonical volume (empty groups are excluded, not counted as
    # trivially-agreeing). attr_agree mirrors the same rule for the attribute
    # field, giving the ML-vs-regex contrast point.
    known = out[out["gtin"].fillna("").str.len() > 0]
    multi = known[known.groupby("gtin")["retailer"].transform("nunique") > 1]
    agree = multi.groupby("gtin").apply(
        lambda x: pd.Series({
            "retailers": x["retailer"].nunique(),
            "skus": len(x),
            "canonical_volumes": sorted(x["canonical_volume_ml"].dropna().unique().tolist()),
            "attr_volumes": sorted(x["attribute_volume_ml"].dropna().unique().tolist()),
            "canonical_agree": (
                x["canonical_volume_ml"].dropna().nunique() <= 1
                if x["canonical_volume_ml"].notna().any() else None
            ),
            "attr_agree": (
                x["attribute_volume_ml"].dropna().nunique() <= 1
                if x["attribute_volume_ml"].notna().any() else None
            ),
        }),
    ).reset_index()
    with_canon = agree.dropna(subset=["canonical_agree"])
    canonical_rate = with_canon["canonical_agree"].mean()
    with_attr = agree.dropna(subset=["attr_agree"])
    attr_rate = with_attr["attr_agree"].mean()
    flagged = agree[agree["canonical_agree"] == False]  # noqa: E712

    # ---- backfill the summary table with the consolidated numbers ----------
    summary.loc[summary["metric"] == "within_gtin_volume_agreement", "value"] = (
        round(float(canonical_rate), 4))
    summary = pd.concat([summary, pd.DataFrame({
        "metric": ["within_gtin_attribute_agreement"],
        "value": [round(float(attr_rate), 4)],
    })], ignore_index=True)
    summary.to_csv(CSV_PER_SKU, index=False)
    print(f"wrote {CSV_PER_SKU} (display table, {len(summary)} rows)")

    print(f"\nmulti-retailer GTIN groups with >=1 canonical volume: {len(with_canon):,}")
    print(f"  canonical volume AGREES within GTIN: {canonical_rate:.1%}")
    print(f"  attribute volume AGREES within GTIN: {attr_rate:.1%}")
    print(f"  DISAGREEING groups (canonical): {len(flagged):,}")

    # ---- figures -----------------------------------------------------------
    plot_disagreement(agree, PNG_DISAGREE)
    print(f"wrote {PNG_DISAGREE}")
    plot_flavor(out, PNG_FLAVOR)
    print(f"wrote {PNG_FLAVOR}")

    # sample of flagged groups
    if len(flagged):
        print("\n=== sample of disagreeing GTIN groups ===")
        for _, r in flagged.head(8).iterrows():
            print(f"GTIN {r['gtin']}: canonical={r['canonical_volumes']} attr={r['attr_volumes']}")


if __name__ == "__main__":
    main()
