"""02c: Case-sensitivity probe for the matching step (blocking-key prep).

Question: does upper/lower case in title cost us matches? Ground truth
here is the multi-retailer BARCODE groups (same BARCODE => same product): a group
"matches" when its member names collapse to ONE unique value after whitespace
strip. We measure that (a) as-is, (b) lowercased, and count the groups that
differ ONLY by case — the ones lowercasing fixes. Volume agreement is included
as a reference: every regex in _text.py is compiled re.IGNORECASE, so
lowercasing MUST NOT move volume agreement; we measure both variants to prove
it rather than assert it.

Writes (RESULTS = project/experiments/results/euromonitor/):
  02c_uppercase_letters.csv    per-letter A-Z counts (display table)
  02c_case_match_summary.csv   name + volume match, before/after lowercase
  02c_case_normalization.png   name-match bars + letter-frequency bar
"""


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from _common import RESULTS, load_dataset
from _text import extract_volume_ml

CSV_LETTERS = RESULTS / "02c_uppercase_letters.csv"
CSV_SUMMARY = RESULTS / "02c_case_match_summary.csv"
PNG_CASE = RESULTS / "02c_case_normalization.png"


def uppercase_letter_counts(names: pd.Series) -> pd.DataFrame:
    """Per-letter A-Z counts across the name column (display table only)."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    counts = {c: int(names.str.count(c).fillna(0).sum()) for c in letters}
    total = sum(counts.values())
    df = pd.DataFrame({
        "letter": list(letters),
        "count": [counts[c] for c in letters],
        "share_of_uppercase": [
            round(counts[c] / total, 4) if total else 0.0 for c in letters],
    })
    return df


def volume_agree_rate(multi: pd.DataFrame, col: str) -> float:
    """02-style honest within-BARCODE agreement: a group counts only if it has
    >=1 non-null volume (empty groups are excluded, not trivially agreeing)."""
    agree = multi.groupby("barcode")[col].apply(
        lambda s: (s.dropna().nunique() <= 1) if s.notna().any() else None)
    return float(agree.dropna().mean())


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_dataset()

    # ---- ground-truth subset, identical to 02 ----------------------------------
    known = df[df["barcode"].fillna("").str.len() > 0]
    multi = known[known.groupby("barcode")["retailer"].transform("nunique") > 1].copy()
    multi["name_raw"] = multi["title"].str.strip()
    multi["name_lower"] = multi["name_raw"].str.lower()

    # ---- 1) uppercase inventory: display all letters + counts ------------------
    letters = uppercase_letter_counts(df["title"])
    letters.to_csv(CSV_LETTERS, index=False)
    print(f"wrote {CSV_LETTERS} (display table, {len(letters)} rows)")
    txt = df["title"].fillna("")
    has_upper = txt.str.contains(r"[A-Z]", regex=True)
    has_lower = txt.str.contains(r"[a-z]", regex=True)
    n_upper_rows = int(has_upper.sum())
    n_all_caps = int((has_upper & ~has_lower).sum())

    # ---- 2) name identity match within BARCODE groups: as-is vs lowercased --------
    # nunique() excludes NaN; a group with >=1 non-null name is the honest
    # denominator (mirrors 02's volume-agreement rule).
    uniq_raw = multi.groupby("barcode")["name_raw"].nunique()
    uniq_lower = multi.groupby("barcode")["name_lower"].nunique()
    has_name = multi.groupby("barcode")["name_raw"].apply(lambda s: s.notna().any())
    valid = has_name[has_name].index
    n_groups = len(uniq_raw)
    match_raw = float((uniq_raw[valid] == 1).mean())
    match_lower = float((uniq_lower[valid] == 1).mean())
    case_only = int(((uniq_raw[valid] > 1) & (uniq_lower[valid] == 1)).sum())
    real_diff = int(((uniq_raw[valid] > 1) & (uniq_lower[valid] > 1)).sum())

    # ---- 3) volume agreement reference: as-is vs lowercased (must be equal) ----
    multi["vol_raw"] = multi["title"].map(
        lambda s: extract_volume_ml(s)[0] if pd.notna(s) else None)
    multi["vol_lower"] = multi["name_lower"].map(
        lambda s: extract_volume_ml(s)[0] if pd.notna(s) else None)
    vol_raw = volume_agree_rate(multi, "vol_raw")
    vol_lower = volume_agree_rate(multi, "vol_lower")

    # ---- 4) summary display table ----------------------------------------------
    summary = pd.DataFrame({
        "metric": [
            "multi_retailer_barcode_groups", "groups_with_name",
            "name_match_as_is", "name_match_lowercased",
            "case_only_diff_groups", "real_name_diff_groups",
            "rows_with_uppercase", "all_caps_rows", "total_uppercase_letters",
            "within_barcode_volume_agree_as_is", "within_barcode_volume_agree_lowercased",
        ],
        "value": [
            n_groups, len(valid),
            round(match_raw, 4), round(match_lower, 4),
            case_only, real_diff,
            n_upper_rows, n_all_caps, int(letters["count"].sum()),
            round(vol_raw, 4), round(vol_lower, 4),
        ],
    })
    summary.to_csv(CSV_SUMMARY, index=False)
    print(f"wrote {CSV_SUMMARY} (display table, {len(summary)} rows)")

    # ---- 5) figures: name-match before/after + letter frequency ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    bars = axes[0].bar(
        ["as-is", "lowercased"], [match_raw * 100, match_lower * 100],
        color=["#C44E52", "#4C72B0"], width=0.5)
    axes[0].bar_label(bars, fmt="%.1f%%", fontsize=9)
    axes[0].set_ylim(0, min(max(match_raw, match_lower) * 110, 100.0))  # data-derived, capped
    axes[0].set_ylabel("groups with one unique name (%)")
    axes[0].set_title(
        f"Name identity match within BARCODE groups\n"
        f"(case-only diffs: {case_only:,} groups — what lowercasing fixes)")
    axes[1] = sns.barplot(
        x="letter", y="count", data=letters.sort_values("count", ascending=False),
        ax=axes[1], color="#4C72B0")
    axes[1].set_ylim(0, int(letters["count"].max()) * 1.1)  # data-derived headroom
    axes[1].set_title("Uppercase letters in title (A-Z counts)")
    axes[1].set_xlabel("letter")
    fig.suptitle("Case normalization probe (02c)", fontsize=11)
    fig.savefig(PNG_CASE, dpi=150)
    plt.close(fig)
    print(f"wrote {PNG_CASE}")

    # ---- 6) printed report ------------------------------------------------------
    print(f"\nmulti-retailer BARCODE groups: {n_groups:,} ({len(valid):,} with >=1 name)")
    print(f"  name identity match as-is:      {match_raw:.1%}")
    print(f"  name identity match lowercased: {match_lower:.1%}")
    print(f"  groups differing ONLY by case:  {case_only:,}   (real name diffs: {real_diff:,})")
    print(f"  rows with uppercase letters:    {n_upper_rows:,}   all-caps rows: {n_all_caps:,}"
          f"   total uppercase letters: {int(letters['count'].sum()):,}")
    verdict = ("identical — regexes are re.IGNORECASE" if vol_raw == vol_lower
               else "DIFFER — investigate!")
    print(f"  within-BARCODE volume agreement:   as-is {vol_raw:.1%} vs lowercased {vol_lower:.1%} "
          f"({verdict})")


if __name__ == "__main__":
    main()
