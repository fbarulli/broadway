"""04c: Blocking A/B — measure the three candidate keys on real ground truth.

The audit (04b) inferred which category granularity to use; this step MEASURES
it. Three blocking keys are built over every row, and evaluated on ALL true
pairs (same barcode, multi-retailer groups — uncapped combinations):

  A brand | strict category | volume bucket
  B brand | macro category  | volume bucket
  C brand | volume bucket            (is category even needed?)

Blocking recall  = fraction of true pairs whose two rows land in the same block
Candidates       = sum over blocks of n*(n-1)/2 — the pair count the pairwise
                   classifier must score (the cost of the rule)

Sentinels: NO_BRAND / NO_VOL keep rows with missing values in explicit blocks
(no silent None==None 'retained'). Volume is the 02 canonical (name-only,
bucket_ml, nearest 5). Sanity: recall and candidates must be monotone
A <= B <= C (C's blocks are supersets of B's, B's of A's) — fail loudly if not.

Writes (RESULTS = project/experiments/results/euromonitor/):
  04c_blocking_ab.csv   rule, recall, candidates, blocks (display table)
  04c_blocking_ab.png   recall + candidate bars (data-derived limits)
"""


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from _blocking import build_true_pairs, eval_blocking
from _common import RESULTS, load_dataset
from _text import MACRO_MAP, extract_volume_ml

CSV_BLOCKING = RESULTS / "04c_blocking_ab.csv"
PNG_BLOCKING = RESULTS / "04c_blocking_ab.png"

NO_BRAND = "NO_BRAND"
NO_VOL = "NO_VOL"
NO_CATEGORY = "NO_CATEGORY"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    # canonical volume per row (02's name-only extractor, already bucketed).
    # vol_ml stays numeric (NaN = missing) for the loss decomposition; the
    # blocking key uses the NO_VOL sentinel so missing values are explicit.
    vol_ml = df["title"].fillna("").map(extract_volume_ml).map(lambda t: t[0])
    vol = vol_ml.fillna(NO_VOL).astype(str)
    brand = df["brand"].fillna("").str.strip().replace("", NO_BRAND)
    strict = df["category"].fillna("").str.strip().replace("", NO_CATEGORY)
    macro = strict.map(MACRO_MAP).fillna("UNKNOWN")

    # ---- ground truth: ALL true pairs (same barcode, multi-retailer) ---------
    pairs = build_true_pairs(df)
    print(f"true pairs (same barcode): {len(pairs):,}")

    rules = {
        "A brand x strict x vol": brand + "|" + strict + "|" + vol,
        "B brand x macro x vol": brand + "|" + macro + "|" + vol,
        "C brand x vol": brand + "|" + vol,
    }
    results = {}
    for name, key in rules.items():
        rec, cand, n_blocks = eval_blocking(key, pairs)
        results[name] = (rec, cand, n_blocks)
        print(f"  {name:<22} recall={rec:.4f}  candidates={cand:,}  "
              f"blocks={n_blocks:,}")

    # ---- loss decomposition: WHERE do the missing pairs go? -------------------
    va = vol_ml.loc[[a for a, _ in pairs]].to_numpy()
    vb = vol_ml.loc[[b for _, b in pairs]].to_numpy()
    both_vol = pd.notna(va) & pd.notna(vb)
    same_brand = (brand.loc[[a for a, _ in pairs]].values
                  == brand.loc[[b for _, b in pairs]].values)
    same_macro = (macro.loc[[a for a, _ in pairs]].values
                  == macro.loc[[b for _, b in pairs]].values)
    no_vol_recall = float((same_brand & same_macro).mean())  # volume out of the key

    # ---- sanity: coarser keys can only keep or add candidates ----------------
    rec_a, cand_a = results["A brand x strict x vol"][0], results["A brand x strict x vol"][1]
    rec_b, cand_b = results["B brand x macro x vol"][0], results["B brand x macro x vol"][1]
    rec_c, cand_c = results["C brand x vol"][0], results["C brand x vol"][1]
    checks = [
        ("monotone recall A<=B<=C", rec_a <= rec_b + 1e-9 <= rec_c + 1e-9),
        ("monotone candidates A<=B<=C", cand_a <= cand_b + 1e-9 <= cand_c + 1e-9),
        ("recall in (0,1]", 0 < rec_c <= 1),
        ("candidates < total pairs", cand_c < len(df) * (len(df) - 1) // 2),
    ]
    for name, ok in checks:
        if not ok:
            raise AssertionError(f"sanity check FAILED: {name}")
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    # ---- display table ----------------------------------------------------------
    frame = pd.DataFrame({
        "rule": list(rules),
        "blocking_recall": [round(v[0], 4) for v in results.values()],
        "candidate_pairs": [v[1] for v in results.values()],
        "n_blocks": [v[2] for v in results.values()],
    })
    frame = pd.concat([frame, pd.DataFrame({
        "rule": [
            "decomp: both members have volume",
            "decomp: same brand",
            "decomp: same macro",
            "ref: brand x macro (NO volume in key)",
        ],
        "blocking_recall": [
            round(float(both_vol.mean()), 4), round(float(same_brand.mean()), 4),
            round(float(same_macro.mean()), 4), round(no_vol_recall, 4),
        ],
        "candidate_pairs": [None] * 4, "n_blocks": [None] * 4,
    })], ignore_index=True)
    frame.to_csv(CSV_BLOCKING, index=False)
    print(f"wrote {CSV_BLOCKING} (display table, {len(frame)} rows)")

    # ---- figure -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    bars = axes[0].bar(list(rules), [v[0] * 100 for v in results.values()],
                       color=["#C44E52", "#4C72B0", "#2E7D32"], width=0.55)
    axes[0].bar_label(bars, fmt="%.2f%%", fontsize=9)
    axes[0].set_ylim(0, min(max(v[0] for v in results.values()) * 112, 100))
    axes[0].set_title("Blocking recall (true pairs retained)")
    axes[0].set_ylabel("% of true pairs in same block")
    bars = axes[1].bar(list(rules), [v[1] for v in results.values()],
                       color=["#C44E52", "#4C72B0", "#2E7D32"], width=0.55)
    axes[1].bar_label(bars, fmt="{:,.0f}", fontsize=9)
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.9, max(v[1] for v in results.values()) * 5)
    axes[1].set_title("Candidate pairs for the classifier")
    axes[1].set_ylabel("sum n*(n-1)/2 over blocks (log)")
    fig.suptitle("Blocking A/B: recall vs candidate cost", fontsize=12)
    fig.savefig(PNG_BLOCKING, dpi=150)
    plt.close(fig)
    print(f"wrote {PNG_BLOCKING}")

    # ---- verdict -------------------------------------------------------------------
    print(f"\ntotal possible pairs without blocking: "
          f"{len(df) * (len(df) - 1) // 2:,}")
    print("\nverdict: prefer the rule that keeps recall >= ~0.97 at the lowest "
          "candidate count (blocking is recall-first; scoring handles precision).")


if __name__ == "__main__":
    main()
