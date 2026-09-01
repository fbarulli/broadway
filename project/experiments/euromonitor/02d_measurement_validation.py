"""02d: Measurement validation — physical sanity of extracted (value, unit).

After 02 normalized volume to canonical ml, this layer re-inspects the RAW
(value, unit) pair (extract_volume_measurement) and flags physically
impossible measurements: a VOLUME-type category (category taxonomy in
_text.py) that mentions a weight unit (kg/lb — the extractor never yields
one, so the row text is checked via WEIGHT_UNIT_RE) or a liter-family unit
above 10 L (the '350 L coconut water' error class). The kg rule fires
through weight_hint, not the unit column, because VOLUME_RE can never
return a weight unit. Weight-type categories (powder/dry mixes) legitimately
sell by weight and are never flagged.

Documented limitations (known false positives, measured not guessed):
  L1 multipack totals: "10 x 0,20l" extracts 20.0 L and trips the >10 L
     rule while every per-unit volume (0.2 L) is valid. Flagged examples
     carry is_multipack_candidate so the share is auditable; a production
     system would parse the pack and validate per-unit instead.

Per-row columns (the validate_measurement contract):
  extracted_value, extracted_unit, measurement_status
  (Missing | Flagged_Anomaly | Valid), is_multipack_candidate.

Sanity checks (run before writing, fail loudly):
  S1 statuses are only {Missing, Flagged_Anomaly, Valid}
  S2 every Flagged_Anomaly row is a volume-type category (taxonomy)
  S3 every Flagged_Anomaly row has weight_hint OR (liter unit AND value > 10)
  S4 volume-type rows with a weight hint are never Valid
  S5 status counts sum to the full row count
  S6 weight-hint rows equal 01c's measured weight_unit_rows (re-pinned after
     the mg-dosage fix) — same regex, same column, dataset-refresh guard.

Writes (RESULTS = project/experiments/results/euromonitor/):
  02d_measurement_status.csv   status counts + shares (display table)
  02d_sanity_checks.csv        S1-S6 verdicts (display table)
  02d_flagged_examples.csv     top 15 flagged anomalies incl. multipack flag
  02d_measurement_status.png   status distribution bar
"""


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from _common import RESULTS, load_dataset
from _text import (
    _LITER_UNITS,
    MULTIPACK_RE,
    WEIGHT_UNIT_RE,
    extract_volume_measurement,
    get_measurement_type,
    validate_measurement,
)

CSV_STATUS = RESULTS / "02d_measurement_status.csv"
CSV_SANITY = RESULTS / "02d_sanity_checks.csv"
CSV_FLAGGED = RESULTS / "02d_flagged_examples.csv"
PNG_STATUS = RESULTS / "02d_measurement_status.png"

PINNED_WEIGHT_ROWS = 1644  # re-measured after mg + zero-claim fixes (was 1,932)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    title = df["title"].fillna("")

    meas = title.map(extract_volume_measurement)
    df["extracted_value"] = meas.map(lambda t: t[0])
    df["extracted_unit"] = meas.map(lambda t: t[1])
    df["ambiguous"] = meas.map(lambda t: t[2])
    df["weight_hint"] = title.map(
        lambda s: bool(WEIGHT_UNIT_RE.search(s)) if pd.notna(s) else False)
    df["is_multipack_candidate"] = title.str.contains(MULTIPACK_RE, na=False)

    df["measurement_status"] = df.apply(
        lambda r: validate_measurement(
            r["extracted_value"], r["extracted_unit"], r["category"],
            weight_hint=r["weight_hint"]),
        axis=1)

    statuses = df["measurement_status"].value_counts()
    n_missing = int(statuses.get("Missing", 0))
    n_flagged = int(statuses.get("Flagged_Anomaly", 0))
    n_valid = int(statuses.get("Valid", 0))
    n_weight_hint = int(df["weight_hint"].sum())

    # ---- sanity checks (fail loudly) ----------------------------------------
    is_volume = df["category"].map(get_measurement_type) == "volume"
    flagged = df[df["measurement_status"] == "Flagged_Anomaly"]
    s3 = flagged.apply(
        lambda r: bool(r["weight_hint"]) or (
            str(r["extracted_unit"]) in _LITER_UNITS
            and r["extracted_value"] > 10),
        axis=1) if len(flagged) else pd.Series(dtype=bool)
    wb = df[is_volume & df["weight_hint"]]

    checks = [
        ("S1 statuses only Missing/Flagged_Anomaly/Valid",
         set(statuses.index).issubset({"Missing", "Flagged_Anomaly", "Valid"})),
        ("S2 flagged => volume-type category (taxonomy)",
         flagged.empty or bool(is_volume[flagged.index].all())),
        ("S3 flagged => weight_hint or liter>10",
         bool(s3.all())),
        ("S4 volume-type + weight_hint never Valid",
         bool((wb["measurement_status"] != "Valid").all())),
        ("S5 counts sum to full row count",
         n_missing + n_flagged + n_valid == len(df)),
        ("S6 weight_hint rows == 01c weight_unit_rows (re-pinned)",
         n_weight_hint == PINNED_WEIGHT_ROWS),
    ]
    for name, ok in checks:
        if not ok:
            raise AssertionError(f"sanity check FAILED: {name}")

    # ---- multipack share among flagged (documented limitation L1) ------------
    flagged_multipack = int(flagged["is_multipack_candidate"].sum())
    multipack_share = flagged_multipack / len(flagged) if len(flagged) else 0.0

    # ---- display tables ------------------------------------------------------
    status_frame = pd.DataFrame({
        "status": ["Missing", "Flagged_Anomaly", "Valid"],
        "count": [n_missing, n_flagged, n_valid],
        "share": [round(v / len(df), 4) for v in (n_missing, n_flagged, n_valid)],
    })
    status_frame.to_csv(CSV_STATUS, index=False)
    print(f"wrote {CSV_STATUS} (display table, {len(status_frame)} rows)")
    pd.DataFrame(checks, columns=["check", "pass"]).to_csv(CSV_SANITY, index=False)
    print(f"wrote {CSV_SANITY} (display table, {len(checks)} rows)")

    flagged_top = flagged.head(15)[
        ["title", "category", "extracted_value", "extracted_unit",
         "weight_hint", "is_multipack_candidate"]]
    flagged_top.to_csv(CSV_FLAGGED, index=False)
    print(f"wrote {CSV_FLAGGED} (display table, {len(flagged_top)} rows)")

    # ---- figure ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4), constrained_layout=True)
    bars = ax.bar(status_frame["status"], status_frame["count"],
                  color=["#BBBBBB", "#C44E52", "#4C72B0"], width=0.55)
    ax.bar_label(bars, fmt="{:,.0f}", fontsize=9)
    ax.set_ylim(0, int(status_frame["count"].max()) * 1.15)  # data-derived
    ax.set_ylabel("SKUs")
    ax.set_title("Measurement validation: extracted (value, unit) sanity")
    fig.savefig(PNG_STATUS, dpi=150)
    plt.close(fig)
    print(f"wrote {PNG_STATUS}")

    # ---- printed report --------------------------------------------------------
    print(f"\nmeasurement_status distribution ({len(df):,} rows):")
    for _, r in status_frame.iterrows():
        print(f"  {r['status']:<16} {r['count']:>7,}  ({r['share']:.2%})")
    print(f"  weight-hint rows: {n_weight_hint:,}")
    print(f"  flagged multipack candidates: {flagged_multipack:,} "
          f"({multipack_share:.1%} of flagged) — documented limitation L1")
    print("\nsanity checks:")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if len(flagged):
        print(f"\ntop {len(flagged_top)} flagged anomalies:")
        for _, r in flagged_top.iterrows():
            mp = " [multipack]" if r["is_multipack_candidate"] else ""
            print(f"  {r['extracted_value']} {r['extracted_unit']:<5} | "
                  f"{r['category'][:34]:<34} | {r['title'][:56]}{mp}")


if __name__ == "__main__":
    main()
