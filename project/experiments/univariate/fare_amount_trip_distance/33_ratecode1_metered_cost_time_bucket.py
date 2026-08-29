"""33: metered_cost as target — fare + extra + mta_tax + tolls, time buckets, all rows.

The hypothesis test from the data dictionary: if `extra` really is the
peak/overnight surcharge, then on ALL 42,806 metered rows with
metered_cost = fare_amount + extra + mta_tax + tolls_amount as the target,
the time buckets should snap to time_peak ~ +$1.00 and time_overnight ~
+$0.50. This step runs that test, renders the time-of-day visuals adapted
to metered_cost, and prints the forensic evidence about what the `extra`
column actually contains (spoiler: it does not match the TLC dictionary —
negatives, toll/congestion contamination, only ~68% total_amount
reconciliation).
"""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import RESULTS, WORKING_DATASET, load_metered
from _ols_bp import estimation_table, fit_time_bucket, time_bucket
from _tests import write_tests_json

from broadway.stats.regression import bp_jb

OUT = RESULTS / f"{Path(__file__).stem}.png"
ESTIMATES_CSV = RESULTS / f"{Path(__file__).stem}.csv"

TARGET = "metered_cost"
METERED_COST_DEF = "fare_amount + extra + mta_tax + tolls_amount"

# (start_hour, end_hour_exclusive, color, band label)
BUCKET_BANDS = [
    (0, 6, "#C8DCFF", "overnight"),
    (16, 20, "#FFD8B0", "peak"),
    (20, 24, "#C8DCFF", "overnight"),
]

TERM_LABELS = {
    "const": "intercept ($)",
    "trip_distance": "trip_distance ($/mi)",
    "duration_minutes": "duration_minutes ($/min)",
    "time_peak": "time_peak ($/trip)",
    "time_overnight": "time_overnight ($/trip)",
}


def draw_hourly_profile(ax, df: pd.DataFrame) -> None:
    """Hourly mean target (line) + trip counts (bars) with bucket bands."""
    hourly = df.groupby("pickup_hour")[TARGET].agg(["mean", "count"])
    for start, end, color, label in BUCKET_BANDS:
        ax.axvspan(start, end, color=color, alpha=0.45,
                   label=label if start == 0 else None)
    ax2 = ax.twinx()
    ax2.bar(hourly.index, hourly["count"], color="gray", alpha=0.25, label="trips")
    ax2.set_ylabel("trips")
    ax.plot(hourly.index, hourly["mean"], marker="o", color="black", ms=4,
            label=f"mean {TARGET} ($)")
    ax.set_xlabel("hour of day")
    ax.set_ylabel(f"mean {TARGET} ($)")
    ax.set_xticks(range(0, 24, 2))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")


def draw_bucket_effects(ax, est: pd.DataFrame) -> None:
    """Estimated bucket effects (95% HC3 CIs) vs the NYC surcharge rules."""
    terms = ["time_peak", "time_overnight"]
    labels = ["peak (16-19h)", "overnight (0-5, 20-23h)"]
    coef = [est.loc[t, "coef"] for t in terms]
    lo = [est.loc[t, "CI_low"] for t in terms]
    hi = [est.loc[t, "CI_high"] for t in terms]
    y = [1, 0]
    ax.errorbar(coef, y,
                xerr=[[c - l for c, l in zip(coef, lo)],
                      [h - c for h, c in zip(hi, coef)]],
                fmt="o", color="black", capsize=4, ms=6)
    ax.axvline(0, color="black", linestyle=":", linewidth=1)
    ax.axvline(1.00, color="green", linestyle="--", linewidth=1,
               label="NYC peak rule +$1.00")
    ax.axvline(0.50, color="green", linestyle=":", linewidth=1,
               label="NYC overnight rule +$0.50")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("$ per trip vs day (95% HC3 CI)")
    ax.set_title("Bucket effects vs NYC surcharge rules")
    for i, t in enumerate(terms):
        ax.annotate(f"${coef[i]:+.2f}", xy=(coef[i], y[i]), xytext=(6, 0),
                    textcoords="offset points", fontsize=9, va="center")
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend(fontsize=8, loc="lower right")


def draw_full_forest(ax, est: pd.DataFrame) -> None:
    """Coefficient forest: all terms with 95% HC3 CIs, zero line."""
    terms = list(est.index)
    ypos = list(range(len(terms)))[::-1]
    coef = est["coef"].to_numpy()
    lo = est["CI_low"].to_numpy()
    hi = est["CI_high"].to_numpy()
    ax.errorbar(coef, ypos, xerr=[coef - lo, hi - coef],
                fmt="o", color="black", ecolor="black", capsize=3, ms=5)
    ax.axvline(0, color="black", linestyle=":", linewidth=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([TERM_LABELS.get(t, t) for t in terms], fontsize=9)
    ax.set_xlabel("dollar effect per unit (95% HC3 CI)")
    ax.set_title("Full coefficient forest")
    for i, t in enumerate(terms):
        ax.annotate(f"{coef[i]:.3f} [{lo[i]:.2f}, {hi[i]:.2f}]",
                    xy=(hi[i], ypos[i]), xytext=(6, 0),
                    textcoords="offset points", fontsize=8, va="center")
    ax.grid(True, alpha=0.3, axis="x")


def draw_extra_structure(ax, df: pd.DataFrame) -> None:
    """Mean `extra` by hour (weekday vs weekend) vs the rule reference lines."""
    df = df.copy()
    df["hour"] = df["pickup_hour"]
    df["weekday"] = df["tpep_pickup_datetime"].dt.weekday < 5
    wk = df[df["weekday"]].groupby("hour")["extra"].mean()
    we = df[~df["weekday"]].groupby("hour")["extra"].mean()
    ax.plot(wk.index, wk, marker="o", color="black", ms=3, label="weekday")
    ax.plot(we.index, we, marker="s", color="darkred", ms=3, label="weekend")
    ax.axhline(1.00, color="green", linestyle="--", linewidth=1,
               label="peak rule +$1.00")
    ax.axhline(0.50, color="green", linestyle=":", linewidth=1,
               label="overnight rule +$0.50")
    ax.set_xlabel("hour of day")
    ax.set_ylabel("mean extra ($)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title("What 'extra' actually contains vs the rules")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")


def main() -> None:
    df = load_metered()
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df[TARGET] = df["fare_amount"] + df["extra"] + df["mta_tax"] + df["tolls_amount"]
    model = fit_time_bucket(df, target=TARGET)

    counts = df["pickup_hour"].map(time_bucket).value_counts()
    print(f"rows: {len(df)} (ALL metered rows, no exclusion) | buckets: "
          + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print(model.summary())

    est = estimation_table(model)
    print("\n=== time-bucket estimates (HC3 95% CIs) ===")
    print(est.round(4).to_string())
    for term in ("time_peak", "time_overnight"):
        row = est.loc[term]
        print(f"{term}: ${row['coef']:+.2f} per trip "
              f"(95% CI [{row['CI_low']:+.2f}, {row['CI_high']:+.2f}])")

    diag = bp_jb(model)
    print(f"R^2={model.rsquared:.4f} | BP={diag['bp_stat']:.1f} | "
          f"JB={diag['jb_stat']:.2e} | resid kurtosis={diag['kurtosis']:.1f}")

    extra_by = df.assign(weekday=df["tpep_pickup_datetime"].dt.weekday < 5,
                         bucket=df["pickup_hour"].map(time_bucket))
    print("\n=== forensic: mean `extra` by bucket (weekday | weekend) ===")
    for b in ("day", "peak", "overnight"):
        wk = extra_by[(extra_by["bucket"] == b) & extra_by["weekday"]]["extra"].mean()
        we = extra_by[(extra_by["bucket"] == b) & ~extra_by["weekday"]]["extra"].mean()
        print(f"  {b:>9}: {wk:+.2f} | {we:+.2f}")
    print("NOTE: dictionary surcharge would be day=0, peak=1.00 (weekday), "
          "overnight=0.50; the data does not show this.")

    # Clean-rows structure: rows where `extra` is a dictionary-plausible surcharge
    clean_mask = df["extra"].round(2).isin([0, 0.5, 1.0, 1.5])
    clean = df[clean_mask]
    clean = clean.assign(weekday=clean["tpep_pickup_datetime"].dt.weekday < 5)
    hour_wk = {h: round(float(clean[(clean["pickup_hour"] == h) & clean["weekday"]]
                               ["extra"].mean()), 3) for h in range(24)}
    hour_we = {h: round(float(clean[(clean["pickup_hour"] == h) & ~clean["weekday"]]
                               ["extra"].mean()), 3) for h in range(24)}
    print("\n=== clean-`extra` rows: mean extra by hour (weekday | weekend) ===")
    print("structure: +$1.00 overnight (20-23h, 0-5h), $0.00 6-19h — no peak surcharge")
    print("  " + " ".join(f"{h:2d}h:{hour_wk[h]:+.2f}/{hour_we[h]:+.2f}" for h in range(24)))

    fig, ((ax_profile, ax_effects), (ax_forest, ax_extra)) = plt.subplots(
        2, 2, figsize=(16, 11))
    draw_hourly_profile(ax_profile, df)
    ax_profile.set_title("Metered cost by hour of day")
    draw_bucket_effects(ax_effects, est)
    draw_full_forest(ax_forest, est)
    draw_extra_structure(ax_extra, df)
    fig.suptitle(
        f"Time-of-day analysis — {TARGET} target, all rows (N={len(df)})",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}")

    est_out = est.rename(columns={
        "HC3_SE": "hc3_se", "CI_low": "ci_low", "CI_high": "ci_high",
    }).to_dict("index")
    results = {
        "hc3_metered_cost_trip_distance_duration_time_bucket": {
            "method": (f"{TARGET} = {METERED_COST_DEF}; raw ~ distance + duration "
                       "+ time_peak + time_overnight (day ref), HC3; ALL rows"),
            "n": int(model.nobs),
            "rsquared": float(model.rsquared),
            "params": {name: float(v) for name, v in model.params.items()},
            "bse": {name: float(v) for name, v in model.bse.items()},
            "pvalues": {name: float(v) for name, v in model.pvalues.items()},
            "bp_jb": diag,
            "bucket_counts": {k: int(v) for k, v in counts.items()},
            "estimation": {
                term: {
                    "coef": float(v["coef"]),
                    "hc3_se": float(v["hc3_se"]),
                    "ci_low": float(v["ci_low"]),
                    "ci_high": float(v["ci_high"]),
                }
                for term, v in est_out.items()
            },
            "extra_forensics": {
                "note": ("`extra` does not match the TLC dictionary (negatives, "
                         "toll/congestion contamination, ~68% total reconciliation); "
                         "the +$1.00/+$0.50 test is not valid on this column"),
                "mean_extra_by_bucket_weekday": {
                    b: round(float(extra_by[(extra_by["bucket"] == b)
                                            & extra_by["weekday"]]["extra"].mean()), 3)
                    for b in ("day", "peak", "overnight")
                },
                "mean_extra_by_bucket_weekend": {
                    b: round(float(extra_by[(extra_by["bucket"] == b)
                                            & ~extra_by["weekday"]]["extra"].mean()), 3)
                    for b in ("day", "peak", "overnight")
                },
                "clean_rows": {
                    "share": round(float(clean_mask.mean()), 3),
                    "rule": ("clean rows show +$1.00 overnight (8pm-6am = hours "
                             "20-23 + 0-5), $0.00 for 6-19h; no peak-weekday "
                             "surcharge in `extra`; the assumed +$1.00 peak / "
                             "+$0.50 overnight rules do NOT match the data"),
                    "mean_extra_by_hour_weekday": hour_wk,
                    "mean_extra_by_hour_weekend": hour_we,
                },
            },
        }
    }
    out = write_tests_json(
        WORKING_DATASET,
        results,
        "33_ratecode1_metered_cost_time_bucket.py",
        n_rows=len(pd.read_parquet(WORKING_DATASET)),
    )
    print(f"wrote {out}")

    csv_rows = []
    for term, v in results["hc3_metered_cost_trip_distance_duration_time_bucket"]["estimation"].items():
        csv_rows.append({
            "term": term, "coef": v["coef"], "hc3_se": v["hc3_se"],
            "ci_low": v["ci_low"], "ci_high": v["ci_high"],
        })
    pd.DataFrame(csv_rows).to_csv(ESTIMATES_CSV, index=False)
    print(f"wrote {ESTIMATES_CSV}")


if __name__ == "__main__":
    main()
