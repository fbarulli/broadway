"""32: time-of-day visuals for the bucket model — profile, rules, forest, lineage.

One 2x2 figure for the step-31 model (raw fare ~ distance + duration +
peak/overnight dummies, HC3, day reference, |M|>10 excluded):
(1) hourly mean-fare profile with trip counts and day/peak/overnight bands,
(2) estimated bucket effects vs the NYC surcharge rules (+$1.00 / +$0.50),
(3) full coefficient forest with 95% HC3 CIs, (4) diagnostics lineage
across models (baseline 15 / capped 27 / excluded+hour 28 / buckets 31).
All numbers come from the fitted model or the persisted JSON — nothing
hardcoded.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from _common import RESULTS, WORKING_DATASET, load_metered
from _ols_bp import (
    OUTLIER_Z_THRESHOLD,
    estimation_table,
    fit_time_bucket,
    outlier_mask,
)
from _tests import tests_path_for

OUT = RESULTS / f"{Path(__file__).stem}.png"

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
    """Hourly mean fare (line) + trip counts (bars) with bucket bands."""
    hourly = df.groupby("pickup_hour")["fare_amount"].agg(["mean", "count"])
    for start, end, color, label in BUCKET_BANDS:
        ax.axvspan(start, end, color=color, alpha=0.45,
                   label=label if start == 0 else None)
    ax2 = ax.twinx()
    ax2.bar(hourly.index, hourly["count"], color="gray", alpha=0.25, label="trips")
    ax2.set_ylabel("trips")
    ax.plot(hourly.index, hourly["mean"], marker="o", color="black", ms=4,
            label="mean fare ($)")
    ax.set_xlabel("hour of day")
    ax.set_ylabel("mean fare_amount ($)")
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
    ax.set_xlim(-0.7, 1.5)
    ax.set_title("Bucket effects vs NYC surcharge rules")
    for i, t in enumerate(terms):
        ax.annotate(f"${coef[i]:+.2f}", xy=(coef[i], y[i]), xytext=(6, 0),
                    textcoords="offset points", fontsize=9, va="center")
    ax.annotate("surcharges are NOT part of fare_amount",
                xy=(0.02, 0.04), xycoords="axes fraction", fontsize=8,
                color="darkred")
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


def load_lineage() -> list[dict]:
    """R^2 and residual kurtosis for models 15 / 27 / 28, from persisted JSON."""
    results = json.loads(tests_path_for(WORKING_DATASET).read_text())["results"]
    entries = [
        ("baseline 15", "hc3_fare_trip_distance_duration",
         "bp_jb_fare_trip_distance_duration"),
        ("capped 27", "hc3_fare_capped_trip_distance_duration_pickup_hour", None),
        ("excluded 28", "hc3_fare_trip_distance_duration_pickup_hour_excluded", None),
    ]
    rows = []
    for name, model_key, bp_key in entries:
        entry = results.get(model_key)
        if not entry:
            continue
        bp = results.get(bp_key) if bp_key else entry.get("bp_jb", {})
        rows.append({
            "name": name,
            "rsquared": entry.get("rsquared"),
            "kurtosis": bp.get("kurtosis"),
        })
    return rows


def draw_lineage(ax, rows: list[dict]) -> None:
    """R^2 (bars) and residual kurtosis (line) across the model lineage."""
    names = [r["name"] for r in rows]
    x = range(len(names))
    ax.bar(x, [r["rsquared"] for r in rows], color="#4C72B0", alpha=0.85,
           label="R^2")
    ax.set_ylabel("R^2")
    ax.set_ylim(0.98, 1.0)
    ax2 = ax.twinx()
    ax2.plot(x, [r["kurtosis"] for r in rows], marker="o", color="darkred",
             label="resid kurtosis")
    ax2.set_ylabel("resid kurtosis")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_title("Diagnostics lineage")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")


def main() -> None:
    df = load_metered()
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    excluded = outlier_mask().reindex(df.index).fillna(False)
    kept = df[~excluded]
    model = fit_time_bucket(kept)
    est = estimation_table(model)
    lineage = load_lineage() + [{
        "name": "buckets 31",
        "rsquared": float(model.rsquared),
        "kurtosis": float(model.resid.kurt()),
    }]

    fig, ((ax_profile, ax_effects), (ax_forest, ax_lineage)) = plt.subplots(
        2, 2, figsize=(16, 11))
    draw_hourly_profile(ax_profile, kept)
    ax_profile.set_title("Fare by hour of day")
    draw_bucket_effects(ax_effects, est)
    draw_full_forest(ax_forest, est)
    draw_lineage(ax_lineage, lineage)
    n = int(model.nobs)
    fig.suptitle(
        f"Time-of-day analysis — raw fare HC3, |M|>{OUTLIER_Z_THRESHOLD} excluded (N={n})",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
