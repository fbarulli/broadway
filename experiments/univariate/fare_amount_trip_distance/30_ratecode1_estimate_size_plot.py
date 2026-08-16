"""30: plot estimate-size results — coefficient forest plot + realistic-trip bars.

Visualizes step 29's estimation on the step-28 model (raw fare ~ distance +
duration + pickup_hour, HC3, modified-z outliers excluded). Left panel: the
canonical coefficient plot — point estimate + 95% HC3 CI per term, zero
reference line ("how big and how sure"). Right panel: the distance/duration
coefficients translated into dollar effects of realistic trips (5-mile
trip, 10-min wait, typical 1.6-mi trip) with CI whiskers. All numbers come
from the fitted model — nothing hardcoded.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from _common import RESULTS, load_metered
from _ols_bp import estimation_table, fit_raw_hc3, outlier_mask, scenario_dollars

OUT = RESULTS / f"{Path(__file__).stem}.png"

TERM_LABELS = {
    "const": "intercept ($)",
    "trip_distance": "trip_distance ($/mi)",
    "duration_minutes": "duration_minutes ($/min)",
    "pickup_hour": "pickup_hour ($/hr-of-day)",
}


def draw_forest(ax, est: pd.DataFrame) -> None:
    """Coefficient plot: point estimate + 95% CI per term, zero line."""
    terms = list(est.index)
    ypos = list(range(len(terms)))[::-1]  # first term (intercept) on top
    coef = est["coef"].to_numpy()
    lo = est["CI_low"].to_numpy()
    hi = est["CI_high"].to_numpy()
    ax.errorbar(coef, ypos, xerr=[coef - lo, hi - coef],
                fmt="o", color="black", ecolor="black", capsize=3, ms=5)
    ax.axvline(0, color="black", linestyle=":", linewidth=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([TERM_LABELS.get(t, t) for t in terms], fontsize=9)
    ax.set_xlabel("dollar effect per unit (95% HC3 CI)")
    ax.set_title("Coefficients")
    for i, t in enumerate(terms):
        ax.annotate(f"{coef[i]:.3f} [{lo[i]:.2f}, {hi[i]:.2f}]",
                    xy=(hi[i], ypos[i]), xytext=(6, 0),
                    textcoords="offset points", fontsize=8, va="center")
    ax.grid(True, alpha=0.3, axis="x")


def draw_scenarios(ax, rows: list[dict], est: pd.DataFrame) -> None:
    """Horizontal bars: dollar effect of each realistic trip, with CI whiskers."""
    ypos = list(range(len(rows)))[::-1]
    dollars = [r["dollars"] for r in rows]
    err_lo = [r["dollars"] - r["change"] * est.loc[r["term"], "CI_low"] for r in rows]
    err_hi = [r["change"] * est.loc[r["term"], "CI_high"] - r["dollars"] for r in rows]
    ax.barh(ypos, dollars, color="#4C72B0", alpha=0.85,
            xerr=[err_lo, err_hi], capsize=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9)
    ax.set_xlabel("dollar effect of realistic change")
    ax.set_title("Realistic trips")
    for i, r in enumerate(rows):
        ax.annotate(f"+${r['dollars']:.2f}", xy=(r["dollars"], ypos[i]),
                    xytext=(6, 0), textcoords="offset points", fontsize=9, va="center")
    ax.grid(True, alpha=0.3, axis="x")


def main() -> None:
    df = load_metered()
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    excluded = outlier_mask().reindex(df.index).fillna(False)
    kept = df[~excluded]
    model = fit_raw_hc3(kept, use_hour=True)

    est = estimation_table(model)
    rows = scenario_dollars(model, kept)

    fig, (ax_forest, ax_scen) = plt.subplots(1, 2, figsize=(14, 5.5))
    draw_forest(ax_forest, est)
    draw_scenarios(ax_scen, rows, est)
    n = int(model.nobs)
    fig.suptitle(f"Estimate size — raw fare HC3, |M|>10 excluded (N={n})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
