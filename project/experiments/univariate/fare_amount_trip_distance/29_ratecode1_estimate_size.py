"""29: estimate size — meaningful-units effects, HC3 CIs, and all plot variants.

Interprets the step-28 model (raw fare ~ trip_distance + duration_minutes +
pickup_hour, HC3, modified-z outliers excluded): translates the distance
and duration coefficients into dollar effects for realistic trips, reports
standardized coefficients (beta_std = coef * sd_x / sd_y), and — the
estimation-first part — the 95% confidence intervals from the HC3
covariance instead of p-values.

Persists the estimation table to ratecode1_sample.json and a tracked CSV
(29_ratecode1_estimate_size.csv), and renders one 2x2 figure with all plot
variants: (1) coefficient forest plot in dollars, (2) realistic-trip effect
bars with CI whiskers, (3) standardized-coefficient forest plot, (4) the
classic minimal single-panel forest plot. All numbers come from the fitted
model — nothing hardcoded.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from _common import RESULTS, WORKING_DATASET, load_metered
from _ols_bp import (
    ALPHA,
    estimation_table,
    fit_raw_hc3,
    outlier_mask,
    scenario_dollars,
    standardized_coefs,
)
from _tests import write_tests_json

OUT = RESULTS / f"{Path(__file__).stem}.png"
ESTIMATES_CSV = RESULTS / f"{Path(__file__).stem}.csv"

TERM_LABELS = {
    "const": "intercept ($)",
    "trip_distance": "trip_distance ($/mi)",
    "duration_minutes": "duration_minutes ($/min)",
    "pickup_hour": "pickup_hour ($/hr-of-day)",
}


def draw_forest(ax, est: pd.DataFrame, annotate: bool = True) -> None:
    """Coefficient plot: point estimate + 95% HC3 CI per term, zero line."""
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
    if annotate:
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
    for i, r in enumerate(rows):
        ax.annotate(f"+${r['dollars']:.2f}", xy=(r["dollars"], ypos[i]),
                    xytext=(6, 0), textcoords="offset points", fontsize=9, va="center")
    ax.grid(True, alpha=0.3, axis="x")


def draw_std_forest(ax, est: pd.DataFrame, std: dict) -> None:
    """Standardized coefficient plot: beta_std with CIs scaled by sd_x / sd_y."""
    terms = list(std.keys())
    ypos = list(range(len(terms)))[::-1]
    beta = [std[t]["beta_std"] for t in terms]
    scale = {t: std[t]["sd_x"] / std[t]["sd_y"] for t in terms}
    lo = [est.loc[t, "CI_low"] * scale[t] for t in terms]
    hi = [est.loc[t, "CI_high"] * scale[t] for t in terms]
    ax.errorbar(beta, ypos, xerr=[[b - l for b, l in zip(beta, lo)],
                                  [h - b for h, b in zip(hi, beta)]],
                fmt="o", color="black", ecolor="black", capsize=3, ms=5)
    ax.axvline(0, color="black", linestyle=":", linewidth=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(terms, fontsize=9)
    ax.set_xlabel("standardized effect (β*, 95% CI)")
    for i, t in enumerate(terms):
        ax.annotate(f"{beta[i]:.3f}", xy=(hi[i], ypos[i]), xytext=(6, 0),
                    textcoords="offset points", fontsize=8, va="center")
    ax.grid(True, alpha=0.3, axis="x")


def main() -> None:
    df = load_metered()
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    excluded = outlier_mask().reindex(df.index).fillna(False)
    kept = df[~excluded]
    model = fit_raw_hc3(kept, use_hour=True)

    est = estimation_table(model)
    rows = scenario_dollars(model, kept)
    std = standardized_coefs(model, kept)

    print(f"estimate size for the step-28 model (N={len(kept)}, raw fare, HC3)\n")
    print("=== coefficients + 95% HC3 confidence intervals ===")
    print(est.round(4).to_string())

    dist_ci = est.loc["trip_distance", ["CI_low", "CI_high"]]
    print(f"\n$ per mile: {est.loc['trip_distance', 'coef']:.3f} "
          f"(95% CI [{dist_ci['CI_low']:.3f}, {dist_ci['CI_high']:.3f}])")
    hour_ci = est.loc["pickup_hour", ["CI_low", "CI_high"]]
    print(f"$ per hour-of-day: {est.loc['pickup_hour', 'coef']:.4f} "
          f"(95% CI [{hour_ci['CI_low']:.4f}, {hour_ci['CI_high']:.4f}]) — "
          "statistically significant, economically negligible")

    print("\n=== meaningful-units scenarios ===")
    for row in rows:
        print(f"{row['label']}: +${row['dollars']:.2f}")

    print("\n=== standardized coefficients (beta_std = coef * sd_x / sd_y) ===")
    for col, s in std.items():
        print(f"{col}: beta_std={s['beta_std']:.3f} "
              f"(sd_x={s['sd_x']:.2f}, sd_y={s['sd_y']:.2f})")

    fig, ((ax_forest, ax_scen), (ax_std, ax_min)) = plt.subplots(2, 2, figsize=(15, 10))
    draw_forest(ax_forest, est)
    ax_forest.set_title("Coefficients ($, 95% HC3 CI)")
    draw_scenarios(ax_scen, rows, est)
    ax_scen.set_title("Realistic trips")
    draw_std_forest(ax_std, est, std)
    ax_std.set_title("Standardized coefficients")
    draw_forest(ax_min, est, annotate=False)
    ax_min.set_title("Coefficients — classic single panel")
    n = int(model.nobs)
    fig.suptitle(f"Estimate size — raw fare HC3, |M|>10 excluded (N={n})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}")

    est_out = est.rename(columns={
        "HC3_SE": "hc3_se", "CI_low": "ci_low", "CI_high": "ci_high",
    }).to_dict("index")
    results = {
        "estimate_size_hc3_excluded": {
            "method": "step-28 model: raw fare ~ distance + duration + pickup_hour, "
                      "HC3, modified-z |M|>10 excluded",
            "n": int(model.nobs),
            "alpha": ALPHA,
            "estimation": {
                term: {
                    "coef": float(v["coef"]),
                    "hc3_se": float(v["hc3_se"]),
                    "ci_low": float(v["ci_low"]),
                    "ci_high": float(v["ci_high"]),
                }
                for term, v in est_out.items()
            },
            "scenarios": rows,
            "standardized": std,
        }
    }
    out = write_tests_json(
        WORKING_DATASET,
        results,
        "29_ratecode1_estimate_size.py",
        n_rows=len(pd.read_parquet(WORKING_DATASET)),
    )
    print(f"wrote {out}")

    csv_rows = []
    for term, v in results["estimate_size_hc3_excluded"]["estimation"].items():
        s = std.get(term, {})
        csv_rows.append({
            "term": term,
            "coef": v["coef"],
            "hc3_se": v["hc3_se"],
            "ci_low": v["ci_low"],
            "ci_high": v["ci_high"],
            "beta_std": s.get("beta_std", ""),
        })
    pd.DataFrame(csv_rows).to_csv(ESTIMATES_CSV, index=False)
    print(f"wrote {ESTIMATES_CSV}")


if __name__ == "__main__":
    main()
