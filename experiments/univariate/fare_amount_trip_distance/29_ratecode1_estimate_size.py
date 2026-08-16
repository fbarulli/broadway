"""29: estimate size — meaningful-units effects + HC3 confidence intervals.

Interprets the step-28 model (raw fare ~ trip_distance + duration_minutes +
pickup_hour, HC3, modified-z outliers excluded): translates the distance
and duration coefficients into dollar effects for realistic trips, reports
standardized coefficients (beta_std = coef * sd_x / sd_y), and — the
estimation-first part — the 95% confidence intervals from the HC3
covariance instead of p-values. Persists the estimation table to
ratecode1_sample.json and a tracked CSV (ratecode1_sample_estimates.csv).
"""

from pathlib import Path

import pandas as pd
from statsmodels.regression.linear_model import RegressionResultsWrapper

from _common import RESULTS, WORKING_DATASET, load_metered
from _ols_bp import fit_raw_hc3, outlier_mask
from _tests import write_tests_json

ESTIMATES_CSV = RESULTS / "ratecode1_sample_estimates.csv"

ALPHA = 0.05  # 95% confidence intervals

# (label, predictor, realistic change in predictor units)
SCENARIOS = [
    ("5-mile trip", "trip_distance", 5.0),
    ("10 min waiting", "duration_minutes", 10.0),
    ("typical 1.6mi trip", "trip_distance", 1.6),
]


def estimation_table(model: RegressionResultsWrapper) -> pd.DataFrame:
    """coef / HC3 SE / 95% CI table (p-values deliberately absent)."""
    ci = model.conf_int(alpha=ALPHA)
    return pd.DataFrame({
        "coef": model.params,
        "HC3_SE": model.bse,
        "CI_low": ci[0],
        "CI_high": ci[1],
    })


def scenario_dollars(model: RegressionResultsWrapper, df: pd.DataFrame) -> list[dict]:
    """Dollar effect of each realistic scenario, from the fitted coefs."""
    rows = []
    for label, term, change in SCENARIOS:
        coef = float(model.params[term])
        rows.append({
            "label": label,
            "term": term,
            "change": change,
            "dollars": change * coef,
        })
    return rows


def standardized_coefs(model: RegressionResultsWrapper, df: pd.DataFrame) -> dict:
    """beta_std = coef * sd_x / sd_y for each predictor."""
    sd_y = float(df["fare_amount"].std())
    return {
        col: {
            "coef": float(model.params[col]),
            "sd_x": float(df[col].std()),
            "sd_y": sd_y,
            "beta_std": float(model.params[col]) * float(df[col].std()) / sd_y,
        }
        for col in ("trip_distance", "duration_minutes")
    }


def main() -> None:
    df = load_metered()
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    excluded = outlier_mask().reindex(df.index).fillna(False)
    kept = df[~excluded]
    model = fit_raw_hc3(kept, use_hour=True)

    print(f"estimate size for the step-28 model (N={len(kept)}, raw fare, HC3)\n")

    est = estimation_table(model)
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
    for row in scenario_dollars(model, kept):
        print(f"{row['label']}: +${row['dollars']:.2f}")

    print("\n=== standardized coefficients (beta_std = coef * sd_x / sd_y) ===")
    std = standardized_coefs(model, kept)
    for col, s in std.items():
        print(f"{col}: beta_std={s['beta_std']:.3f} "
              f"(sd_x={s['sd_x']:.2f}, sd_y={s['sd_y']:.2f})")

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
            "scenarios": scenario_dollars(model, kept),
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
