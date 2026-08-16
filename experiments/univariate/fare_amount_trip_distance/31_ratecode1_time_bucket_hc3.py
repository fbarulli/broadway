"""31: categorical time buckets (NYC surcharge rules) instead of linear pickup_hour.

Replaces the continuous pickup_hour (economically negligible, step 29) with
rule-based time buckets on the same kept rows as steps 28/29 (modified-z
outliers excluded): day = 6-15h (reference group), peak = 16-19h (+$1.00
rule), overnight = otherwise (+$0.50 rule). HC3 OLS of raw fare ~
trip_distance + duration_minutes + time_peak + time_overnight, with 95% HC3
confidence intervals so the estimated surcharges can be compared against
the NYC rule values.
"""

from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper

from _common import RESULTS, WORKING_DATASET, load_metered
from _ols_bp import OUTLIER_Z_THRESHOLD, estimation_table, outlier_mask
from _tests import write_tests_json
from broadway.stats.regression import bp_jb

ESTIMATES_CSV = RESULTS / "ratecode1_sample_time_bucket_estimates.csv"


def time_bucket(hour: int) -> str:
    """NYC-style surcharge bucket: day (6-15, ref), peak (16-19, +$1.00), overnight (else, +$0.50)."""
    if 6 <= hour < 16:
        return "day"
    if 16 <= hour < 20:
        return "peak"
    return "overnight"


def fit_time_bucket(df: pd.DataFrame) -> RegressionResultsWrapper:
    """HC3 raw fare ~ distance + duration + peak/overnight dummies (day reference)."""
    buckets = df["pickup_hour"].map(time_bucket)
    X = df[["trip_distance", "duration_minutes"]].copy()
    dummies = pd.get_dummies(buckets, prefix="time")
    # get_dummies yields bool columns in pandas 2.x; statsmodels needs numeric
    X = pd.concat([X, dummies[["time_peak", "time_overnight"]]], axis=1).astype(float)
    X = sm.add_constant(X)
    return sm.OLS(df["fare_amount"], X).fit(cov_type="HC3")


def main() -> None:
    df = load_metered()
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    excluded = outlier_mask().reindex(df.index).fillna(False)
    kept = df[~excluded]
    model = fit_time_bucket(kept)

    counts = kept["pickup_hour"].map(time_bucket).value_counts()
    print(f"rows: {len(kept)} (|M|>{OUTLIER_Z_THRESHOLD} excluded) | buckets: "
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

    est_out = est.rename(columns={
        "HC3_SE": "hc3_se", "CI_low": "ci_low", "CI_high": "ci_high",
    }).to_dict("index")
    results = {
        "hc3_fare_trip_distance_duration_time_bucket": {
            "method": "raw fare ~ distance + duration + time_peak + time_overnight "
                      "(day ref), HC3; NYC-style time buckets",
            "bucket_def": {
                "day": "6 <= hour < 16 (reference)",
                "peak": "16 <= hour < 20 (+$1.00 rule)",
                "overnight": "else (+$0.50 rule)",
            },
            "n": int(model.nobs),
            "rsquared": float(model.rsquared),
            "params": {name: float(v) for name, v in model.params.items()},
            "bse": {name: float(v) for name, v in model.bse.items()},
            "pvalues": {name: float(v) for name, v in model.pvalues.items()},
            "bp_jb": diag,
            "n_removed": int(excluded.sum()),
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
        }
    }
    out = write_tests_json(
        WORKING_DATASET,
        results,
        "31_ratecode1_time_bucket_hc3.py",
        n_rows=len(pd.read_parquet(WORKING_DATASET)),
    )
    print(f"wrote {out}")

    csv_rows = []
    for term, v in results["hc3_fare_trip_distance_duration_time_bucket"]["estimation"].items():
        csv_rows.append({
            "term": term, "coef": v["coef"], "hc3_se": v["hc3_se"],
            "ci_low": v["ci_low"], "ci_high": v["ci_high"],
        })
    pd.DataFrame(csv_rows).to_csv(ESTIMATES_CSV, index=False)
    print(f"wrote {ESTIMATES_CSV}")


if __name__ == "__main__":
    main()
