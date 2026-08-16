"""28: drop the modified-z outliers entirely; raw-fare model + pickup hour, HC3.

The "genuinely extreme" trips (modified z-score |M| > 10 on trip_distance
or fare_amount, step 22) are REMOVED — no capping. Fits raw fare ~
trip_distance + duration_minutes + pickup_hour (HC3) on the kept rows and
re-runs the residual diagnostics. Compares against the step-15 baseline
(raw, all rows, no hour) and the step-27 previous model (capped, all rows,
+ hour), reporting each model's constraints (rows / mask / target /
features) so every delta is qualified: the only fully controlled delta is
28 vs its no-hour twin on the identical kept rows.
"""

import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper

from _common import RESULTS, WORKING_DATASET, load_metered, load_working
from _ols_bp import modified_zscore, plot_resid_vs_fitted_qq
from _tests import tests_path_for, write_tests_json
from broadway.stats.regression import bp_jb

OUT = RESULTS / f"{Path(__file__).stem}.png"

MASK_THRESHOLD = 10.0  # step-22 modified-z |M| threshold (union over fare/distance)

# (pair label, new model, old model, comparable, reason)
DELTA_PAIRS = [
    ("new 28 vs previous 27", "new 28", "previous 27", False,
     "rows differ (kept vs all) and target differs (raw vs capped)"),
    ("new 28 vs baseline 15", "new 28", "baseline 15", False,
     "rows differ (kept vs all) and pickup_hour added"),
    ("new 28 vs new 28 (no hour)", "new 28", "new 28 (no hour)", True,
     "identical rows and target; only pickup_hour differs"),
]


def outlier_mask() -> pd.Series:
    """Step-22 mask: |modified z| > 10 on trip_distance or fare_amount."""
    working = load_working()
    z_dist = modified_zscore(working["trip_distance"]).abs()
    z_fare = modified_zscore(working["fare_amount"]).abs()
    return (z_dist > MASK_THRESHOLD) | (z_fare > MASK_THRESHOLD)


def fit_excluded(df: pd.DataFrame, use_hour: bool) -> RegressionResultsWrapper:
    """HC3 fit of raw fare on kept rows; optional pickup_hour feature."""
    cols = ["trip_distance", "duration_minutes"] + (["pickup_hour"] if use_hour else [])
    X = sm.add_constant(df[cols])
    return sm.OLS(df["fare_amount"], X).fit(cov_type="HC3")


def model_entry(model: RegressionResultsWrapper, **extra: object) -> dict:
    """JSON-ready fit entry (params/bse/pvalues/bp_jb) plus extras."""
    diag = bp_jb(model)
    return {
        "n": int(model.nobs),
        "rsquared": float(model.rsquared),
        "params": {name: float(v) for name, v in model.params.items()},
        "bse": {name: float(v) for name, v in model.bse.items()},
        "pvalues": {name: float(v) for name, v in model.pvalues.items()},
        "bp_jb": diag,
        **extra,
    }


def print_models(models: list[dict]) -> None:
    """Print the constraints + headline metrics table."""
    print("\n=== model comparison (constraints) ===")
    print(f"{'model':<20} {'n':>8} {'mask':<16} {'target':<8} {'features':<16} {'R^2':>9} {'kurt':>8}")
    for m in models:
        print(f"{m['name']:<20} {m['n']:>8} {m['mask']:<16} {m['target']:<8}"
              f" {m['features']:<16} {m['rsquared']:>9.4f} {m['kurtosis']:>8.2f}")


def build_deltas(models: list[dict]) -> list[dict]:
    """Deltas (new − old) for each pair, flagged direct vs qualified."""
    by_name = {m["name"]: m for m in models}
    deltas = []
    for label, new_name, old_name, comparable, reason in DELTA_PAIRS:
        nw, ol = by_name[new_name], by_name[old_name]
        deltas.append({
            "pair": label,
            "delta_rsquared": round(float(nw["rsquared"]) - float(ol["rsquared"]), 4),
            "delta_kurtosis": round(float(nw["kurtosis"]) - float(ol["kurtosis"]), 2),
            "comparable": comparable,
            "reason": reason,
        })
    return deltas


def print_deltas(deltas: list[dict]) -> None:
    print("\n=== deltas (new − old) ===")
    for d in deltas:
        tag = "DIRECT" if d["comparable"] else "qualified"
        print(f"{d['pair']}: ΔR2={d['delta_rsquared']:+.4f} "
              f"Δkurt={d['delta_kurtosis']:+.2f} [{tag}] — {d['reason']}")


def main() -> None:
    mask = outlier_mask()
    df = load_metered()
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    excluded = mask.reindex(df.index).fillna(False)
    kept = df[~excluded]

    model = fit_excluded(kept, use_hour=True)
    model_no_hour = fit_excluded(kept, use_hour=False)
    diag = bp_jb(model)

    print(f"metered rows: {len(df)} | removed (|M|>{MASK_THRESHOLD} union): "
          f"{int(excluded.sum())} | kept: {len(kept)}")
    print(model.summary())
    print(f"R^2={model.rsquared:.4f} | resid kurtosis={diag['kurtosis']:.2f} | "
          f"JB={diag['jb_stat']:.2e} | BP={diag['bp_stat']:.1f}")

    plot_resid_vs_fitted_qq(
        model,
        OUT,
        suptitle="Excluded Outliers: raw fare + pickup hour (HC3)",
    )
    print(f"wrote {OUT}")

    results = json.loads(tests_path_for(WORKING_DATASET).read_text())["results"]
    base = results.get("hc3_fare_trip_distance_duration", {})
    base_bp = results.get("bp_jb_fare_trip_distance_duration", {})
    prev = results.get("hc3_fare_capped_trip_distance_duration_pickup_hour", {})
    mask_label = f"|M|>{MASK_THRESHOLD} union"
    models = [
        {"name": "baseline 15", "n": base.get("n"), "mask": "none", "target": "raw",
         "features": "dist + dur", "rsquared": base.get("rsquared"),
         "kurtosis": base_bp.get("kurtosis")},
        {"name": "previous 27", "n": prev.get("n"), "mask": "none", "target": "capped",
         "features": "dist + dur + hour", "rsquared": prev.get("rsquared"),
         "kurtosis": prev.get("bp_jb", {}).get("kurtosis")},
        {"name": "new 28", "n": int(model.nobs), "mask": mask_label, "target": "raw",
         "features": "dist + dur + hour", "rsquared": float(model.rsquared),
         "kurtosis": diag["kurtosis"]},
        {"name": "new 28 (no hour)", "n": int(model_no_hour.nobs), "mask": mask_label,
         "target": "raw", "features": "dist + dur",
         "rsquared": float(model_no_hour.rsquared),
         "kurtosis": float(bp_jb(model_no_hour)["kurtosis"])},
    ]
    deltas = build_deltas(models)
    print_models(models)
    print_deltas(deltas)

    out = write_tests_json(
        WORKING_DATASET,
        {
            "hc3_fare_trip_distance_duration_pickup_hour_excluded": model_entry(
                model,
                method=("raw fare ~ dist + duration + pickup_hour, HC3, "
                        "modified-z |M|>10 outliers removed"),
                n_removed=int(excluded.sum()),
                mask_threshold=MASK_THRESHOLD,
            ),
            "hc3_fare_trip_distance_duration_excluded": model_entry(
                model_no_hour,
                method=("raw fare ~ dist + duration, HC3, "
                        "modified-z |M|>10 outliers removed (no hour)"),
                n_removed=int(excluded.sum()),
                mask_threshold=MASK_THRESHOLD,
            ),
            "model_comparison": {
                "source_script": "28_ratecode1_excluded_outliers_hc3.py",
                "note": ("deltas are DIRECT only when rows, target and features "
                         "all match; otherwise qualified"),
                "models": models,
                "deltas": deltas,
            },
        },
        "28_ratecode1_excluded_outliers_hc3.py",
        n_rows=len(pd.read_parquet(WORKING_DATASET)),
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
