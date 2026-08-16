"""17: collect all analysis metrics for the working dataset (ratecode1).

Writes every metric for the working dataset (RatecodeID == 1) into its
results JSON (ratecode1_sample.json): VIF, plain OLS + HC3 fits for
fare ~ trip_distance (steps 13/14) and fare ~ trip_distance +
duration_minutes (step 15), plus Breusch-Pagan/Jarque-Bera diagnostics.
"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from _common import WORKING_DATASET, load_metered
from _ols_bp import fit_log_hc3
from _tests import write_tests_json
from broadway.stats.regression import bp_jb, fit_ols, fit_robust


def _model_named(model, attr: str) -> dict:
    """Model attribute as {param_name: float}; handles Series (plain) and ndarray (robust)."""
    values = getattr(model, attr)
    items = values.items() if isinstance(values, pd.Series) else zip(model.model.exog_names, values)
    return {k: float(v) for k, v in items}


def fit_model_entry(df: pd.DataFrame, formula: str, robust: bool) -> dict:
    model = fit_ols(df, formula)
    if robust:
        model = fit_robust(model)
    return {
        "n": int(model.nobs),
        "rsquared": float(model.rsquared),
        "params": _model_named(model, "params"),
        "bse": _model_named(model, "bse"),
        "pvalues": _model_named(model, "pvalues"),
    }


def compute_vif(df: pd.DataFrame) -> dict:
    X = sm.add_constant(df[["trip_distance", "duration_minutes"]])
    return {
        col: float(variance_inflation_factor(X.values, i))
        for i, col in enumerate(X.columns)
    }


def main() -> None:
    dist_df = pd.read_parquet(WORKING_DATASET)
    metered = load_metered()

    results = {
        "ols_fare_trip_distance": fit_model_entry(
            dist_df, "fare_amount ~ trip_distance", robust=False
        ),
        "hc3_fare_trip_distance": fit_model_entry(
            dist_df, "fare_amount ~ trip_distance", robust=True
        ),
        "bp_jb_fare_trip_distance": bp_jb(
            fit_ols(dist_df, "fare_amount ~ trip_distance")
        ),
        "ols_fare_trip_distance_duration": fit_model_entry(
            metered, "fare_amount ~ trip_distance + duration_minutes", robust=False
        ),
        "hc3_fare_trip_distance_duration": fit_model_entry(
            metered, "fare_amount ~ trip_distance + duration_minutes", robust=True
        ),
        "bp_jb_fare_trip_distance_duration": bp_jb(
            fit_ols(metered, "fare_amount ~ trip_distance + duration_minutes")
        ),
        "vif": compute_vif(metered),
    }

    log_model = fit_log_hc3(metered)
    results["ols_log_hc3_fare_trip_distance_duration"] = {
        "n": int(log_model.nobs),
        "rsquared": float(log_model.rsquared),
        "params": _model_named(log_model, "params"),
        "bse": _model_named(log_model, "bse"),
        "pvalues": _model_named(log_model, "pvalues"),
    }
    results["bp_jb_log_fare_trip_distance_duration"] = bp_jb(log_model)

    out = write_tests_json(
        WORKING_DATASET, results, "17_ratecode1_metrics.py", n_rows=len(dist_df)
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
