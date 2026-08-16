"""27: revert to RAW fare + capping + HC3, add pickup-hour (time of day).

Abandons the log transform (steps 18-26): y is the capped raw fare
(fare_amount clipped at its 99.5th percentile — every row kept) and the
regressors are capped trip_distance, duration_minutes, and the new
pickup_hour (0-23), the "hidden rule" of the meter. HC3 keeps the standard
errors valid under the fan-out. Diagnostics mirror steps 15/19:
residuals-vs-fitted + Q-Q with the BP/JB/skew/kurtosis legend, and the fit
is persisted to ratecode1_sample.json.
"""

from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper

from _common import RESULTS, WORKING_DATASET, load_metered
from _ols_bp import plot_resid_vs_fitted_qq, winsorize
from _tests import write_tests_json
from broadway.stats.regression import bp_jb

OUT = RESULTS / f"{Path(__file__).stem}.png"


def fit_final(df: pd.DataFrame) -> RegressionResultsWrapper:
    """Raw capped fare ~ capped distance + duration + pickup_hour, HC3."""
    X = sm.add_constant(df[["trip_distance", "duration_minutes", "pickup_hour"]])
    y = df["fare_amount"]
    return sm.OLS(y, X).fit(cov_type="HC3")


def main() -> None:
    df = load_metered()
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    capped, meta = winsorize(df)
    model = fit_final(capped)

    print(f"metered rows: {len(capped)}")
    print(
        f"cap (99.5th pct): fare=${meta['caps']['fare_amount']:.2f}, "
        f"distance={meta['caps']['trip_distance']:.2f} mi"
    )
    print(
        f"rows capped: fare → {meta['n_capped']['fare_amount']}, "
        f"distance → {meta['n_capped']['trip_distance']}"
    )
    print(model.summary())

    diag = bp_jb(model)
    print(
        f"R^2={model.rsquared:.4f} | BP stat={diag['bp_stat']:.1f} "
        f"(p={diag['bp_pval']:.2e}) | JB stat={diag['jb_stat']:.2e} | "
        f"skew={diag['skew']:.2f}, kurtosis={diag['kurtosis']:.1f}"
    )

    plot_resid_vs_fitted_qq(
        model,
        OUT,
        suptitle="RatecodeID == 1, raw capped fare + pickup hour (HC3)",
    )
    print(f"wrote {OUT}")

    results = {
        "hc3_fare_capped_trip_distance_duration_pickup_hour": {
            "method": (
                "raw capped fare ~ capped trip_distance + duration_minutes "
                "+ pickup_hour, HC3"
            ),
            "n": int(model.nobs),
            "rsquared": float(model.rsquared),
            "params": {name: float(v) for name, v in model.params.items()},
            "bse": {name: float(v) for name, v in model.bse.items()},
            "pvalues": {name: float(v) for name, v in model.pvalues.items()},
            "caps": meta["caps"],
            "n_capped": meta["n_capped"],
            "bp_jb": diag,
        }
    }
    out = write_tests_json(
        WORKING_DATASET,
        results,
        "27_ratecode1_raw_capped_hour_hc3.py",
        n_rows=len(pd.read_parquet(WORKING_DATASET)),
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
