"""20: inspect the worst log-fare predictions (extreme negative residuals).

Adds predictions and residuals back to the working frame, filters trips
whose log residual is below -2 (the model badly over-predicted fare), and
prints the raw data for the worst trips, worst first. Result summary is
persisted to the working dataset's results JSON.
"""

import pandas as pd

from _common import WORKING_DATASET, load_metered
from _ols_bp import fit_log_hc3
from _tests import write_tests_json

THRESHOLD = -2.0


def main() -> None:
    df = load_metered()
    model = fit_log_hc3(df)

    df["predicted_log_fare"] = model.fittedvalues
    df["log_residuals"] = model.resid

    worst = df[df["log_residuals"] < THRESHOLD].sort_values("log_residuals")
    print(f"trips with log_residuals < {THRESHOLD}: {len(worst)} of {len(df)}")
    print(
        worst[["trip_distance", "duration_minutes", "fare_amount", "log_residuals"]]
        .head(10)
        .to_string(index=False)
    )

    results = {
        "worst_predictions_log": {
            "threshold": THRESHOLD,
            "n": int(len(worst)),
            "head": worst[
                ["trip_distance", "duration_minutes", "fare_amount", "log_residuals"]
            ]
            .head(10)
            .to_dict("records"),
        }
    }
    out = write_tests_json(
        WORKING_DATASET,
        results,
        "20_ratecode1_worst_predictions.py",
        n_rows=len(pd.read_parquet(WORKING_DATASET)),
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
