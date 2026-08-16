"""19: residuals-vs-fitted + Q-Q plots for the log-fare HC3 model (step 18).

The bottom legend compares raw (non-log model, step 15) vs log-model
residual diagnostics, plus the raw vs log target distribution — the same
raw → log comparison as the platform's raw-vs-log Q-Q figure.
"""

from pathlib import Path

import numpy as np
from scipy import stats

from _common import RESULTS, load_metered
from _ols_bp import fit_log_hc3, plot_log_resid_qq
from broadway.stats.regression import bp_jb, fit_ols

OUT = RESULTS / f"{Path(__file__).stem}.png"


def main() -> None:
    df = load_metered()
    model = fit_log_hc3(df)
    raw_model = fit_ols(df, "fare_amount ~ trip_distance + duration_minutes")
    target = {
        "skew_raw": float(stats.skew(df["fare_amount"])),
        "kurt_raw": float(stats.kurtosis(df["fare_amount"])),
        "skew_log": float(stats.skew(np.log1p(df["fare_amount"]))),
        "kurt_log": float(stats.kurtosis(np.log1p(df["fare_amount"]))),
    }

    plot_log_resid_qq(
        model,
        OUT,
        suptitle="RatecodeID == 1, log-fare (HC3)",
        raw_result=bp_jb(raw_model),
        target=target,
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
