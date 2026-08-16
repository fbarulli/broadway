"""26: winsorize fare_amount & trip_distance at the 99.5th percentile and re-evaluate.

Capping keeps every row but bounds the extremes: the model sees the 99.5th
percentile instead of the raw tail, so outlier influence is limited without
deleting records. Reuses the step-15/19 machinery (fit_log_hc3 +
plot_log_resid_qq + bp_jb): residual-vs-fitted + residual Q-Q for the
winsorized log-fare HC3 model, and a before → after diagnostics table on
the console (R^2, coefficients, HC3 SEs, BP/JB, skew/kurtosis).
"""

from pathlib import Path

import pandas as pd
from statsmodels.regression.linear_model import RegressionResultsWrapper

from _common import RESULTS, load_metered
from _ols_bp import fit_log_hc3, plot_log_resid_qq
from broadway.stats.regression import bp_jb

OUT = RESULTS / f"{Path(__file__).stem}.png"

CAP_QUANTILE = 0.995  # winsorization cap, just below the outlier threshold
CAPPED_COLUMNS = ("fare_amount", "trip_distance")


def winsorize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Cap the capped columns at their 99.5th percentiles; return df + caps."""
    caps = {col: float(df[col].quantile(CAP_QUANTILE)) for col in CAPPED_COLUMNS}
    out = df.copy()
    for col, cap in caps.items():
        out[col] = out[col].clip(upper=cap)
    n_capped = {col: int((df[col] > cap).sum()) for col, cap in caps.items()}
    return out, {"caps": caps, "n_capped": n_capped}


def summarize(model: RegressionResultsWrapper) -> dict:
    """Diagnostics summary for one log-fare HC3 fit."""
    diag = bp_jb(model)
    return {
        "r2": float(model.rsquared),
        "params": {name: float(v) for name, v in model.params.items()},
        "se": {name: float(v) for name, v in model.bse.items()},
        **diag,
    }


def print_comparison(before: dict, after: dict, caps: dict, n_capped: dict) -> None:
    """Print the before → after winsorization comparison table."""
    print(
        f"cap (99.5th pct): fare=${caps['fare_amount']:.2f}, "
        f"distance={caps['trip_distance']:.2f} mi"
    )
    print(
        f"rows capped: fare → {n_capped['fare_amount']}, "
        f"distance → {n_capped['trip_distance']}"
    )
    rows = []
    for name in before["params"]:
        rows.append((f"coef {name}", before["params"][name], after["params"][name]))
        rows.append((f"se_hc3 {name}", before["se"][name], after["se"][name]))
    rows += [
        ("R^2", before["r2"], after["r2"]),
        ("BP stat", before["bp_stat"], after["bp_stat"]),
        ("JB stat", before["jb_stat"], after["jb_stat"]),
        ("resid skew", before["skew"], after["skew"]),
        ("resid kurtosis", before["kurtosis"], after["kurtosis"]),
    ]
    print(f"{'metric':<18}{'before':>14}{'after':>14}")
    for label, b, a in rows:
        print(f"{label:<18}{b:>14.4g}{a:>14.4g}")


def main() -> None:
    df = load_metered()
    before_model = fit_log_hc3(df)
    win_df, meta = winsorize(df)
    after_model = fit_log_hc3(win_df)

    print(f"metered rows: {len(df)}")
    print_comparison(summarize(before_model), summarize(after_model),
                     meta["caps"], meta["n_capped"])

    plot_log_resid_qq(
        after_model,
        OUT,
        suptitle="RatecodeID == 1, log-fare (HC3), 99.5pct winsorized",
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
