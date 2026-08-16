"""26: winsorize fare_amount & trip_distance at the 99.5th percentile and re-evaluate.

Capping keeps every row but bounds the extremes: the model sees the 99.5th
percentile instead of the raw tail, so outlier influence is limited without
deleting records. Reuses the step-15/19 machinery (fit_log_hc3 +
plot_log_resid_qq + bp_jb): residual-vs-fitted + residual Q-Q for the
winsorized log-fare HC3 model, a before → after diagnostics table on the
console, and the comparison persisted to ratecode1_sample.json.
"""

from pathlib import Path

import pandas as pd
from statsmodels.regression.linear_model import RegressionResultsWrapper

from _common import RESULTS, WORKING_DATASET, load_metered
from _ols_bp import CAP_QUANTILE, fit_log_hc3, plot_log_resid_qq, winsorize
from _tests import write_tests_json
from broadway.stats.regression import bp_jb

OUT = RESULTS / f"{Path(__file__).stem}.png"


def summarize(model: RegressionResultsWrapper) -> dict:
    """Full diagnostics summary for one log-fare HC3 fit (JSON-ready)."""
    return {
        "n": int(model.nobs),
        "rsquared": float(model.rsquared),
        "params": {name: float(v) for name, v in model.params.items()},
        "bse": {name: float(v) for name, v in model.bse.items()},
        "pvalues": {name: float(v) for name, v in model.pvalues.items()},
        "bp_jb": bp_jb(model),
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
        rows.append((f"se_hc3 {name}", before["bse"][name], after["bse"][name]))
    rows += [
        ("R^2", before["rsquared"], after["rsquared"]),
        ("BP stat", before["bp_jb"]["bp_stat"], after["bp_jb"]["bp_stat"]),
        ("JB stat", before["bp_jb"]["jb_stat"], after["bp_jb"]["jb_stat"]),
        ("resid skew", before["bp_jb"]["skew"], after["bp_jb"]["skew"]),
        ("resid kurtosis", before["bp_jb"]["kurtosis"], after["bp_jb"]["kurtosis"]),
    ]
    print(f"{'metric':<18}{'before':>14}{'after':>14}")
    for label, b, a in rows:
        print(f"{label:<18}{b:>14.4g}{a:>14.4g}")


def main() -> None:
    df = load_metered()
    before_model = fit_log_hc3(df)
    win_df, meta = winsorize(df)
    after_model = fit_log_hc3(win_df)

    before = summarize(before_model)
    after = summarize(after_model)
    print(f"metered rows: {len(df)}")
    print_comparison(before, after, meta["caps"], meta["n_capped"])

    plot_log_resid_qq(
        after_model,
        OUT,
        suptitle="RatecodeID == 1, log-fare (HC3), 99.5pct winsorized",
    )
    print(f"wrote {OUT}")

    results = {
        "winsorize_99_5": {
            "method": "clip fare_amount & trip_distance at 99.5th percentile; refit log-fare HC3",
            "cap_quantile": CAP_QUANTILE,
            "caps": meta["caps"],
            "n_capped": meta["n_capped"],
            "before": before,
            "after": after,
        }
    }
    out = write_tests_json(
        WORKING_DATASET,
        results,
        "26_ratecode1_winsorize_evaluate.py",
        n_rows=len(pd.read_parquet(WORKING_DATASET)),
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
