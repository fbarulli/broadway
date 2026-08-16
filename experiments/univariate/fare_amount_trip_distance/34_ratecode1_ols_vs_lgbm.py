"""34: OLS vs LightGBM side-by-side on the identical 42,806 rows.

Parametric (OLS, raw fare ~ distance + duration) vs non-parametric (LGBM,
same two features) battle on the same data — no caps, no HC3, no outlier
exclusion (intentionally different from steps 28/29: this is a fair fight
on all rows). Trees partition rather than extrapolate, so an extreme trip
sits in its own leaf without pulling the fit. Compares MAE / RMSE / tail
MAE (the platform's baseline.evaluate tail benchmark), renders both
residual plots (the fan-out is a property of the data; trees add banding),
and records LGBM feature importance.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error

from _common import RESULTS, WORKING_DATASET, load_metered
from _tests import write_tests_json
from broadway.stats.baseline import train_lgbm

OUT = RESULTS / f"{Path(__file__).stem}.png"
METRICS_CSV = RESULTS / "ratecode1_sample_ols_lgbm.csv"

LGBM_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 5,
    "random_state": 42,
}
TAIL_QUANTILE = 0.95  # tail benchmark = most expensive 5% of trips


def fit_ols(df: pd.DataFrame) -> tuple[object, pd.DataFrame]:
    """Plain OLS (no HC3 — predictions are identical either way)."""
    X = sm.add_constant(df[["trip_distance", "duration_minutes"]])
    return sm.OLS(df["fare_amount"], X).fit(), X


def metrics(y: np.ndarray, preds: np.ndarray) -> dict:
    """MAE, RMSE, and tail MAE (top 5% of actual fares)."""
    mae = mean_absolute_error(y, preds)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    tail = y >= np.quantile(y, TAIL_QUANTILE)
    tail_mae = mean_absolute_error(y[tail], preds[tail])
    return {"mae": mae, "rmse": rmse, "tail_mae": tail_mae}


def draw_residuals(ax, preds: np.ndarray, resid: np.ndarray, title: str) -> None:
    ax.scatter(preds, resid, s=5, alpha=0.2, color="black")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("predicted fare ($)")
    ax.set_ylabel("residual ($)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def main() -> None:
    df = load_metered()
    X_raw = df[["trip_distance", "duration_minutes"]]
    y = df["fare_amount"].to_numpy()

    ols, X_const = fit_ols(df)
    ols_preds = ols.predict(X_const)
    lgbm = train_lgbm(X_raw, y, **LGBM_PARAMS)
    lgbm_preds = lgbm.predict(X_raw)

    ols_m = metrics(y, ols_preds)
    lgbm_m = metrics(y, lgbm_preds)
    print(f"rows: {len(df)} (all metered, no caps/exclusions) | "
          "X: trip_distance, duration_minutes | y: fare_amount (in-sample)")
    print(f"\n{'metric':<10}{'OLS':>10}{'LGBM':>10}{'delta':>10}")
    for k in ("mae", "rmse", "tail_mae"):
        print(f"{k:<10}{ols_m[k]:>10.2f}{lgbm_m[k]:>10.2f}{(lgbm_m[k] - ols_m[k]):>+10.2f}")
    print(f"\nMAE  gap: {lgbm_m['mae'] - ols_m['mae']:+.2f} "
          f"({(lgbm_m['mae'] / ols_m['mae'] - 1) * 100:+.1f}%)")
    print(f"RMSE gap: {lgbm_m['rmse'] - ols_m['rmse']:+.2f} "
          f"({(lgbm_m['rmse'] / ols_m['rmse'] - 1) * 100:+.1f}%)")

    imp = pd.Series(lgbm.feature_importances_,
                    index=X_raw.columns).sort_values(ascending=False)
    print("\n=== LGBM feature importance ===")
    print(imp.round(1).to_string())

    fig, (ax_ols, ax_lgbm) = plt.subplots(1, 2, figsize=(14, 5.5))
    draw_residuals(ax_ols, ols_preds, y - ols_preds, "OLS residuals vs fitted")
    draw_residuals(ax_lgbm, lgbm_preds, y - lgbm_preds, "LGBM residuals vs fitted")
    n = len(df)
    fig.suptitle(f"OLS vs LightGBM — same data (N={n})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}")

    out = write_tests_json(
        WORKING_DATASET,
        {
            "ols_vs_lgbm": {
                "method": ("same 42,806 metered rows; y = fare_amount; "
                           "X = trip_distance + duration_minutes; in-sample"),
                "n": n,
                "lgbm_params": LGBM_PARAMS,
                "metrics_ols": ols_m,
                "metrics_lgbm": lgbm_m,
                "deltas": {
                    k: round(float(lgbm_m[k] - ols_m[k]), 4)
                    for k in ("mae", "rmse", "tail_mae")
                },
                "lgbm_feature_importance": {
                    k: float(v) for k, v in imp.items()
                },
                "note": ("in-sample comparison; LGBM has capacity to memorize — "
                         "a holdout split would be a stricter test"),
            }
        },
        "34_ratecode1_ols_vs_lgbm.py",
        n_rows=len(pd.read_parquet(WORKING_DATASET)),
    )
    print(f"wrote {out}")

    pd.DataFrame([
        {"model": "OLS", **{k: round(ols_m[k], 4) for k in ("mae", "rmse", "tail_mae")}},
        {"model": "LGBM", **{k: round(lgbm_m[k], 4) for k in ("mae", "rmse", "tail_mae")}},
    ]).to_csv(METRICS_CSV, index=False)
    print(f"wrote {METRICS_CSV}")


if __name__ == "__main__":
    main()
