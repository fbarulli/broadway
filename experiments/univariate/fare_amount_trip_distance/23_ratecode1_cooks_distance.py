"""23: Cook's distance — influential observations for the log-fare model.

Single plot: Cook's distance per trip (y) vs predicted log-fare (x), with
the classic 4/n threshold line and influential points (D > 4/n) in red.
Complements the residual-based worst-predictions analysis (steps 20/21):
those found trips with big residuals; this finds trips that dominate the
fit itself.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from _common import RESULTS, WORKING_DATASET, load_metered
from _ols_bp import fit_log_hc3
from _tests import write_tests_json

OUT = RESULTS / f"{Path(__file__).stem}.png"


def main() -> None:
    df = load_metered()
    model = fit_log_hc3(df)
    cooks, _ = model.get_influence().cooks_distance

    n = len(df)
    threshold = 4 / n
    infl_mask = cooks > threshold
    n_influential = int(infl_mask.sum())
    print(f"influential trips (Cook's D > 4/n = {threshold:.2e}): {n_influential} of {n}")

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1,
               label=f"threshold 4/n = {threshold:.2e}")
    ax.scatter(model.fittedvalues, cooks, s=5, alpha=0.2, color="gray")
    ax.scatter(model.fittedvalues[infl_mask], cooks[infl_mask], s=20,
               color="red", edgecolor="black", label=f"influential (n={n_influential})")
    ax.set_xlabel("Predicted Log-Fare")
    ax.set_ylabel("Cook's distance")
    ax.set_title(f"Cook's distance — log-fare model (N={n})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}")

    top = (
        df.loc[infl_mask, ["trip_distance", "duration_minutes", "fare_amount"]]
        .assign(cooks_d=cooks[infl_mask])
        .sort_values("cooks_d", ascending=False)
        .head(20)
    )
    results = {
        "cooks_distance": {
            "method": "D > 4/n on log-fare OLS (HC3 fit)",
            "threshold": float(threshold),
            "n_influential": n_influential,
            "top": top.to_dict("records"),
        }
    }
    out = write_tests_json(
        WORKING_DATASET,
        results,
        "23_ratecode1_cooks_distance.py",
        n_rows=len(pd.read_parquet(WORKING_DATASET)),
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
