"""04: fit OLS on cleaned fares, show residuals, then run Breusch-Pagan.

Fits fare_amount ~ trip_distance on the cleaned sample, plots the
residuals-vs-fitted, and runs the Breusch-Pagan test, visualizing the p-value
as the shaded tails of a standard normal (z) distribution.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from _common import CLEAN_PARQUET, RESULTS
from broadway.stats.regression import bp_jb, fit_ols

OUT = RESULTS / f"{Path(__file__).stem}.png"

ALPHA = 0.05


def load_cleaned() -> pd.DataFrame:
    if not CLEAN_PARQUET.exists():
        raise FileNotFoundError(f"{CLEAN_PARQUET} not found — run 01_filtered_min_max_scatter.py first")
    return pd.read_parquet(CLEAN_PARQUET)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_cleaned()

    model = fit_ols(df, "fare_amount ~ trip_distance")
    result = bp_jb(model)
    lm_stat = result["bp_stat"]
    p_value = result["bp_pval"]
    reject = p_value < ALPHA

    fig, (ax_resid, ax_bp) = plt.subplots(1, 2, figsize=(14, 6))

    ax_resid.scatter(model.fittedvalues, model.resid, s=2, alpha=0.2)
    ax_resid.axhline(0, color="red", linewidth=1)
    ax_resid.set_xlabel("fitted values")
    ax_resid.set_ylabel("residuals")
    ax_resid.set_title("residuals vs fitted")
    ax_resid.grid(True, alpha=0.3)

    z_obs = float(np.sqrt(lm_stat))
    span = max(4.0, z_obs + 1.0)
    x = np.linspace(-span, span, 500)
    pdf = stats.norm.pdf(x)
    ax_bp.plot(x, pdf, color="black")
    fill_color = "#d62728" if reject else "#aaaaaa"
    ax_bp.fill_between(x, pdf, where=(x <= -z_obs), color=fill_color, alpha=0.4)
    ax_bp.fill_between(x, pdf, where=(x >= z_obs), color=fill_color, alpha=0.4)
    ax_bp.axvline(-z_obs, color="black", linestyle="--", linewidth=1)
    ax_bp.axvline(z_obs, color="black", linestyle="--", linewidth=1)
    verdict = "reject H0 (heteroskedastic)" if reject else "fail to reject H0 (homoskedastic)"
    ax_bp.set_xlabel("z-score")
    ax_bp.set_ylabel("density (standard normal)")
    ax_bp.set_title(f"p = {p_value:.4f} → {verdict}")
    ax_bp.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)

    print(f"cleaned rows: {len(df)}")
    print(f"Breusch-Pagan p-value: {p_value:.4f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
