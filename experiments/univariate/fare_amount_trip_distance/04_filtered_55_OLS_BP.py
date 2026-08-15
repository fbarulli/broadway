"""04: fit OLS on metered fares, show residuals, then run Breusch-Pagan.

Fits fare_amount ~ trip_distance on metered fares (fare_amount < $55), plots
the residuals-vs-fitted, and runs the Breusch-Pagan test, visualizing the
p-value as the shaded tail of a chi-square(1) distribution.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from broadway.stats.regression import bp_jb, fit_ols

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[2] / "experiments" / "results" / HERE.parents[0].name / HERE.name

METERED_CUTOFF = 55.0
ALPHA = 0.05


def load_metered() -> pd.DataFrame:
    path = RESULTS / "sample_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run 01_filtered_min_max_scatter.py first")
    df = pd.read_parquet(path)
    return df[df["fare_amount"] < METERED_CUTOFF]


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    metered = load_metered()

    model = fit_ols(metered, "fare_amount ~ trip_distance")
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

    df_chi2 = 1
    x = np.linspace(0, max(10.0, lm_stat + 4.0), 400)
    pdf = stats.chi2.pdf(x, df_chi2)
    ax_bp.plot(x, pdf, color="black")
    fill_color = "#d62728" if reject else "#aaaaaa"
    ax_bp.fill_between(x, pdf, where=(x >= lm_stat), color=fill_color, alpha=0.4)
    ax_bp.axvline(lm_stat, color="black", linestyle="--", linewidth=1)
    verdict = "reject H0 (heteroskedastic)" if reject else "fail to reject H0 (homoskedastic)"
    ax_bp.set_xlabel("Breusch-Pagan LM statistic")
    ax_bp.set_ylabel("density (chi-square, df=1)")
    ax_bp.set_title(f"p = {p_value:.4f} → {verdict}")
    ax_bp.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS / "04_filtered_55_OLS_BP.png", dpi=150)
    plt.close(fig)

    print(f"metered rows: {len(metered)}")
    print(f"Breusch-Pagan p-value: {p_value:.4f}")
    print(f"wrote {RESULTS / '04_filtered_55_OLS_BP.png'}")


if __name__ == "__main__":
    main()
