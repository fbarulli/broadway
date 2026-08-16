"""Shared OLS + Breusch-Pagan analysis/plot for this experiment (reused by 04 and 13)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from broadway.stats.regression import bp_jb, fit_ols

ALPHA = 0.05


def fit_and_plot_ols_bp(
    df: pd.DataFrame,
    formula: str,
    out_path: Path,
    suptitle: str | None = None,
) -> dict:
    """Fit OLS on df, plot residuals-vs-fitted + BP rejection region.

    Returns the bp_jb result dict (bp_stat, bp_pval, jb_stat, jb_pval,
    skew, kurtosis).
    """
    model = fit_ols(df, formula)
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

    z_crit = stats.norm.ppf(1 - ALPHA / 2)
    z_obs = float(np.sqrt(lm_stat))
    x = np.linspace(-4, 4, 500)
    pdf = stats.norm.pdf(x)
    ax_bp.plot(x, pdf, color="black")
    fill_color = "#d62728" if reject else "#aaaaaa"
    ax_bp.fill_between(x, pdf, where=(x <= -z_crit), color=fill_color, alpha=0.4)
    ax_bp.fill_between(x, pdf, where=(x >= z_crit), color=fill_color, alpha=0.4)
    ax_bp.axvline(-z_crit, color="black", linestyle="--", linewidth=1)
    ax_bp.axvline(z_crit, color="black", linestyle="--", linewidth=1)
    verdict = "reject H0 (heteroskedastic)" if reject else "fail to reject H0 (homoskedastic)"
    ax_bp.annotate(
        f"observed z = {z_obs:.1f}",
        xy=(3.9, 0.08), xytext=(2.0, 0.22),
        ha="right", va="bottom",
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9,
    )
    ax_bp.set_xlabel("z-score")
    ax_bp.set_ylabel("density (standard normal)")
    ax_bp.set_title(f"p = {p_value:.4f} → {verdict}")
    ax_bp.grid(True, alpha=0.3)

    if suptitle is not None:
        fig.suptitle(suptitle)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return result
