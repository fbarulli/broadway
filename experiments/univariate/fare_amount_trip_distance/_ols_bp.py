"""Shared OLS + Breusch-Pagan analysis/plot for this experiment (reused by 04, 13, 18, 19)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

from broadway.stats.regression import bp_jb, fit_ols
from matplotlib.lines import Line2D
from statsmodels.regression.linear_model import RegressionResultsWrapper

ALPHA = 0.05


def _fmt_p(p: float) -> str:
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def attach_stats_legend(fig, result: dict) -> None:
    """Bottom legend with BP/JB/skew/kurtosis, outside the axes.

    Mirrors src/broadway/discover/qq.py::attach_qq_legend (figure-level,
    horizontal columns, framed), but anchored in the bottom band reserved by
    subplots_adjust(bottom=...) — the legend never overlaps the graphs.
    """
    handles = [
        Line2D([0], [0], color="none", label=f"Breusch-Pagan: stat={result['bp_stat']:.2f}, {_fmt_p(result['bp_pval'])}"),
        Line2D([0], [0], color="none", label=f"Jarque-Bera: stat={result['jb_stat']:.2f}, {_fmt_p(result['jb_pval'])}"),
        Line2D([0], [0], color="none", label=f"skew={result['skew']:.2f}, kurtosis={result['kurtosis']:.2f}"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07),
        ncol=len(handles),
        fontsize=9,
        frameon=True,
        framealpha=0.85,
    )


def fit_log_hc3(df: pd.DataFrame) -> RegressionResultsWrapper:
    """OLS on log1p(fare_amount) ~ trip_distance + duration_minutes, HC3 SEs."""
    y = np.log1p(df["fare_amount"])
    X = sm.add_constant(df[["trip_distance", "duration_minutes"]])
    return sm.OLS(y, X).fit(cov_type="HC3")


def plot_log_resid_qq(
    model: RegressionResultsWrapper,
    out_path: Path,
    suptitle: str | None = None,
) -> None:
    """Log-fare model: seaborn residuals-vs-fitted + residual Q-Q, side by side.

    The figure is taller than the axes and subplots_adjust reserves a bottom
    band, so the diagnostics legend sits outside the graphs in its own space.
    """
    fig, (ax_resid, ax_qq) = plt.subplots(1, 2, figsize=(14, 7.5))
    fig.subplots_adjust(bottom=0.30, top=0.90)
    sns.scatterplot(x=model.fittedvalues, y=model.resid, alpha=0.2, s=10, ax=ax_resid)
    ax_resid.axhline(0, color="red", linestyle="--")
    ax_resid.set_xlabel("Predicted Log-Fare")
    ax_resid.set_ylabel("Residual")
    ax_resid.grid(True, alpha=0.3)
    sm.qqplot(model.resid, line="s", ax=ax_qq)
    ax_qq.set_title("Q-Q plot (residuals)")
    ax_qq.grid(True, alpha=0.3)
    attach_stats_legend(fig, bp_jb(model))
    n = int(model.nobs)
    fig.suptitle(f"{suptitle} (N={n})" if suptitle else f"N={n}")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def draw_resid_vs_fitted(ax, model: RegressionResultsWrapper) -> None:
    """Draw the residuals-vs-fitted panel on an existing axis."""
    ax.scatter(model.fittedvalues, model.resid, s=2, alpha=0.2)
    ax.axhline(0, color="red", linewidth=1)
    ax.set_xlabel("fitted values")
    ax.set_ylabel("residuals")
    ax.set_title("residuals vs fitted")
    ax.grid(True, alpha=0.3)


def plot_resid_vs_fitted(
    model: RegressionResultsWrapper,
    out_path: Path,
    suptitle: str | None = None,
) -> None:
    """Standalone residuals-vs-fitted scatter for a fitted model."""
    fig, ax = plt.subplots(figsize=(7, 6))
    draw_resid_vs_fitted(ax, model)
    n = int(model.nobs)
    fig.suptitle(f"{suptitle} (N={n})" if suptitle else f"N={n}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_resid_vs_fitted_qq(
    model: RegressionResultsWrapper,
    out_path: Path,
    suptitle: str | None = None,
) -> None:
    """Residuals-vs-fitted and residual Q-Q plot, side by side."""
    fig, (ax_resid, ax_qq) = plt.subplots(1, 2, figsize=(14, 6))
    draw_resid_vs_fitted(ax_resid, model)
    stats.probplot(model.resid, dist="norm", plot=ax_qq)
    ax_qq.set_title("Q-Q plot (residuals)")
    ax_qq.grid(True, alpha=0.3)
    n = int(model.nobs)
    fig.suptitle(f"{suptitle} (N={n})" if suptitle else f"N={n}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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

    draw_resid_vs_fitted(ax_resid, model)

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
        fig.suptitle(f"{suptitle} (N={len(df)})")
    else:
        fig.suptitle(f"N={len(df)}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return result
