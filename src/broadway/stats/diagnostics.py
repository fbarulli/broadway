"""Residual diagnostics: Breusch-Pagan, Jarque-Bera, Durbin-Watson, plots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson as _durbin_watson
from statsmodels.stats.stattools import jarque_bera

from broadway import viz
from broadway.stats.diagnostic_models import DiagnosticResult


def bp_test(resid: np.ndarray, exog: np.ndarray) -> tuple[float, float]:
    statistic, p_value, _, _ = het_breuschpagan(resid, exog)
    return float(statistic), float(p_value)


def jb_test(resid: np.ndarray) -> tuple[float, float, float, float]:
    statistic, p_value, skew, kurtosis = jarque_bera(resid)
    return float(statistic), float(p_value), float(skew), float(kurtosis)


def durbin_watson(resid: np.ndarray) -> float:
    return float(_durbin_watson(resid))


def _plot_residuals_vs_fitted(ax, model) -> None:
    fitted = model.fittedvalues
    resid = model.resid
    ax.scatter(fitted, resid, s=2, alpha=0.15, color=viz.palette_colors(1)[0])
    ax.axhline(0, color="red", linewidth=1)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")
    viz.despine(ax)


def plot_residuals_vs_fitted(model: object, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_residuals_vs_fitted(ax, model)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def mean_specification_diagnostic(model: object, out_path: str) -> DiagnosticResult:
    plot_residuals_vs_fitted(model, out_path)
    return DiagnosticResult(
        question="Is the mean relationship correctly specified?",
        evidence=[f"residual-vs-fitted plot persisted at {out_path}"],
        ramification=(
            "Residual-vs-fitted plots may reveal systematic structure that suggests "
            "the assumed mean function is inadequate; if such structure is observed, "
            "consider an appropriate transformation, nonlinear term, spline, or "
            "interaction, then refit and re-diagnose."
        ),
    )


def _bp_statistics(model) -> tuple[float, float]:
    return bp_test(model.resid, model.model.exog)


def constant_variance_diagnostic(model: object, out_path: str) -> DiagnosticResult:
    plot_residuals_vs_fitted(model, out_path)
    statistic, p_value = _bp_statistics(model)
    return DiagnosticResult(
        question="Is the error variance constant?",
        evidence=[
            f"residual-vs-fitted plot persisted at {out_path}",
            f"Breusch-Pagan statistic={statistic:.4f}, p={p_value:.4f}",
        ],
        ramification=(
            "Heteroskedasticity does not bias OLS point estimates but invalidates "
            "conventional standard errors, p-values, and confidence intervals; if "
            "detected, refit with HC3 robust standard errors before trusting "
            "inference on the coefficients."
        ),
    )


def plot_residuals(model, out_path: str) -> None:
    resid = model.resid

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    _plot_residuals_vs_fitted(axes[0], model)

    sm.qqplot(resid, line="45", ax=axes[1], markersize=2, alpha=0.15)
    axes[1].set_title("Q-Q Plot of Residuals")

    axes[2].hist(resid, bins=100, color=viz.palette_colors(1)[0])
    axes[2].set_title("Residual Distribution")
    axes[2].set_xlabel("Residual")

    for ax in axes:
        viz.despine(ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
