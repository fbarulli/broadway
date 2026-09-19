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


def _influence_statistics(model) -> tuple[np.ndarray, np.ndarray]:
    infl = model.get_influence()
    cooks, _ = infl.cooks_distance
    return cooks, infl.hat_matrix_diag


def _plot_cooks_distance(ax, model) -> None:
    cooks, _ = _influence_statistics(model)
    threshold = 4 / len(cooks)
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1)
    ax.scatter(
        np.arange(len(cooks)),
        cooks,
        s=2,
        alpha=0.15,
        color=viz.palette_colors(1)[0],
    )
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Cook's distance")
    ax.set_title("Cook's distance by observation")
    viz.despine(ax)


def plot_cooks_distance(model: object, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_cooks_distance(ax, model)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def influence_diagnostic(model: object, out_path: str) -> DiagnosticResult:
    plot_cooks_distance(model, out_path)
    cooks, hat = _influence_statistics(model)
    n = len(cooks)
    threshold = 4 / n
    n_influential = int(np.sum(cooks > threshold))
    return DiagnosticResult(
        question="Is the result being driven by a few observations?",
        evidence=[
            f"Cook's-distance plot persisted at {out_path}",
            (
                f"max Cook's distance={float(np.max(cooks)):.4f}, "
                f"{n_influential} of {n} observations exceed 4/n={threshold:.4f}, "
                f"max leverage={float(np.max(hat)):.4f}"
            ),
        ],
        ramification=(
            "A few high-leverage or high-influence observations can dominate the "
            "fit, so the reported results may not be robust; inspect the flagged "
            "observations, and if the conclusions change when they are removed, "
            "consider robust regression or report the sensitivity explicitly."
        ),
    )


def _plot_residuals_qq(ax, model) -> None:
    resid = model.resid
    sm.qqplot(resid, line="45", ax=ax, markersize=2, alpha=0.15)
    ax.set_title("Q-Q Plot of Residuals")
    viz.despine(ax)


def _plot_residuals_histogram(ax, model) -> None:
    resid = model.resid
    ax.hist(resid, bins=100, color=viz.palette_colors(1)[0])
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residual")
    viz.despine(ax)


def plot_residuals_qq(model: object, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_residuals_qq(ax, model)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_residuals_histogram(model: object, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_residuals_histogram(ax, model)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _jb_statistics(model) -> tuple[float, float, float, float]:
    return jb_test(model.resid)


def residual_distribution_diagnostic(model: object, out_path: str) -> DiagnosticResult:
    plot_residuals_qq(model, out_path)
    statistic, p_value, skew, kurtosis = _jb_statistics(model)
    return DiagnosticResult(
        question="Is residual non-normality problematic for inference?",
        evidence=[
            f"Q-Q plot of residuals persisted at {out_path}",
            (
                f"Jarque-Bera statistic={statistic:.4f}, p={p_value:.4f}, "
                f"skew={skew:.4f}, kurtosis={kurtosis:.4f}"
            ),
        ],
        ramification=(
            "Residual non-normality is primarily a concern for small samples where "
            "normal-theory inference is unreliable; if the sample is small and the "
            "residuals depart substantially from normality, consider a "
            "transformation, robust/nonparametric inference, or bootstrap-based "
            "inference rather than relying on normal-theory p-values."
        ),
    )


def plot_residuals(model, out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    _plot_residuals_vs_fitted(axes[0], model)
    _plot_residuals_qq(axes[1], model)
    _plot_residuals_histogram(axes[2], model)

    for ax in axes:
        viz.despine(ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
