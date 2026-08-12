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


def bp_test(resid: np.ndarray, exog: np.ndarray) -> tuple[float, float]:
    statistic, p_value, _, _ = het_breuschpagan(resid, exog)
    return float(statistic), float(p_value)


def jb_test(resid: np.ndarray) -> tuple[float, float, float, float]:
    statistic, p_value, skew, kurtosis = jarque_bera(resid)
    return float(statistic), float(p_value), float(skew), float(kurtosis)


def durbin_watson(resid: np.ndarray) -> float:
    return float(_durbin_watson(resid))


def plot_residuals(model, out_path: str) -> None:
    fitted = model.fittedvalues
    resid = model.resid

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(fitted, resid, s=2, alpha=0.15)
    axes[0].axhline(0, color="red", linewidth=1)
    axes[0].set_xlabel("Fitted values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Fitted")

    sm.qqplot(resid, line="45", ax=axes[1], markersize=2, alpha=0.15)
    axes[1].set_title("Q-Q Plot of Residuals")

    axes[2].hist(resid, bins=100)
    axes[2].set_title("Residual Distribution")
    axes[2].set_xlabel("Residual")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
