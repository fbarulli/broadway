"""Durbin-Watson and ACF plot."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf as _plot_acf
from statsmodels.stats.stattools import durbin_watson


def durbin_watson_test(resid: np.ndarray) -> float:
    return float(durbin_watson(np.asarray(resid, dtype=float)))


def plot_acf(resid: np.ndarray, lags: int, out_path: str) -> None:
    fig = _plot_acf(np.asarray(resid, dtype=float), lags=lags)
    fig.savefig(out_path)
    plt.close(fig)
