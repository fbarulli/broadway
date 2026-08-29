"""Shared analysis/plot helpers for this experiment (reused by 04, 13, 18-27)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from _common import load_working, time_bucket
from matplotlib.lines import Line2D
from scipy import stats
from statsmodels.regression.linear_model import RegressionResultsWrapper

from broadway.stats import robust as _robust
from broadway.stats.regression import bp_jb, fit_ols

ALPHA = 0.05


def modified_zscore(series: pd.Series) -> pd.Series:
    """Robust standardized score from median and MAD: 0.6745 * (x - median) / MAD."""
    return _robust.modified_zscore(series)


def _fmt_p(p: float) -> str:
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def attach_stats_legend(
    fig,
    result: dict,
    raw_result: dict | None = None,
    target: dict | None = None,
) -> None:
    """Bottom legend with BP/JB/skew/kurtosis, outside the axes.

    Mirrors src/broadway/discover/qq.py::attach_qq_legend (figure-level,
    horizontal columns, framed), but anchored in the bottom band reserved by
    subplots_adjust(bottom=...) — the legend never overlaps the graphs.

    When raw_result (bp_jb of the non-log model) and target (raw/log target
    distribution stats) are given, the legend shows the raw → log comparison,
    like the platform's raw-vs-log Q-Q figure.
    """
    if raw_result is None or target is None:
        handles = [
            Line2D([0], [0], color="none", label=f"Breusch-Pagan: stat={result['bp_stat']:.2f}, {_fmt_p(result['bp_pval'])}"),
            Line2D([0], [0], color="none", label=f"Jarque-Bera: stat={result['jb_stat']:.2f}, {_fmt_p(result['jb_pval'])}"),
            Line2D([0], [0], color="none", label=f"skew={result['skew']:.2f}, kurtosis={result['kurtosis']:.2f}"),
        ]
        ncol = len(handles)
    else:
        handles = [
            Line2D([0], [0], color="none", label=f"BP stat: {raw_result['bp_stat']:.2f} → {result['bp_stat']:.2f} ({_fmt_p(result['bp_pval'])})"),
            Line2D([0], [0], color="none", label=f"JB stat: {raw_result['jb_stat']:.2e} → {result['jb_stat']:.2e} ({_fmt_p(result['jb_pval'])})"),
            Line2D([0], [0], color="none", label=f"resid skew/kurt: {raw_result['skew']:.2f}/{raw_result['kurtosis']:.1f} → {result['skew']:.2f}/{result['kurtosis']:.1f}"),
            Line2D([0], [0], color="none", label=f"target skew/kurt: {target['skew_raw']:.2f}/{target['kurt_raw']:.1f} → {target['skew_log']:.2f}/{target['kurt_log']:.1f}"),
        ]
        ncol = 2
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07),
        ncol=ncol,
        fontsize=9,
        frameon=True,
        framealpha=0.85,
    )


def fit_log_hc3(df: pd.DataFrame) -> RegressionResultsWrapper:
    """OLS on log1p(fare_amount) ~ trip_distance + duration_minutes, HC3 SEs."""
    y = np.log1p(df["fare_amount"])
    X = sm.add_constant(df[["trip_distance", "duration_minutes"]])
    return sm.OLS(y, X).fit(cov_type="HC3")


CAP_QUANTILE = 0.995  # winsorization cap (steps 26/27)
CAPPED_COLUMNS = ("fare_amount", "trip_distance")


def winsorize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Cap the capped columns at their 99.5th percentiles; return df + caps."""
    clipped = _robust.winsorize(df, list(CAPPED_COLUMNS), CAP_QUANTILE)
    caps = {c: float(df[c].quantile(CAP_QUANTILE)) for c in CAPPED_COLUMNS}
    n_capped = {c: int((df[c] > caps[c]).sum()) for c in CAPPED_COLUMNS}
    return clipped, {"caps": caps, "n_capped": n_capped}


OUTLIER_Z_THRESHOLD = 10.0  # step-22 modified-z |M| threshold (union over fare/distance)


def outlier_mask() -> pd.Series:
    """Step-22 mask: |modified z| > 10 on trip_distance or fare_amount."""
    return _robust.outlier_mask(load_working(), list(CAPPED_COLUMNS), OUTLIER_Z_THRESHOLD)


def fit_raw_hc3(df: pd.DataFrame, use_hour: bool) -> RegressionResultsWrapper:
    """HC3 fit of raw fare; optional pickup_hour feature (steps 28/29)."""
    cols = ["trip_distance", "duration_minutes"] + (["pickup_hour"] if use_hour else [])
    X = sm.add_constant(df[cols])
    return sm.OLS(df["fare_amount"], X).fit(cov_type="HC3")


# (label, predictor, realistic change in predictor units) — estimate-size steps 29/30
SCENARIOS: tuple[tuple[str, str, float], ...] = (
    ("5-mile trip", "trip_distance", 5.0),
    ("10 min waiting", "duration_minutes", 10.0),
    ("typical 1.6mi trip", "trip_distance", 1.6),
)


def estimation_table(model: RegressionResultsWrapper,
                     alpha: float = ALPHA) -> pd.DataFrame:
    """coef / HC3 SE / 95% CI table (p-values deliberately absent)."""
    return _robust.estimation_table(model, alpha)


def scenario_dollars(model: RegressionResultsWrapper, df: pd.DataFrame,
                     scenarios: tuple[tuple[str, str, float], ...] = SCENARIOS
                     ) -> list[dict]:
    """Dollar effect of each realistic scenario, from the fitted coefs."""
    # `df` is retained for backward compatibility with earlier callers.
    return _robust.scenario_dollars(model, scenarios)


def standardized_coefs(model: RegressionResultsWrapper, df: pd.DataFrame) -> dict:
    """beta_std = coef * sd_x / sd_y for each predictor."""
    return _robust.standardized_coefs(
        model, df, ["trip_distance", "duration_minutes"], "fare_amount"
    )


def fit_time_bucket(df: pd.DataFrame, target: str = "fare_amount") -> RegressionResultsWrapper:
    """HC3 fit of `target` ~ distance + duration + peak/overnight dummies (day ref)."""
    buckets = df["pickup_hour"].map(time_bucket)
    X = df[["trip_distance", "duration_minutes"]].copy()
    dummies = pd.get_dummies(buckets, prefix="time")
    # get_dummies yields bool columns in pandas 2.x; statsmodels needs numeric
    X = pd.concat([X, dummies[["time_peak", "time_overnight"]]], axis=1).astype(float)
    X = sm.add_constant(X)
    return sm.OLS(df[target], X).fit(cov_type="HC3")


def add_log_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with predicted_log_fare and log_residuals columns."""
    model = fit_log_hc3(df)
    out = df.copy()
    out["predicted_log_fare"] = model.fittedvalues
    out["log_residuals"] = model.resid
    return out


def plot_log_resid_qq(
    model: RegressionResultsWrapper,
    out_path: Path,
    suptitle: str | None = None,
    raw_result: dict | None = None,
    target: dict | None = None,
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
    attach_stats_legend(fig, bp_jb(model), raw_result=raw_result, target=target)
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
    """Standalone residuals-vs-fitted scatter with bottom stats legend."""
    fig, ax = plt.subplots(figsize=(10, 7.5))
    fig.subplots_adjust(bottom=0.30, top=0.90)
    draw_resid_vs_fitted(ax, model)
    attach_stats_legend(fig, bp_jb(model))
    n = int(model.nobs)
    fig.suptitle(f"{suptitle} (N={n})" if suptitle else f"N={n}")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_resid_vs_fitted_qq(
    model: RegressionResultsWrapper,
    out_path: Path,
    suptitle: str | None = None,
) -> None:
    """Residuals-vs-fitted and residual Q-Q, side by side, with bottom stats legend."""
    fig, (ax_resid, ax_qq) = plt.subplots(1, 2, figsize=(14, 7.5))
    fig.subplots_adjust(bottom=0.30, top=0.90)
    draw_resid_vs_fitted(ax_resid, model)
    stats.probplot(model.resid, dist="norm", plot=ax_qq)
    ax_qq.set_title("Q-Q plot (residuals)")
    ax_qq.grid(True, alpha=0.3)
    attach_stats_legend(fig, bp_jb(model))
    n = int(model.nobs)
    fig.suptitle(f"{suptitle} (N={n})" if suptitle else f"N={n}")
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

    fig, (ax_resid, ax_bp) = plt.subplots(1, 2, figsize=(14, 7.5))
    fig.subplots_adjust(bottom=0.30, top=0.90)

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
        arrowprops={"arrowstyle": "->", "color": "black"},
        fontsize=9,
    )
    ax_bp.set_xlabel("z-score")
    ax_bp.set_ylabel("density (standard normal)")
    ax_bp.set_title(f"p = {p_value:.4f} → {verdict}")
    ax_bp.grid(True, alpha=0.3)

    attach_stats_legend(fig, result)

    if suptitle is not None:
        fig.suptitle(f"{suptitle} (N={len(df)})")
    else:
        fig.suptitle(f"N={len(df)}")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return result
