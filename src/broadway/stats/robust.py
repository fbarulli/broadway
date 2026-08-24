"""Robust regression helpers: outlier screening, winsorizing, and coefficient tables."""

from __future__ import annotations

import pandas as pd


def modified_zscore(series: pd.Series) -> pd.Series:
    """Modified z-score: 0.6745 * (x - median) / MAD; zeros when MAD == 0."""
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        return pd.Series(0.0, index=series.index)
    return 0.6745 * (series - median) / mad


def outlier_mask(df: pd.DataFrame, columns: list[str], threshold: float) -> pd.Series:
    """Boolean OR of |modified z-score| > threshold across the given columns."""
    mask = pd.Series(False, index=df.index)
    for col in columns:
        mask |= modified_zscore(df[col]).abs() > threshold
    return mask


def winsorize(df: pd.DataFrame, columns: list[str], cap_quantile: float) -> pd.DataFrame:
    """Return a copy with each column clipped at its cap_quantile upper quantile."""
    out = df.copy()
    for col in columns:
        cap = out[col].quantile(cap_quantile)
        out[col] = out[col].clip(upper=cap)
    return out


def estimation_table(model, alpha: float = 0.05) -> pd.DataFrame:
    """Coefficient table with HC3 robust SEs and confidence interval bounds.

    The HC3_SE / CI_low / CI_high columns are derived from an HC3-robust
    covariance: this function internally re-fits the covariance of the passed
    results object with ``get_robustcov_results(cov_type="HC3")`` and reads
    ``bse`` and ``conf_int(alpha=alpha)`` from that object. The HC3 labels are
    therefore truthful by construction, regardless of how the input model was
    fitted (e.g. a plain non-robust OLS fit still yields true HC3 columns).

    The input must be a fitted statsmodels regression results object exposing
    ``get_robustcov_results``; anything else raises TypeError.
    """
    if not hasattr(model, "get_robustcov_results"):
        raise TypeError(
            "estimation_table requires a fitted statsmodels regression results "
            "object exposing get_robustcov_results (needed to derive the HC3 "
            "standard errors and confidence intervals); "
            f"got {type(model).__name__!r}"
        )
    robust = model.get_robustcov_results("HC3")
    names = model.model.exog_names
    ci = pd.DataFrame(robust.conf_int(alpha=alpha), index=names)
    return pd.DataFrame(
        {
            "coef": pd.Series(model.params, index=names),
            "HC3_SE": pd.Series(robust.bse, index=names),
            "CI_low": ci.iloc[:, 0],
            "CI_high": ci.iloc[:, 1],
        }
    )


def standardized_coefs(
    model, df: pd.DataFrame, predictors: list[str], target: str
) -> dict:
    """Per-predictor beta_std = coef * sd_x / sd_y with the raw inputs."""
    sd_y = float(df[target].std())
    out = {}
    for predictor in predictors:
        coef = float(model.params[predictor])
        sd_x = float(df[predictor].std())
        out[predictor] = {
            "coef": coef,
            "sd_x": sd_x,
            "sd_y": sd_y,
            "beta_std": coef * sd_x / sd_y,
        }
    return out


def scenario_dollars(model, scenarios: list[tuple[str, str, float]]) -> list[dict]:
    """Dollar impact of each (label, term, change) scenario: change * coef."""
    rows = []
    for label, term, change in scenarios:
        rows.append(
            {
                "label": label,
                "term": term,
                "change": change,
                "dollars": change * float(model.params[term]),
            }
        )
    return rows
