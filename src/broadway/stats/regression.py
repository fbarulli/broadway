"""OLS fitting with robust covariance and residual-based diagnostics."""

from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf

from broadway.stats.diagnostics import bp_test, jb_test


def fit_ols(df: pd.DataFrame, formula: str) -> object:
    return smf.ols(formula, data=df).fit()


def fit_robust(model, cov_type: str = "HC3") -> object:
    return model.get_robustcov_results(cov_type)


def bp_jb(model) -> dict:
    resid = model.resid
    exog = model.model.exog

    bp_stat, bp_pval = bp_test(resid, exog)
    jb_stat, jb_pval, skew, kurtosis = jb_test(resid)

    return {
        "bp_stat": bp_stat,
        "bp_pval": bp_pval,
        "jb_stat": jb_stat,
        "jb_pval": jb_pval,
        "skew": skew,
        "kurtosis": kurtosis,
    }
