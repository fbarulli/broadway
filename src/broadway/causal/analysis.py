"""T-test, chi-square, difference-in-differences for experimental data."""

from __future__ import annotations

import numpy as np
from scipy import stats

from broadway.causal.contracts import ExperimentResult
from broadway.stats.effect_size import cohens_d


def _welch_df(na: int, nb: int, va: float, vb: float) -> float:
    if na < 2 or nb < 2 or (va == 0.0 and vb == 0.0):
        return float(na + nb - 2)
    num = (va / na + vb / nb) ** 2
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    if den == 0.0:
        return float(na + nb - 2)
    return num / den


def _mean_diff_ci(a: np.ndarray, b: np.ndarray, alpha: float) -> tuple[float, float]:
    na, nb = a.size, b.size
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    df = _welch_df(na, nb, va, vb)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    diff = a.mean() - b.mean()
    return float(diff - t_crit * se), float(diff + t_crit * se)


def analyze_two_groups(
    treated: np.ndarray,
    control: np.ndarray,
    alpha: float,
    small_group_threshold: int,
) -> ExperimentResult:
    treated = np.asarray(treated, dtype=float)
    control = np.asarray(control, dtype=float)

    statistic, p_value = stats.ttest_ind(treated, control, equal_var=False)
    d = cohens_d(treated, control)
    ci_lower, ci_upper = _mean_diff_ci(treated, control, alpha)

    passed = bool(p_value < alpha)
    reason = [
        f"Welch's two-sample t-test (treated n={treated.size}, control n={control.size})",
        f"t={statistic:.3f}, p={p_value:.4e}, Cohen's d={d:.3f}",
        "reject H0: treatment effect detected"
        if passed
        else "fail to reject H0: no detectable effect",
    ]
    warnings = ["does not assume equal variance"]
    if treated.size < small_group_threshold or control.size < small_group_threshold:
        warnings.append("small sample size")

    return ExperimentResult(
        test_name="Welch's t-test",
        statistic=float(statistic),
        p_value=float(p_value),
        effect_size=d,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        passed=passed,
        reason=reason,
        warnings=warnings,
    )
