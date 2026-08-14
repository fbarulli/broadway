"""One-way ANOVA, Welch's ANOVA, Kruskal-Wallis on any group column."""

from __future__ import annotations

import numpy as np
from scipy import stats

from broadway.stats.effect_size import epsilon_squared, eta_squared, group_imbalance, omega_squared
from broadway.stats.guards import validate_groups
from broadway.stats.plan import AnalysisPlan


def _group_sizes(groups: dict[str, np.ndarray]) -> dict[str, int]:
    return {name: int(np.asarray(vals).size) for name, vals in groups.items()}


def _welch_anova(groups: dict[str, np.ndarray]) -> tuple[float, float, float, float]:
    k = len(groups)
    ns = np.array([np.asarray(v).size for v in groups.values()], dtype=float)
    means = np.array([np.asarray(v).mean() for v in groups.values()])
    variances = np.array([np.asarray(v).var(ddof=1) for v in groups.values()])

    weights = ns / variances
    grand_mean = np.sum(weights * means) / np.sum(weights)

    numerator = np.sum(weights * (means - grand_mean) ** 2) / (k - 1)

    denom_term = np.sum((1 - weights / np.sum(weights)) ** 2 / (ns - 1))
    denominator = 1 + (2 * (k - 2) / (k**2 - 1)) * denom_term

    f_stat = numerator / denominator

    df1 = float(k - 1)
    df2 = (k**2 - 1) / (3 * denom_term)

    p_val = stats.f.sf(f_stat, df1, df2)
    return float(f_stat), float(p_val), df1, float(df2)


def _build_plan(
    test_name: str,
    script: str,
    statistic: float,
    p_value: float,
    df1: float,
    df2: float,
    sizes: dict[str, int],
    alpha: float,
    reason: list[str],
    warnings: list[str],
    effect_sizes: dict[str, float],
    small_group_threshold: int,
) -> AnalysisPlan:
    imbalance_ratio = group_imbalance(sizes)
    any_small_group = any(s < small_group_threshold for s in sizes.values())
    passed = bool(p_value < alpha)
    return AnalysisPlan(
        script=script,
        analysis_type="group_comparison",
        test_name=test_name,
        statistics={"statistic": statistic, "p_value": p_value},
        effect_sizes=effect_sizes,
        threshold_context={
            "imbalance_ratio": imbalance_ratio,
            "any_small_group": any_small_group,
        },
        reason=reason,
        warnings=warnings,
        passed=passed,
        next_step="posthoc" if passed else None,
    )


def run_anova(
    groups: dict[str, np.ndarray], alpha: float = 0.05, small_group_threshold: int = 30
) -> AnalysisPlan:
    warnings = validate_groups(groups)
    sizes = _group_sizes(groups)
    n_total = sum(sizes.values())
    k = len(groups)

    f_stat, p_value = stats.f_oneway(*groups.values())
    df1 = k - 1
    df2 = n_total - k

    passed = bool(p_value < alpha)
    reason = [
        f"one-way ANOVA across {k} groups (N={n_total})",
        f"F({df1}, {df2})={f_stat:.3f}, p={p_value:.4e}",
        "reject H0: at least one group mean differs" if passed else "fail to reject H0: no mean difference",
    ]
    if any(s < small_group_threshold for s in sizes.values()):
        warnings.append("underpowered: small group(s)")
    if group_imbalance(sizes) > 1.5:
        warnings.append("n imbalance")

    return _build_plan(
        test_name="one-way ANOVA",
        script="anova",
        statistic=float(f_stat),
        p_value=float(p_value),
        df1=df1,
        df2=df2,
        sizes=sizes,
        alpha=alpha,
        reason=reason,
        warnings=warnings,
        effect_sizes={
            "eta_squared": eta_squared(f_stat, int(df1), int(df2)),
            "omega_squared": omega_squared(f_stat, int(df1), int(df2), n_total),
        },
        small_group_threshold=small_group_threshold,
    )


def run_welch(
    groups: dict[str, np.ndarray], alpha: float = 0.05, small_group_threshold: int = 30
) -> AnalysisPlan:
    warnings = validate_groups(groups)
    if any("zero variance" in w for w in warnings):
        raise ValueError("Welch's ANOVA requires non-zero within-group variance in every group")
    warnings.append("does not assume equal variance")
    sizes = _group_sizes(groups)
    n_total = sum(sizes.values())
    k = len(groups)

    f_stat, p_value, df1, df2 = _welch_anova(groups)

    passed = bool(p_value < alpha)
    reason = [
        f"Welch's ANOVA across {k} groups (N={n_total})",
        f"F({df1:.0f}, {df2:.2f})={f_stat:.3f}, p={p_value:.4e}",
        "reject H0: at least one group mean differs" if passed else "fail to reject H0: no mean difference",
    ]
    if any(s < small_group_threshold for s in sizes.values()):
        warnings.append("underpowered: small group(s)")
    if group_imbalance(sizes) > 1.5:
        warnings.append("n imbalance")

    return _build_plan(
        test_name="Welch's ANOVA",
        script="welch",
        statistic=f_stat,
        p_value=p_value,
        df1=df1,
        df2=df2,
        sizes=sizes,
        alpha=alpha,
        reason=reason,
        warnings=warnings,
        effect_sizes={
            "eta_squared": eta_squared(f_stat, int(df1), int(df2)),
            "omega_squared": omega_squared(f_stat, int(df1), int(df2), n_total),
        },
        small_group_threshold=small_group_threshold,
    )


def run_kruskal(
    groups: dict[str, np.ndarray], alpha: float = 0.05, small_group_threshold: int = 30
) -> AnalysisPlan:
    warnings = validate_groups(groups)
    warnings.append("non-parametric; no normality/variance assumptions")
    sizes = _group_sizes(groups)
    n_total = sum(sizes.values())
    k = len(groups)

    h_stat, p_value = stats.kruskal(*groups.values())
    df1 = k - 1
    df2 = n_total - 1

    passed = bool(p_value < alpha)
    reason = [
        f"Kruskal-Wallis across {k} groups (N={n_total})",
        f"H({df1})={h_stat:.3f}, p={p_value:.4e}",
        "reject H0: group distributions differ" if passed else "fail to reject H0: distributions equal",
    ]
    if any(s < small_group_threshold for s in sizes.values()):
        warnings.append("underpowered: small group(s)")

    return _build_plan(
        test_name="Kruskal-Wallis",
        script="kruskal",
        statistic=float(h_stat),
        p_value=float(p_value),
        df1=df1,
        df2=df2,
        sizes=sizes,
        alpha=alpha,
        reason=reason,
        warnings=warnings,
        effect_sizes={"epsilon_squared": epsilon_squared(h_stat, k, n_total)},
        small_group_threshold=small_group_threshold,
    )
