"""
Step 6: Compare four ways of testing "do boroughs differ in trip duration",
given that we found unequal variance AND non-normal data.

1. Standard (Fisher's) ANOVA        - what we ran originally, assumptions violated
2. Log-transformed + standard ANOVA - fixes skew, may fix variance too
3. Welch's ANOVA                    - doesn't assume equal variance
4. Kruskal-Wallis                   - doesn't assume normality OR equal variance

Run with: python learning/stats/06_anova_comparison.py
"""
from pyspark.sql import functions as F
from scipy import stats
import numpy as np

from _config import (
    BOROUGHS, MIN_ROWS_FOR_SAMPLING, RANDOM_STATE, SAMPLE_FRACTION,
    PICKUP_BOROUGH_COL, DURATION_COL,
    get_spark_session, _load_boroughs,
)

spark = get_spark_session()

trips_with_borough = _load_boroughs(spark)

groups = {}
for borough in BOROUGHS:
    borough_df = trips_with_borough.filter(F.col(PICKUP_BOROUGH_COL) == borough)
    total = borough_df.count()
    if total > MIN_ROWS_FOR_SAMPLING:
        rows = borough_df.select(DURATION_COL).sample(fraction=SAMPLE_FRACTION, seed=RANDOM_STATE).collect()
    else:
        rows = borough_df.select(DURATION_COL).collect()
    groups[borough] = np.array([r[DURATION_COL] for r in rows])

print("Group sizes:", {k: len(v) for k, v in groups.items()})
print()

f_stat, p_val = stats.f_oneway(*groups.values())
print(f"1. Standard ANOVA:        F={f_stat:>10.2f}  p={p_val:.4e}")

log_groups = {k: np.log(v) for k, v in groups.items()}

print("\nSkew before vs after log-transform:")
for name in groups:
    print(f"  {name}: {stats.skew(groups[name]):.2f} -> {stats.skew(log_groups[name]):.2f}")

f_stat_log, p_val_log = stats.f_oneway(*log_groups.values())
print(f"\n2. Log + standard ANOVA:  F={f_stat_log:>10.2f}  p={p_val_log:.4e}")

levene_log_stat, levene_log_p = stats.levene(*log_groups.values())
print(f"   Levene's test on log-transformed data: p={levene_log_p:.4e} "
      f"({'still violated' if levene_log_p < 0.05 else 'assumption now holds'})")


def welch_anova(groups_dict):
    k = len(groups_dict)
    ns = np.array([len(v) for v in groups_dict.values()])
    means = np.array([v.mean() for v in groups_dict.values()])
    variances = np.array([v.var(ddof=1) for v in groups_dict.values()])

    weights = ns / variances
    grand_mean = np.sum(weights * means) / np.sum(weights)

    numerator = np.sum(weights * (means - grand_mean) ** 2) / (k - 1)

    denom_term = np.sum((1 - weights / np.sum(weights)) ** 2 / (ns - 1))
    denominator = 1 + (2 * (k - 2) / (k ** 2 - 1)) * denom_term

    f_stat = numerator / denominator

    df1 = k - 1
    df2 = (k ** 2 - 1) / (3 * denom_term)

    p_val = stats.f.sf(f_stat, df1, df2)
    return f_stat, p_val, df1, df2


welch_f, welch_p, df1, df2 = welch_anova(groups)
print(f"\n3. Welch's ANOVA:          F={welch_f:>10.2f}  p={welch_p:.4e}  (df1={df1}, df2={df2:.1f})")

ag_result = stats.alexandergovern(*groups.values())
print(f"   Alexander-Govern (scipy cross-check): stat={ag_result.statistic:.2f}  p={ag_result.pvalue:.4e}")

h_stat, p_val_kw = stats.kruskal(*groups.values())
print(f"\n4. Kruskal-Wallis:         H={h_stat:>10.2f}  p={p_val_kw:.4e}")

print("\n=== Summary ===")
results = {
    "Standard ANOVA": p_val,
    "Log + ANOVA": p_val_log,
    "Welch's ANOVA": welch_p,
    "Kruskal-Wallis": p_val_kw,
}
for name, p in results.items():
    verdict = "significant" if p < 0.05 else "NOT significant"
    print(f"  {name:20s} p={p:.4e}  -> {verdict}")

spark.stop()
