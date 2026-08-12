"""
Step 4: ANOVA - does trip duration differ across boroughs?

A t-test compares 2 groups. ANOVA compares 3+ groups at once, asking
one question: "is at least one group's mean different from the others?"
It does NOT tell you which one(s) - that needs a post-hoc test (next step).

Run with: python learning/stats/04_anova_boroughs.py
"""
from scipy import stats

from _config import get_spark_session, load_borough_durations, BOROUGHS

spark = get_spark_session()

groups = load_borough_durations(spark)

for borough, values in groups.items():
    print(f"{borough}: sampled_n={len(values)}, mean={values.mean():.2f}")

print()

f_stat, p_value = stats.f_oneway(*groups.values())

print("=== One-way ANOVA ===")
print(f"F-statistic: {f_stat:.2f}")
print(f"p-value:     {p_value:.6e}")

if p_value < 0.05:
    print("-> p < 0.05: reject the null hypothesis.")
    print("   At least one borough's average trip duration is significantly")
    print("   different from the others. ANOVA doesn't say which one(s) -")
    print("   that needs a post-hoc test (e.g. Tukey's HSD) as a follow-up.")
else:
    print("-> p >= 0.05: no significant difference detected across boroughs.")

spark.stop()
