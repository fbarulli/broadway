"""
Step 4: ANOVA - does trip duration differ across boroughs?

A t-test compares 2 groups. ANOVA compares 3+ groups at once, asking
one question: "is at least one group's mean different from the others?"
It does NOT tell you which one(s) - that needs a post-hoc test (next step).

Run with: python learning/stats/04_anova_boroughs.py
"""
from pyspark.sql import functions as F
from scipy import stats

from _config import BOROUGHS, MIN_ROWS_FOR_SAMPLING, RANDOM_STATE, SAMPLE_FRACTION, PICKUP_BOROUGH_COL, DURATION_COL, get_spark_session, _load_boroughs

spark = get_spark_session()

trips_with_borough = _load_boroughs(spark)

real_boroughs = list(BOROUGHS)

# Small groups (e.g. Staten Island: 84 rows total) get taken in FULL.
# Large groups get sampled for speed - still thousands of rows either way.

groups = {}
for borough in real_boroughs:
    borough_df = trips_with_borough.filter(F.col(PICKUP_BOROUGH_COL) == borough)
    total = borough_df.count()

    if total > MIN_ROWS_FOR_SAMPLING:
        rows = borough_df.select(DURATION_COL).sample(fraction=SAMPLE_FRACTION, seed=RANDOM_STATE).collect()
    else:
        rows = borough_df.select(DURATION_COL).collect()

    values = [r[DURATION_COL] for r in rows]
    groups[borough] = values
    print(f"{borough}: total_rows={total}, sampled_n={len(values)}, mean={sum(values)/len(values):.2f}")

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
