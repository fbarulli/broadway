"""
Step 4: ANOVA - does trip duration differ across boroughs?

A t-test compares 2 groups. ANOVA compares 3+ groups at once, asking
one question: "is at least one group's mean different from the others?"
It does NOT tell you which one(s) - that needs a post-hoc test (next step).

Run with: python learning/stats/04_anova_boroughs.py
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from scipy import stats

spark = (
    SparkSession.builder
    .appName("stats-learning")
    .master("local[*]")
    .getOrCreate()
)

trips = spark.read.parquet("data/processed/training_data.parquet")
zones = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv("data/raw/taxi_zone_lookup.csv")
)

trips_with_borough = trips.join(
    zones.select(
        F.col("LocationID").alias("pickup_location_id"),
        F.col("Borough").alias("pickup_borough"),
    ),
    on="pickup_location_id",
    how="left",
)

real_boroughs = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]

# Small groups (e.g. Staten Island: 84 rows total) get taken in FULL.
# Large groups get sampled for speed - still thousands of rows either way.
MIN_ROWS_FOR_SAMPLING = 10_000
SAMPLE_FRACTION = 0.02

groups = {}
for borough in real_boroughs:
    borough_df = trips_with_borough.filter(F.col("pickup_borough") == borough)
    total = borough_df.count()

    if total > MIN_ROWS_FOR_SAMPLING:
        rows = borough_df.select("trip_duration_minutes").sample(fraction=SAMPLE_FRACTION, seed=42).collect()
    else:
        rows = borough_df.select("trip_duration_minutes").collect()

    values = [r["trip_duration_minutes"] for r in rows]
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
