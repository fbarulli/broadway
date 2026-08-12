"""
Step 5: Check ANOVA's assumptions before trusting the last step's result.

1. Homogeneity of variance -> Levene's test
2. Normality within each group -> skew/kurtosis + Shapiro (on a subsample)

Run with: python learning/stats/05_anova_assumptions.py
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from scipy import stats
import numpy as np

from _config import BOROUGHS, DATA_PATH, LOOKUP_PATH, MIN_ROWS_FOR_SAMPLING, SAMPLE_FRACTION

spark = (
    SparkSession.builder
    .appName("stats-learning")
    .master("local[*]")
    .getOrCreate()
)

trips = spark.read.parquet(DATA_PATH)
zones = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(LOOKUP_PATH)
)

trips_with_borough = trips.join(
    zones.select(
        F.col("LocationID").alias("pickup_location_id"),
        F.col("Borough").alias("pickup_borough"),
    ),
    on="pickup_location_id",
    how="left",
)

real_boroughs = list(BOROUGHS)

groups = {}
for borough in real_boroughs:
    borough_df = trips_with_borough.filter(F.col("pickup_borough") == borough)
    total = borough_df.count()
    if total > MIN_ROWS_FOR_SAMPLING:
        rows = borough_df.select("trip_duration_minutes").sample(fraction=SAMPLE_FRACTION, seed=42).collect()
    else:
        rows = borough_df.select("trip_duration_minutes").collect()
    groups[borough] = np.array([r["trip_duration_minutes"] for r in rows])

# --- Assumption 1: Homogeneity of variance (Levene's test) ---
# H0: all groups have equal variance. Unlike ANOVA, Levene's test is
# robust even when the data isn't normal - good, since we're about to
# check normality separately and don't want to assume it yet.
print("=== Variance per group ===")
for name, vals in groups.items():
    print(f"{name}: variance={vals.var():.2f}, std={vals.std():.2f}")

levene_stat, levene_p = stats.levene(*groups.values())
print(f"\nLevene's test: stat={levene_stat:.2f}, p={levene_p:.6e}")
if levene_p < 0.05:
    print("-> p < 0.05: variances are significantly different across groups.")
    print("   Homogeneity of variance is VIOLATED. Standard ANOVA's F-test")
    print("   is not reliable here - use Welch's ANOVA instead (next step).")
else:
    print("-> p >= 0.05: no evidence variances differ. Assumption holds.")

# --- Assumption 2: Normality within each group ---
# Shapiro-Wilk needs a small sample (it's overly sensitive / breaks down
# on large n - even trivial deviations from normal get p<0.05 past a few
# thousand rows). We check skew/kurtosis on the full group, and run
# Shapiro on a small random subsample just to see the shape.
print("\n=== Normality per group ===")
for name, vals in groups.items():
    skewness = stats.skew(vals)
    kurt = stats.kurtosis(vals)
    subsample = np.random.choice(vals, size=min(500, len(vals)), replace=False)
    shapiro_stat, shapiro_p = stats.shapiro(subsample)
    print(f"{name}: skew={skewness:.2f}, kurtosis={kurt:.2f}, "
          f"Shapiro p (n=500 subsample)={shapiro_p:.4f}")

spark.stop()
