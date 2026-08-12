"""
Step 7: Games-Howell post-hoc test - which specific boroughs differ?

ANOVA/Welch/Kruskal only tell us "at least one group differs." Games-Howell
does all pairwise comparisons (Manhattan vs Brooklyn, Manhattan vs Queens,
etc.) while correcting for multiple comparisons AND not assuming equal
variance or equal sample sizes - which fits our data (Levene's test failed,
group sizes range from 84 to 155,502).

Run with: python learning/stats/07_games_howell.py
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pandas as pd
import pingouin as pg

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
MIN_ROWS_FOR_SAMPLING = 10_000
SAMPLE_FRACTION = 0.02

rows = []
for borough in real_boroughs:
    borough_df = trips_with_borough.filter(F.col("pickup_borough") == borough)
    total = borough_df.count()
    if total > MIN_ROWS_FOR_SAMPLING:
        sampled = borough_df.select("trip_duration_minutes").sample(fraction=SAMPLE_FRACTION, seed=42).collect()
    else:
        sampled = borough_df.select("trip_duration_minutes").collect()
    for r in sampled:
        rows.append({"borough": borough, "duration": r["trip_duration_minutes"]})

df = pd.DataFrame(rows)
print(f"Total rows across all groups: {len(df)}")
print(df.groupby("borough")["duration"].agg(["count", "mean", "std"]))
print()

# pingouin's Games-Howell: pairwise comparisons, Welch-Satterthwaite df,
# studentized range distribution, no equal-variance/equal-n assumption
result = pg.pairwise_gameshowell(data=df, dv="duration", between="borough")
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", None)
print("=== Games-Howell pairwise comparisons ===")
print(result.round(4))

print("\n=== Plain-English summary ===")
for _, row in result.iterrows():
    sig = "DIFFERENT" if row["pval"] < 0.05 else "not significantly different"
    print(f"{row['A']} vs {row['B']}: mean diff={row['diff']:.2f} min, p={row['pval']:.4f} -> {sig}")

spark.stop()
