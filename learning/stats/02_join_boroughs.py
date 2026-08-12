"""
Step 2: Join taxi_zone_lookup.csv to get pickup borough, then see
group sizes and mean trip duration per borough.
Run with: python learning/stats/02_join_boroughs.py
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

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

print("=== Zone lookup schema ===")
zones.printSchema()
zones.show(5, truncate=False)

# Join trips to zones on pickup_location_id -> LocationID
# (adjust "LocationID" below if your CSV uses a different column name,
#  check the printSchema output above first)
trips_with_borough = trips.join(
    zones.select(
        F.col("LocationID").alias("pickup_location_id"),
        F.col("Borough").alias("pickup_borough"),
    ),
    on="pickup_location_id",
    how="left",
)

print("=== Trip counts per borough ===")
trips_with_borough.groupBy("pickup_borough").count().orderBy(F.desc("count")).show()

print("=== Mean trip duration per borough ===")
trips_with_borough.groupBy("pickup_borough").agg(
    F.count("*").alias("n_trips"),
    F.round(F.mean("trip_duration_minutes"), 2).alias("avg_duration_min"),
    F.round(F.stddev("trip_duration_minutes"), 2).alias("stddev_duration_min"),
).orderBy(F.desc("n_trips")).show()

spark.stop()
