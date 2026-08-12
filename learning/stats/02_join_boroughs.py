"""
Step 2: Join taxi_zone_lookup.csv to get pickup borough, then see
group sizes and mean trip duration per borough.
Run with: python learning/stats/02_join_boroughs.py
"""
from pyspark.sql import functions as F

from _config import LOOKUP_PATH, PICKUP_BOROUGH_COL, DURATION_COL, get_spark_session, _load_boroughs

spark = get_spark_session()

print("=== Zone lookup schema ===")
zones = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(LOOKUP_PATH)
)
zones.printSchema()
zones.show(5, truncate=False)

trips_with_borough = _load_boroughs(spark)

print("=== Trip counts per borough ===")
trips_with_borough.groupBy(PICKUP_BOROUGH_COL).count().orderBy(F.desc("count")).show()

print("=== Mean trip duration per borough ===")
trips_with_borough.groupBy(PICKUP_BOROUGH_COL).agg(
    F.count("*").alias("n_trips"),
    F.round(F.mean(DURATION_COL), 2).alias("avg_duration_min"),
    F.round(F.stddev(DURATION_COL), 2).alias("stddev_duration_min"),
).orderBy(F.desc("n_trips")).show()

spark.stop()
