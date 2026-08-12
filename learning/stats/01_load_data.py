"""
Step 1: Load training_data.parquet with Spark and get a feel for it.
Run with: python learning/stats/01_load_data.py
"""
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("stats-learning")
    .master("local[*]")          # run locally using all cores, no cluster needed
    .getOrCreate()
)

DATA_PATH = "data/processed/training_data.parquet"

df = spark.read.parquet(DATA_PATH)

print("=== Schema ===")
df.printSchema()

print("=== Row count ===")
print(df.count())

print("=== Sample rows ===")
df.show(10, truncate=False)

print("=== Column names ===")
print(df.columns)

spark.stop()
