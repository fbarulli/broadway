"""Spark session lifecycle and stratified sampling."""

from __future__ import annotations

import pandas as pd
from pyspark.sql import SparkSession


def get_spark_session(app_name: str = "stats-learning") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def stratified_sample(
    df: pd.DataFrame, group_col: str, frac: float, random_state: int
) -> pd.DataFrame:
    return (
        df.groupby(group_col)
        .sample(frac=frac, random_state=random_state)
        .reset_index(drop=True)
    )
