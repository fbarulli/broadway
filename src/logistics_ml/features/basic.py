import numpy as np
import pandas as pd


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add deterministic time and trip features.

    These features can be computed independently for both train and inference.
    """

    df = df.copy()

    dt = df["pickup_datetime"]

    df["pickup_hour"] = dt.dt.hour
    df["pickup_day_of_week"] = dt.dt.dayofweek
    df["pickup_month"] = dt.dt.month

    df["is_weekend"] = (df["pickup_day_of_week"] >= 5).astype(np.int8)

    df["rush_hour"] = (
        (
            (df["pickup_hour"] >= 7)
            & (df["pickup_hour"] <= 9)
        )
        |
        (
            (df["pickup_hour"] >= 16)
            & (df["pickup_hour"] <= 19)
        )
    ).astype(np.int8)

    df["is_night"] = (
        (df["pickup_hour"] >= 22)
        |
        (df["pickup_hour"] <= 5)
    ).astype(np.int8)

    df["passenger_count"] = df["passenger_count"].clip(1, 6)

    df["log_distance"] = np.log1p(df["trip_distance"])

    return df
