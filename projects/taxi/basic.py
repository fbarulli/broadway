import numpy as np
import pandas as pd


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["pickup_datetime"]
    df["pickup_hour"] = dt.dt.hour
    df["pickup_day_of_week"] = dt.dt.dayofweek
    df["pickup_month"] = dt.dt.month
    df["is_weekend"] = (df["pickup_day_of_week"] >= 5).astype(np.int8)
    return df


def _add_rush_hour(
    df: pd.DataFrame,
    morning_start: int,
    morning_end: int,
    evening_start: int,
    evening_end: int,
) -> pd.DataFrame:
    df["rush_hour"] = (
        ((df["pickup_hour"] >= morning_start) & (df["pickup_hour"] <= morning_end))
        | ((df["pickup_hour"] >= evening_start) & (df["pickup_hour"] <= evening_end))
    ).astype(np.int8)
    return df


def _add_night(df: pd.DataFrame, night_start: int, night_end: int) -> pd.DataFrame:
    df["is_night"] = (
        (df["pickup_hour"] >= night_start) | (df["pickup_hour"] <= night_end)
    ).astype(np.int8)
    return df


def add_basic_features(
    df: pd.DataFrame,
    rush_hour_morning_start: int,
    rush_hour_morning_end: int,
    rush_hour_evening_start: int,
    rush_hour_evening_end: int,
    night_start: int,
    night_end: int,
    passenger_count_min: int,
    passenger_count_max: int,
) -> pd.DataFrame:
    df = df.copy()
    df = _add_time_features(df)
    df = _add_rush_hour(
        df, rush_hour_morning_start, rush_hour_morning_end,
        rush_hour_evening_start, rush_hour_evening_end,
    )
    df = _add_night(df, night_start, night_end)
    df["passenger_count"] = df["passenger_count"].clip(
        passenger_count_min, passenger_count_max
    )
    df["log_distance"] = np.log1p(df["trip_distance"])
    return df
