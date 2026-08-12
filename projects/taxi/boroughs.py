from pathlib import Path

import numpy as np
import pandas as pd


def load_zones(path: str | Path) -> pd.DataFrame:
    zones = pd.read_csv(path)
    return zones[["LocationID", "Borough"]]


def _merge_borough(df: pd.DataFrame, zones: pd.DataFrame, side: str) -> pd.DataFrame:
    loc_col = f"{side}_location_id"
    boro_col = f"{side}_borough"
    return df.merge(
        zones.rename(columns={"LocationID": loc_col, "Borough": boro_col}),
        on=loc_col,
        how="left",
    )


def add_borough_features(df: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _merge_borough(df, zones, "pickup")
    df = _merge_borough(df, zones, "dropoff")
    df["borough_pair"] = (
        df["pickup_borough"].fillna("Unknown")
        + "_"
        + df["dropoff_borough"].fillna("Unknown")
    )
    df["same_borough"] = (
        df["pickup_borough"] == df["dropoff_borough"]
    ).astype(np.int8)
    return df
