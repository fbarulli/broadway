from pathlib import Path

import numpy as np
import pandas as pd


def load_zones(path: str | Path) -> pd.DataFrame:
    """
    Load the NYC taxi zone lookup table.
    """
    zones = pd.read_csv(path)

    return zones[["LocationID", "Borough"]]


def add_borough_features(
    df: pd.DataFrame,
    zones: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add pickup/dropoff borough information.
    """

    df = df.copy()

    df = df.merge(
        zones.rename(
            columns={
                "LocationID": "pickup_location_id",
                "Borough": "pickup_borough",
            }
        ),
        on="pickup_location_id",
        how="left",
    )

    df = df.merge(
        zones.rename(
            columns={
                "LocationID": "dropoff_location_id",
                "Borough": "dropoff_borough",
            }
        ),
        on="dropoff_location_id",
        how="left",
    )

    df["borough_pair"] = (
        df["pickup_borough"].fillna("Unknown")
        + "_"
        + df["dropoff_borough"].fillna("Unknown")
    )

    df["same_borough"] = (
        df["pickup_borough"] == df["dropoff_borough"]
    ).astype(np.int8)

    return df
