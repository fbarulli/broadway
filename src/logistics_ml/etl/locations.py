import pandas as pd

from logistics_ml.config import data as data_config
from logistics_ml.db import engine


def load_locations():
    df = pd.read_csv(data_config.taxi_lookup)

    df.columns = [
        "location_id",
        "borough",
        "zone",
        "service_zone",
    ]

    df.to_sql(
        "locations",
        engine,
        if_exists="replace",
        index=False,
    )

    print(f"Loaded {len(df):,} locations")


def main():
    load_locations()


if __name__ == "__main__":
    main()
