"""08: scatter of fare_amount vs trip_distance for airport pickups/dropoffs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from _common import FULL_PARQUET, RESULTS

from project.data import LOOKUP_PATH

OUT = RESULTS / f"{Path(__file__).stem}.png"

AIRPORT_SERVICE_ZONES = ["Airports", "EWR"]


def airport_location_ids() -> set[int]:
    zones = pd.read_csv(LOOKUP_PATH)
    return set(zones.loc[zones["service_zone"].isin(AIRPORT_SERVICE_ZONES), "LocationID"])


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not FULL_PARQUET.exists():
        raise FileNotFoundError(f"{FULL_PARQUET} not found — run 06_full_dataset.py first")
    df = pd.read_parquet(FULL_PARQUET)
    airports = airport_location_ids()
    trips = df[
        df["pickup_location_id"].isin(airports) | df["dropoff_location_id"].isin(airports)
    ]

    x = trips["trip_distance"].to_numpy()
    y = trips["fare_amount"].to_numpy()
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(x, y, s=1.5, alpha=0.15, edgecolors="none")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("trip_distance (miles)")
    ax.set_ylabel("fare_amount ($)")
    ax.set_title(f"fare_amount vs trip_distance (airport pickups/dropoffs, N={len(trips)})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)

    print(f"airport trips: {len(trips)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
