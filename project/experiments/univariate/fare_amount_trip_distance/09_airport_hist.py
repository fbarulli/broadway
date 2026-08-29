"""09: histogram of fare_amount for airport pickups/dropoffs."""

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

    airport_trips = df[
        df["pickup_location_id"].isin(airports) | df["dropoff_location_id"].isin(airports)
    ]

    values = airport_trips["fare_amount"]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(values, bins=60, edgecolor="white", linewidth=0.5)
    ax.set_xlim(float(values.min()), float(values.max()))
    ax.set_xlabel("fare_amount ($)")
    ax.set_ylabel("count")
    ax.set_title(f"fare_amount histogram (airport pickups/dropoffs, N={len(airport_trips)})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)

    print(f"airport trips: {len(airport_trips)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
