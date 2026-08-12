"""
10_durbin_watson_time_ordered.py

Caveat on the 08 result: DW=1.994 was computed on a stratified RANDOM sample
of rows -- row order had nothing to do with pickup_datetime, so that number
only proved "shuffled rows aren't autocorrelated with each other," which is
true by construction and says nothing about real temporal autocorrelation
(rush hour clustering, weather events, etc). This script re-runs it properly.

Two things differ from 08's sampling approach:
  1. We sort by pickup_datetime before fitting, so residual order reflects
     actual chronological order.
  2. We take a CONTIGUOUS time slice (e.g. one month) rather than a random
     sample, since Durbin-Watson tests adjacency -- a random sample from a
     giant date range would put temporally distant trips next to each other
     and manufacture an artificially low autocorrelation reading either way.

Also plots the ACF of residuals over a bounded lag window, which is more
informative than the single DW scalar (DW only captures lag-1 autocorrelation;
if there's an hourly/daily cycle, DW can look fine while short-range
autocorrelation is still very real).
"""

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

from _config import DATA_PATH, LOOKUP_PATH, TIME_SLICE_START, TIME_SLICE_END, ACF_LAGS


def load_time_slice() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    zones = pd.read_csv(LOOKUP_PATH)
    df = df.merge(
        zones[["LocationID", "Borough"]],
        left_on="pickup_location_id",
        right_on="LocationID",
        how="left",
    ).rename(columns={"Borough": "pickup_borough"})

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    mask = (df["pickup_datetime"] >= TIME_SLICE_START) & (
        df["pickup_datetime"] < TIME_SLICE_END
    )
    df = df.loc[mask].sort_values("pickup_datetime").reset_index(drop=True)
    return df


def main():
    print(f"Loading contiguous slice: {TIME_SLICE_START} to {TIME_SLICE_END}")
    df = load_time_slice()
    print(f"Rows in slice: {len(df)}\n")

    if len(df) == 0:
        print("No rows in this date range -- adjust TIME_SLICE_START/END "
              "to match the actual data coverage and rerun.")
        return

    model = smf.ols(
        "trip_duration_minutes ~ trip_distance + C(pickup_borough)", data=df
    ).fit()

    dw = durbin_watson(model.resid)
    print(f"Durbin-Watson (time-ordered): {dw:.3f}")
    print(
        "  -> ~2.0 = no lag-1 autocorrelation, <1.5 = notable positive "
        "autocorrelation, >2.5 = notable negative autocorrelation.\n"
        "  Compare this to 08's DW=1.994, which was computed on shuffled "
        "rows and is not a valid comparison point -- this is the number "
        "that actually means something."
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(model.resid, lags=ACF_LAGS, ax=ax)
    ax.set_title("ACF of residuals, time-ordered (bounded lag window)")
    plt.tight_layout()
    plt.savefig("residual_acf_time_ordered.png", dpi=150)
    print("\nSaved ACF plot to residual_acf_time_ordered.png")
    print(
        "Look for: a slow-decaying ACF (trend), or spikes at lag~24 "
        "(daily cycle) -- either would mean pickup_datetime carries signal "
        "this model isn't using yet, motivating the timeseries features "
        "flagged in the original handoff (hour/day/season, ADF test)."
    )


if __name__ == "__main__":
    main()