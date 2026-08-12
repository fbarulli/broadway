"""
10_durbin_watson_time.py

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

from projects.taxi import data
from broadway.stats import regression, time_series


def main() -> None:
    print(f"Loading contiguous slice: {data.TIME_SLICE_START} to {data.TIME_SLICE_END}")
    df = data.load_time_slice()
    print(f"Rows in slice: {len(df)}\n")

    if len(df) == 0:
        print("No rows in this date range -- adjust TIME_SLICE_START/END "
              "to match the actual data coverage and rerun.")
        return

    formula = (
        f"{data.TARGET_COL} ~ {data.TRIP_DISTANCE_COL} + C({data.PICKUP_BOROUGH_COL})"
    )
    model = regression.fit_ols(df, formula)

    dw = time_series.durbin_watson_test(model.resid)
    print(f"Durbin-Watson (time-ordered): {dw:.3f}")
    print(
        "  -> ~2.0 = no lag-1 autocorrelation, <1.5 = notable positive "
        "autocorrelation, >2.5 = notable negative autocorrelation.\n"
        "  Compare this to 08's DW=1.994, which was computed on shuffled "
        "rows and is not a valid comparison point -- this is the number "
        "that actually means something."
    )

    out_path = str(data.RESULTS_DIR / "residual_acf_time_ordered.png")
    time_series.plot_acf(model.resid, lags=data.ACF_LAGS, out_path=out_path)
    print(f"\nSaved ACF plot to {out_path}")
    print(
        "Look for: a slow-decaying ACF (trend), or spikes at lag~24 "
        "(daily cycle) -- either would mean pickup_datetime carries signal "
        "this model isn't using yet, motivating the timeseries features "
        "flagged in the original handoff (hour/day/season, ADF test)."
    )


if __name__ == "__main__":
    main()
