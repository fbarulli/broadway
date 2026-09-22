"""01: temporal profile of the canonical frame (read-only).

Answers: what range is covered, at what frequency, and where the gaps are.
Outputs (all under results/timeseries/):
- 01_series_profile.md: range, row count, hourly/daily coverage, missing hours
- 01_series_profile_hourly_counts.csv: trips per pickup hour (the base demand series)
- 01_series_profile_daily_counts.csv: trips per pickup date
- 01_series_profile.png: hourly and daily demand plots
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from _common import DATETIME_COL, RESULTS, load_canonical


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_canonical(columns=[DATETIME_COL])
    ts = pd.to_datetime(df[DATETIME_COL])
    start, end = ts.min(), ts.max()
    print(f"range: {start} -> {end} ({len(df)} rows)")

    hourly = ts.dt.floor("h").value_counts().sort_index()
    full_hours = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    missing_hours = full_hours.difference(hourly.index)
    print(f"hours covered: {len(hourly)}/{len(full_hours)} (missing: {len(missing_hours)})")

    stem = Path(__file__).stem
    hourly_csv = RESULTS / f"{stem}_hourly_counts.csv"
    hourly.rename("trips").to_csv(hourly_csv)
    daily = ts.dt.floor("D").value_counts().sort_index()
    daily_csv = RESULTS / f"{stem}_daily_counts.csv"
    daily.rename("trips").to_csv(daily_csv)

    # The canonical frame ends partway through an hour on April 1. Preserve
    # every raw bucket in the CSVs, but omit incomplete terminal buckets from
    # plots and extrema so a two-row boundary does not look like demand collapse.
    complete_hourly = hourly[hourly.index < end.floor("h")]
    complete_daily = daily[daily.index < end.normalize()]
    figure, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    complete_hourly.plot(ax=axes[0], color="#2563eb", linewidth=0.8)
    axes[0].set(title="Hourly NYC taxi demand", xlabel="", ylabel="trips")
    complete_daily.plot(ax=axes[1], color="#0f766e", linewidth=1.5)
    axes[1].set(title="Daily NYC taxi demand", xlabel="pickup date", ylabel="trips")
    figure.savefig(RESULTS / f"{stem}.png", dpi=160)
    plt.close(figure)

    lines = [
        "# Series profile — canonical taxi frame",
        "",
        f"- rows: {len(df)}",
        f"- range: {start} .. {end}",
        f"- hourly buckets: {len(hourly)} present / {len(full_hours)} expected",
        f"- missing hours: {len(missing_hours)}",
        f"- busiest complete hour: {complete_hourly.idxmax()} ({complete_hourly.max()} trips)",
        f"- quietest complete hour: {complete_hourly.idxmin()} ({complete_hourly.min()} trips)",
        f"- busiest complete day: {complete_daily.idxmax().date()} ({complete_daily.max()} trips)",
        f"- quietest complete day: {complete_daily.idxmin().date()} ({complete_daily.min()} trips)",
        "- incomplete terminal hour/day: retained in CSVs, excluded from plot and extrema",
        "",
        f"Artifacts: {hourly_csv.name}, {daily_csv.name}, {stem}.png (this directory).",
    ]
    (RESULTS / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {hourly_csv}, {daily_csv}, {stem}.md, {stem}.png")


if __name__ == "__main__":
    main()
