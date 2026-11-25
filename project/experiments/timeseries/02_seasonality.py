"""02: day-of-week and hour-of-day demand seasonality (read-only).

Answers: when does demand peak — which hours, which weekdays — and how large
is the weekend dip. Builds on the hourly buckets from 01 (complete hours
only; the partial terminal hour is excluded).
Outputs (all under results/timeseries/):
- 02_seasonality_hour_profile.csv: mean/std trips per hour-of-day (0-23)
- 02_seasonality_dow.csv: mean/std trips per day-of-week (Mon-Sun)
- 02_seasonality_hour_x_dow.csv: mean trips pivot (dow rows x hour columns)
- 02_seasonality.md: peak hour/day, weekend-vs-weekday ratio
- 02_seasonality.png: hour curve, weekday bars, hour-x-dow heatmap
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from _common import DATETIME_COL, RESULTS, load_canonical

DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_canonical(columns=[DATETIME_COL])
    ts = pd.to_datetime(df[DATETIME_COL])
    end = ts.max()

    complete = ts[ts < end.floor("h")]
    hours = complete.dt.floor("h")
    hourly = hours.value_counts().sort_index()

    frame = pd.DataFrame(
        {"hour": hourly.index.hour, "dow": hourly.index.dayofweek, "trips": hourly.values}
    )
    hour_profile = frame.groupby("hour")["trips"].agg(["mean", "std"]).reset_index()
    dow_profile = frame.groupby("dow")["trips"].agg(["mean", "std"]).reset_index()
    pivot = frame.pivot_table(index="dow", columns="hour", values="trips", aggfunc="mean")

    stem = Path(__file__).stem
    hour_profile.to_csv(RESULTS / f"{stem}_hour_profile.csv", index=False)
    dow_profile.to_csv(RESULTS / f"{stem}_dow.csv", index=False)
    pivot.to_csv(RESULTS / f"{stem}_hour_x_dow.csv")

    peak_hour = int(hour_profile.loc[hour_profile["mean"].idxmax(), "hour"])
    trough_hour = int(hour_profile.loc[hour_profile["mean"].idxmin(), "hour"])
    peak_dow = DOW_LABELS[int(dow_profile.loc[dow_profile["mean"].idxmax(), "dow"])]
    trough_dow = DOW_LABELS[int(dow_profile.loc[dow_profile["mean"].idxmin(), "dow"])]
    weekday = dow_profile.loc[dow_profile["dow"] < 5, "mean"].mean()
    weekend = dow_profile.loc[dow_profile["dow"] >= 5, "mean"].mean()

    # plt.subplots is untyped (partially unknown): pin the boundary to Any once.
    subplots: Any = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
    figure, axes = subplots
    axes[0].plot(hour_profile["hour"], hour_profile["mean"], color="#2563eb")
    axes[0].fill_between(
        hour_profile["hour"],
        hour_profile["mean"] - hour_profile["std"],
        hour_profile["mean"] + hour_profile["std"],
        color="#2563eb",
        alpha=0.2,
    )
    axes[0].set(title="Mean trips by hour of day (±1 std)", xlabel="hour", ylabel="trips")
    axes[1].bar(
        [DOW_LABELS[d] for d in dow_profile["dow"]], dow_profile["mean"], color="#0f766e"
    )
    axes[1].set(title="Mean trips by weekday", xlabel="", ylabel="trips")
    mesh = axes[2].imshow(pivot.values, aspect="auto", cmap="YlGnBu")
    axes[2].set(
        title="Mean trips: weekday x hour",
        xlabel="hour",
        yticks=list(range(7)),
        yticklabels=DOW_LABELS,
    )
    figure.colorbar(mesh, ax=axes[2], label="trips")
    figure.savefig(RESULTS / f"{stem}.png", dpi=160)
    plt.close(figure)

    lines = [
        "# Seasonality — hourly and weekday demand",
        "",
        f"- complete hours analyzed: {len(hourly)}",
        f"- peak hour: {peak_hour:02d}:00, trough hour: {trough_hour:02d}:00",
        f"- peak day: {peak_dow}, trough day: {trough_dow}",
        (
            f"- weekday mean: {weekday:.0f} trips/hour, weekend mean: {weekend:.0f} "
            f"(ratio {weekend / weekday:.2f})"
        ),
        "",
        (
            f"Artifacts: {stem}_hour_profile.csv, {stem}_dow.csv, "
            f"{stem}_hour_x_dow.csv, {stem}.png (this directory)."
        ),
    ]
    (RESULTS / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"peak hour {peak_hour:02d}:00, peak day {peak_dow}, weekend/weekday {weekend / weekday:.2f}")
    print(f"wrote {stem}_hour_profile.csv, {stem}_dow.csv, {stem}_hour_x_dow.csv, {stem}.md, {stem}.png")


if __name__ == "__main__":
    main()
