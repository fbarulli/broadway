"""02: when does demand peak, and is the weekend dip worth a covariate?

Builds on the hourly buckets from 01 (complete hours only; the partial terminal
hour is excluded). Publishes the weekend-vs-weekday gap decision plus the
hour profile, weekday profile and hour×weekday evidence tables.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from _common import DATETIME_COL, RESULTS, SEED, load_canonical, load_decision_rule

from broadway.reports.decision import Evidence, SegmentEvidence
from broadway.reports.measurement import MeasurementEstimator, MeasurementSpec
from broadway.reports.step_output import StepOutput

DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
STEM = "02_seasonality"
BLOCK = "day"


def _weekend_gap(frame: pd.DataFrame) -> float:
    """|1 - mean weekend trips / mean weekday trips| (0 = no weekend effect)."""
    weekend = frame.loc[frame["is_weekend"] == 1, "trips"]
    weekday = frame.loc[frame["is_weekend"] == 0, "trips"]
    if weekend.empty or weekday.empty:
        raise ValueError("weekend gap needs both weekday and weekend hours")
    return float(abs(1.0 - weekend.mean() / weekday.mean()))


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_canonical(columns=[DATETIME_COL])
    ts = pd.to_datetime(df[DATETIME_COL])
    end = ts.max()

    complete = ts[ts < end.floor("h")]
    hours = complete.dt.floor("h")
    hourly = hours.value_counts().sort_index()

    frame = pd.DataFrame(
        {
            "hour": hourly.index.hour,
            "dow": hourly.index.dayofweek,
            "trips": hourly.values,
        },
        index=hourly.index,
    ).assign(
        is_weekend=lambda part: (part["dow"] >= 5).astype(int),
        day=lambda part: part.index.floor("D"),
    )
    hour_profile = frame.groupby("hour")["trips"].agg(["mean", "std"]).reset_index()
    dow_profile = frame.groupby("dow")["trips"].agg(["mean", "std"]).reset_index()
    pivot = frame.pivot_table(index="dow", columns="hour", values="trips", aggfunc="mean")

    rule = load_decision_rule(STEM)
    spec = MeasurementSpec(
        bootstrap="block",
        block_column=BLOCK,
        null="block_shuffle",
        null_column="is_weekend",
        iterations=200,
        seed=SEED,
    )
    measurement = MeasurementEstimator().estimate(
        frame, _weekend_gap, spec, name=rule.metric, unit="relative_gap"
    )

    peak_hour = int(hour_profile.loc[hour_profile["mean"].idxmax(), "hour"])
    trough_hour = int(hour_profile.loc[hour_profile["mean"].idxmin(), "hour"])
    peak_dow = DOW_LABELS[int(dow_profile.loc[dow_profile["mean"].idxmax(), "dow"])]
    trough_dow = DOW_LABELS[int(dow_profile.loc[dow_profile["mean"].idxmin(), "dow"])]
    weekday = dow_profile.loc[dow_profile["dow"] < 5, "mean"].mean()
    weekend = dow_profile.loc[dow_profile["dow"] >= 5, "mean"].mean()

    evidence = Evidence(
        baseline="no weekend effect: weekend and weekday hourly demand are equal",
        segments=[
            SegmentEvidence(
                segment="weekday_hours",
                value=float(frame.loc[frame["is_weekend"] == 0, "trips"].mean()),
                sample_size=int((frame["is_weekend"] == 0).sum()),
            ),
            SegmentEvidence(
                segment="weekend_hours",
                value=float(frame.loc[frame["is_weekend"] == 1, "trips"].mean()),
                sample_size=int((frame["is_weekend"] == 1).sum()),
            ),
        ],
        sensitivity={
            "weekend_over_weekday_ratio": float(weekend / weekday),
            "peak_hour_mean_trips": float(hour_profile["mean"].max()),
            "trough_hour_mean_trips": float(hour_profile["mean"].min()),
        },
        falsifier="the weekend-gap interval falls below the 10% effect floor",
        provenance={
            "artifact": "taxi_canonical.parquet",
            "source": "data/processed/taxi_canonical.parquet",
            "seed": str(measurement.seed),
            "analysis_sample": str(measurement.sample_size),
            "block": BLOCK,
            "resampling": measurement.resampling,
            "exclusions": "partial terminal hour excluded",
        },
    )
    decision = measurement.to_decision(
        rule,
        question="Is the weekend dip large enough to carry day-of-week in the forecast?",
        evidence=evidence,
    )

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

    output = StepOutput(
        results_dir=RESULTS, stem=STEM, decision=decision, source=Path(__file__)
    )
    output.add_table("hour_profile", hour_profile)
    output.add_table("dow", dow_profile)
    output.add_table("hour_x_dow", pivot.reset_index())
    output.save_figure(figure)
    plt.close("all")
    report, plot, decision_path = output.write()
    print(
        f"peak hour {peak_hour:02d}:00, trough hour {trough_hour:02d}:00, "
        f"peak day {peak_dow}, trough day {trough_dow}, "
        f"weekend/weekday {weekend / weekday:.2f}"
    )
    print(f"report: {report}")
    print(f"plot: {plot}")
    print(
        f"decision: {decision_path} -> {decision.verdict} "
        f"({rule.metric} {measurement.value:.4f} "
        f"[{decision.interval_low:.4f}, {decision.interval_high:.4f}], "
        f"n={measurement.sample_size:,}, null={decision.null_threshold:.4f})"
    )


if __name__ == "__main__":
    main()
