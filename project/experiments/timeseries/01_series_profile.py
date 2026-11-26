"""01: is the canonical frame contiguous enough for lag features?

Answers: what range is covered, at what frequency, and where the gaps are.
Outputs (all under results/timeseries/): the coverage decision record, the
hourly/daily count evidence tables, and the hourly + daily demand figure.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from _common import DATETIME_COL, RESULTS, SEED, load_canonical, load_decision_rule

from broadway.reports.decision import Evidence, SegmentEvidence
from broadway.reports.measurement import MeasurementEstimator, MeasurementSpec
from broadway.reports.step_output import StepOutput

STEM = "01_series_profile"
BLOCK = "day"


def _missing_share(frame: pd.DataFrame) -> float:
    """Share of the expected hourly grid with no observed rows."""
    return float(1.0 - frame["present"].mean())


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_canonical(columns=[DATETIME_COL])
    ts = pd.to_datetime(df[DATETIME_COL])
    start, end = ts.min(), ts.max()

    hourly = ts.dt.floor("h").value_counts().sort_index()
    full_hours = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    missing_hours = full_hours.difference(list(hourly.index))
    coverage = pd.DataFrame(
        {"hour": full_hours, "present": full_hours.isin(hourly.index)}
    ).assign(day=lambda part: part["hour"].dt.floor("D"))

    rule = load_decision_rule(STEM)
    spec = MeasurementSpec(
        bootstrap="block",
        block_column=BLOCK,
        iterations=200,
        seed=SEED,
    )
    measurement = MeasurementEstimator().estimate(
        coverage, _missing_share, spec, name=rule.metric, unit="share"
    )

    # The canonical frame ends partway through an hour on April 1. Preserve
    # every raw bucket in the CSVs, but omit incomplete terminal buckets from
    # plots and extrema so a two-row boundary does not look like demand collapse.
    complete_hourly = hourly[hourly.index < end.floor("h")]
    complete_daily = hourly.resample("D").sum()[lambda s: s.index < end.normalize()]
    subplots: Any = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    figure, axes = subplots
    complete_hourly.plot(ax=axes[0], color="#2563eb", linewidth=0.8)
    axes[0].set(title="Hourly NYC taxi demand", xlabel="", ylabel="trips")
    complete_daily.plot(ax=axes[1], color="#0f766e", linewidth=1.5)
    axes[1].set(title="Daily NYC taxi demand", xlabel="pickup date", ylabel="trips")

    day_counts = ts.dt.floor("D").value_counts().sort_index().rename("trips")
    days = coverage["day"].sort_values().unique()
    early, late = days[: len(days) // 2], days[len(days) // 2 :]
    evidence = Evidence(
        baseline="a complete hourly grid (every expected hour observed)",
        segments=[
            SegmentEvidence(
                segment=label,
                value=_missing_share(coverage[coverage["day"].isin(window)]),
                sample_size=int(coverage["day"].isin(window).sum()),
            )
            for label, window in (("first_half", early), ("second_half", late))
        ],
        sensitivity={
            "missing_hours": float(len(missing_hours)),
            "expected_hours": float(len(full_hours)),
        },
        falsifier="the missing-hour share interval exceeds the 1% ceiling",
        provenance={
            "artifact": "taxi_canonical.parquet",
            "source": "data/processed/taxi_canonical.parquet",
            "seed": str(measurement.seed),
            "analysis_sample": str(measurement.sample_size),
            "block": BLOCK,
            "resampling": measurement.resampling,
            "exclusions": "partial terminal hour/day retained in tables, excluded from plots",
        },
    )
    decision = measurement.to_decision(
        rule,
        question="Is the canonical frame contiguous enough for lag features?",
        evidence=evidence,
    )
    output = StepOutput(
        results_dir=RESULTS, stem=STEM, decision=decision, source=Path(__file__)
    )
    output.add_table(
        "hourly_counts",
        hourly.rename("trips").rename_axis("pickup_datetime").reset_index(),
    )
    output.add_table(
        "daily_counts",
        day_counts.rename_axis("pickup_datetime").reset_index(),
    )
    output.add_table(
        "coverage",
        pd.DataFrame(
            {
                "rows": [len(df)],
                "start": [str(start)],
                "end": [str(end)],
                "expected_hours": [len(full_hours)],
                "missing_hours": [len(missing_hours)],
            }
        ),
    )
    output.save_figure(figure)
    plt.close("all")
    report, plot, decision_path = output.write()
    print(f"range: {start} -> {end} ({len(df)} rows)")
    print(f"hours covered: {len(hourly)}/{len(full_hours)} (missing: {len(missing_hours)})")
    print(f"report: {report}")
    print(f"plot: {plot}")
    print(
        f"decision: {decision_path} -> {decision.verdict} "
        f"({rule.metric} {measurement.value:.4f} "
        f"[{decision.interval_low:.4f}, {decision.interval_high:.4f}], n={measurement.sample_size:,})"
    )


if __name__ == "__main__":
    main()
