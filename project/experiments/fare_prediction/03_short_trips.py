"""03: profile the short trips (< 0.31 mi) subset of the named fare_prediction_1m sample.

Answers whether short trips are a material slice that deserves its own error
read (share of the sample below the distance floor). Short trips are a slice
of the validated named sample (only the distance subset is applied here); the
fare/duration policy lives in the sample definition and was applied once at
generation.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from _common import RESULTS, SAMPLE_NAME, SEED, load_decision_rule

from broadway.reports.decision import Evidence, SegmentEvidence
from broadway.reports.measurement import MeasurementEstimator, MeasurementSpec
from broadway.reports.step_output import StepOutput
from broadway.samples import read_named_sample

SHORT_DISTANCE = 0.31
SHORT_COLS = ["fare_amount", "trip_duration_minutes"]
SHORT_PERCENTILES = [0.01, 0.25, 0.5, 0.75, 0.99]
STEM = "03_short_trips"
ANALYSIS_SAMPLE = 50_000
BLOCK = "day"


def _short_share(frame: pd.DataFrame, distance: float = SHORT_DISTANCE) -> float:
    """Share of rows whose distance falls strictly inside (0, distance)."""
    return float(((frame["trip_distance"] > 0) & (frame["trip_distance"] < distance)).mean())


def plot_short_trips(df: pd.DataFrame, short: pd.DataFrame) -> object:
    """Two KDE panels: all trips vs short trips (fare, duration) on log-x."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for ax, col in zip(axes, SHORT_COLS):
        sns.kdeplot(df[col], label="all trips", color="#4c72b0", ax=ax)
        sns.kdeplot(short[col], label="short trips (<0.31 mi)", color="#d62728", ax=ax)
        ax.set_xscale("log")
        ax.set_xlabel(col)
        ax.set_ylabel("density")
        ax.set_title(f"{col}: all vs short trips")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")
    return fig


def _segments(df: pd.DataFrame) -> list[SegmentEvidence]:
    """Re-measure the short-trip share inside weekday/weekend slices."""
    weekend = df["pickup_datetime"].dt.dayofweek >= 5
    return [
        SegmentEvidence(
            segment=label,
            value=_short_share(df.loc[mask]),
            sample_size=int(mask.sum()),
        )
        for label, mask in (("weekday", ~weekend), ("weekend", weekend))
    ]


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    sample = read_named_sample(SAMPLE_NAME)
    df = sample.df
    short = df[(df["trip_distance"] > 0) & (df["trip_distance"] < SHORT_DISTANCE)]
    print(f"Total short trips: {len(short):,} of {len(df):,}")

    rule = load_decision_rule(STEM)
    analysis = df.sample(n=min(ANALYSIS_SAMPLE, len(df)), random_state=SEED).assign(
        day=lambda part: part["pickup_datetime"].dt.floor("D")
    )
    spec = MeasurementSpec(
        bootstrap="block",
        block_column=BLOCK,
        iterations=200,
        seed=SEED,
    )
    measurement = MeasurementEstimator().estimate(
        analysis, _short_share, spec, name=rule.metric, unit="share"
    )
    n = measurement.sample_size
    evidence = Evidence(
        baseline="zero short-trip mass (the slice is absent from the sample)",
        segments=_segments(analysis),
        sensitivity={
            f"share_under_0_5_mi_n{n}": _short_share(analysis, 0.5),
            f"share_under_0_2_mi_n{n}": _short_share(analysis, 0.2),
        },
        falsifier="the 95% interval of the short-trip share falls below the 2% floor",
        provenance={
            "artifact": sample.provenance["name"],
            "artifact_sha256": sample.provenance["artifact_sha256"],
            "seed": str(measurement.seed),
            "analysis_sample": str(measurement.sample_size),
            "block": BLOCK,
            "resampling": measurement.resampling,
            "exclusions": "distance floor 0.31 mi applied for the slice only",
        },
    )
    decision = measurement.to_decision(
        rule,
        question="Are short trips a material slice that deserves its own error read?",
        evidence=evidence,
    )
    output = StepOutput(
        results_dir=RESULTS, stem=STEM, decision=decision, source=Path(__file__)
    )
    output.add_table(
        "describe",
        short[SHORT_COLS].describe(percentiles=SHORT_PERCENTILES).reset_index(),
    )
    output.save_figure(plot_short_trips(analysis, short))
    plt.close("all")
    report, figure, decision_path = output.write()
    print(f"report: {report}")
    print(f"plot: {figure}")
    print(
        f"decision: {decision_path} -> {decision.verdict} "
        f"({rule.metric} {measurement.value:.4f} "
        f"[{decision.interval_low:.4f}, {decision.interval_high:.4f}], "
        f"n={measurement.sample_size:,}, null={decision.null_threshold})"
    )


if __name__ == "__main__":
    main()
