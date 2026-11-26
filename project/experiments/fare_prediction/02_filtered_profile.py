"""02: profile the named fare_prediction_1m sample.

The fare/distance/duration policy lives in the sample definition
(project/config/sample/fare_prediction_1m.yaml), applied once at generation; this
step only consumes the validated sample by name. It answers whether the fare
surface is spread widely enough to need a transformed target and fare-band
segmentation (p99/p01 ratio).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import RESULTS, SAMPLE_NAME, SEED, UNIT_FMT, box_with_marks, load_decision_rule

from broadway.reports.decision import Evidence, SegmentEvidence
from broadway.reports.measurement import MeasurementEstimator, MeasurementSpec
from broadway.reports.step_output import StepOutput
from broadway.samples import read_named_sample

COLS = ["fare_amount", "trip_distance", "trip_duration_minutes"]
PERCENTILES = [0.01, 0.05, 0.50, 0.95, 0.99, 0.999, 1.0]
STEM = "02_filtered_profile"
ANALYSIS_SAMPLE = 50_000
BLOCK = "day"


def _ratio(frame: pd.DataFrame, low: float = 0.01, high: float = 0.99) -> float:
    """p<high> / p<low> of fares (the tail-spread metric)."""
    values = frame["fare_amount"].to_numpy(dtype=float)
    q_low, q_high = np.percentile(values, [low * 100, high * 100])
    return float(q_high / q_low)


def plot_profiles(df: pd.DataFrame) -> object:
    """One figure, 3 box-with-marks panels on log-y (shared helper)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, col in zip(axes, COLS):
        box_with_marks(ax, df[col], UNIT_FMT[col], f"{col} (N={len(df)})")
    return fig


def _describe_table(df: pd.DataFrame) -> pd.DataFrame:
    """The aggregated evidence behind the figure (percentile describe)."""
    return df[COLS].describe(percentiles=PERCENTILES).reset_index()


def _segments(df: pd.DataFrame) -> list[SegmentEvidence]:
    """Re-measure the spread inside weekday/weekend slices."""
    weekend = df["pickup_datetime"].dt.dayofweek >= 5
    return [
        SegmentEvidence(
            segment=label,
            value=_ratio(df.loc[mask]),
            sample_size=int(mask.sum()),
        )
        for label, mask in (("weekday", ~weekend), ("weekend", weekend))
    ]


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    sample = read_named_sample(SAMPLE_NAME)
    df = sample.df
    print(f"sample: {SAMPLE_NAME}@{sample.provenance['version']}")
    print(f"rows: {sample.provenance['row_count']}")

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
        analysis, _ratio, spec, name=rule.metric, unit="ratio"
    )
    n = measurement.sample_size
    evidence = Evidence(
        baseline="uniform-spread reference: p99/p01 ratio of 10",
        segments=_segments(analysis),
        sensitivity={
            f"p999_over_p001_n{n}": _ratio(analysis, 0.001, 0.999),
            f"p95_over_p05_n{n}": _ratio(analysis, 0.05, 0.95),
        },
        falsifier="the 95% interval of the p99/p01 ratio falls below the 10x floor",
        provenance={
            "artifact": sample.provenance["name"],
            "artifact_sha256": sample.provenance["artifact_sha256"],
            "seed": str(measurement.seed),
            "analysis_sample": str(measurement.sample_size),
            "block": BLOCK,
            "resampling": measurement.resampling,
            "exclusions": "fare policy + plausibility exclusions in the sample registry",
        },
    )
    decision = measurement.to_decision(
        rule,
        question="Is the fare surface spread widely enough to need a transformed target?",
        evidence=evidence,
    )
    output = StepOutput(
        results_dir=RESULTS, stem=STEM, decision=decision, source=Path(__file__)
    )
    output.add_table("describe", _describe_table(df))
    output.save_figure(plot_profiles(analysis))
    plt.close("all")
    report, figure, decision_path = output.write()
    print(f"report: {report}")
    print(f"plot: {figure}")
    print(
        f"decision: {decision_path} -> {decision.verdict} "
        f"({rule.metric} {measurement.value:.2f} "
        f"[{decision.interval_low:.2f}, {decision.interval_high:.2f}], "
        f"n={measurement.sample_size:,}, null={decision.null_threshold})"
    )


if __name__ == "__main__":
    main()
