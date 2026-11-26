"""05: temporal features on the v3 sample — shared recipe from _common.

Answers whether pickup time carries fare signal worth keeping: the share of
fare variance explained by pickup hour (eta squared), measured and resampled
on the same day-blocked analysis sample.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from _common import (
    RESULTS,
    SAMPLE_NAME,
    SEED,
    TEMPORAL_FEATURES,
    build_temporal_features,
    load_decision_rule,
)

from broadway.reports.decision import Evidence, SegmentEvidence
from broadway.reports.measurement import MeasurementEstimator, MeasurementSpec
from broadway.reports.step_output import StepOutput
from broadway.samples import read_named_sample

FLAG_COLS = ["is_weekend", "is_rush_hour", "is_night"]
CYCLICAL_COLS = ["hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos"]
FEATURE_COLS = [feat.name for feat in TEMPORAL_FEATURES]
STEM = "05_temporal_features"
ANALYSIS_SAMPLE = 50_000
BLOCK = "day"


def _eta_squared(frame: pd.DataFrame, group: str, target: str = "fare_amount") -> float:
    """Between-group share of the target's variance (one-way eta squared)."""
    grouped = frame.groupby(group, observed=True)[target]
    grand = float(frame[target].mean())
    between = float((grouped.count() * (grouped.mean() - grand) ** 2).sum())
    total = float(((frame[target] - grand) ** 2).sum())
    return between / total if total else 0.0


def plot_temporal(df: pd.DataFrame) -> object:
    """Plot the temporal story a human reads: mean fare by hour and weekday."""
    hourly = df.groupby("pickup_hour")["fare_amount"].mean()
    weekday = df.groupby("pickup_day_of_week")["fare_amount"].mean()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    hourly.plot(ax=axes[0], marker="o", color="#4c78a8")
    axes[0].set_title("mean fare by pickup hour")
    axes[0].set_xlabel("hour")
    axes[0].set_ylabel("mean fare")
    weekday.plot(ax=axes[1], marker="o", color="#e45756")
    axes[1].set_title("mean fare by weekday")
    axes[1].set_xlabel("weekday (0=Mon)")
    return figure


def _evidence_table(df: pd.DataFrame) -> pd.DataFrame:
    """Flag counts plus the cyclical/flag feature describe (one long table)."""
    rows = []
    for col in [*FLAG_COLS, *CYCLICAL_COLS]:
        counts = df[col].value_counts().sort_index()
        for value, count in counts.items():
            rows.append({"feature": col, "value": value, "count": int(count)})
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    sample = read_named_sample(SAMPLE_NAME)
    df = build_temporal_features(sample.df)
    print(f"Derived feature columns ({len(TEMPORAL_FEATURES)}): {FEATURE_COLS}")

    analysis = df.sample(n=min(ANALYSIS_SAMPLE, len(df)), random_state=SEED).assign(
        day=lambda part: part["pickup_datetime"].dt.floor("D")
    )
    rule = load_decision_rule(STEM)
    spec = MeasurementSpec(
        bootstrap="block",
        block_column=BLOCK,
        null="block_shuffle",
        null_column="fare_amount",
        iterations=200,
        seed=SEED,
    )
    measurement = MeasurementEstimator().estimate(
        analysis,
        lambda part: _eta_squared(part, "pickup_hour"),
        spec,
        name=rule.metric,
        unit="variance_share",
    )
    n = measurement.sample_size
    weekend = analysis["is_weekend"] == 1
    half = analysis.sample(n=max(2, len(analysis) // 2), random_state=SEED + 1)
    evidence = Evidence(
        baseline="no temporal signal: fare variance is unrelated to pickup hour",
        segments=[
            SegmentEvidence(
                segment="weekday",
                value=_eta_squared(analysis[~weekend], "pickup_hour"),
                sample_size=int((~weekend).sum()),
            ),
            SegmentEvidence(
                segment="weekend",
                value=_eta_squared(analysis[weekend], "pickup_hour"),
                sample_size=int(weekend.sum()),
            ),
        ],
        sensitivity={
            f"day_of_week_eta_squared_n{n}": _eta_squared(analysis, "pickup_day_of_week"),
            f"hour_eta_squared_weekday_n{int((~weekend).sum())}": _eta_squared(
                analysis[~weekend], "pickup_hour"
            ),
            f"hour_eta_squared_second_seed_n{len(half)}": _eta_squared(
                half, "pickup_hour"
            ),
        },
        falsifier="the hour eta squared interval falls below the 1% variance floor",
        provenance={
            "artifact": sample.provenance["name"],
            "artifact_sha256": sample.provenance["artifact_sha256"],
            "seed": str(measurement.seed),
            "analysis_sample": str(measurement.sample_size),
            "block": BLOCK,
            "resampling": measurement.resampling,
            "exclusions": "temporal flags and cyclical encodings from the shared recipe",
        },
    )
    decision = measurement.to_decision(
        rule,
        question="Does pickup time carry fare signal worth keeping in the model?",
        evidence=evidence,
    )
    output = StepOutput(
        results_dir=RESULTS, stem=STEM, decision=decision, source=Path(__file__)
    )
    output.add_table("features", _evidence_table(df))
    output.add_table("describe", df[FEATURE_COLS].describe().reset_index())
    output.save_figure(plot_temporal(df))
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
