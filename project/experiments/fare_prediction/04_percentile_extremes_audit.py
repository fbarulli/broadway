"""04: audit the bottom and top 1% of fares for plausibility.

Answers whether the fare extremes contain implausible trips the policy must
exclude (flagged share of the bottom/top 1%).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from _common import RESULTS, SAMPLE_NAME, SEED, UNIT_FMT, load_decision_rule
from matplotlib.ticker import FuncFormatter

from broadway.reports.decision import Evidence, SegmentEvidence
from broadway.reports.measurement import MeasurementEstimator, MeasurementSpec
from broadway.reports.step_output import StepOutput
from broadway.samples import read_named_sample

BOTTOM_Q = 0.01
TOP_Q = 0.99
PROFILE_COLS = ["fare_amount", "trip_distance", "trip_duration_minutes"]
BOTTOM_SUSPECT = {"max_distance": 5.0, "max_duration": 30.0, "max_speed": 60.0}
TOP_SUSPECT = {"min_distance": 3.0, "min_duration": 5.0, "max_speed": 100.0}
FLAG_COLS = ["fare_amount", "trip_distance", "trip_duration_minutes", "speed_mph"]
PERCENTILES = [0.01, 0.25, 0.5, 0.75, 0.99]
STEM = "04_percentile_extremes_audit"
ANALYSIS_SAMPLE = 50_000
BLOCK = "day"
ALTERNATE_SPEED_CAP = 40.0


def _profile(subset: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Describe ``subset`` on ``cols`` with the percentile bands of interest."""
    return subset[cols].describe(percentiles=PERCENTILES)


def _flag_mask(frame: pd.DataFrame, thresholds: dict[str, float], high: bool) -> np.ndarray:
    """Boolean plausibility violations for one extreme's fare band."""
    speed = frame["trip_distance"] / (frame["trip_duration_minutes"] / 60.0)
    speed = speed.replace([np.inf, -np.inf], np.finfo(float).max)
    if high:
        return (
            (frame["trip_distance"] < thresholds["min_distance"])
            | (frame["trip_duration_minutes"] < thresholds["min_duration"])
            | (speed > thresholds["max_speed"])
        ).to_numpy()
    return (
        (frame["trip_distance"] > thresholds["max_distance"])
        | (frame["trip_duration_minutes"] > thresholds["max_duration"])
        | (speed > thresholds["max_speed"])
    ).to_numpy()


def _flagged(frame: pd.DataFrame, thresholds: dict[str, float], high: bool) -> pd.DataFrame:
    """Rows of ``frame`` violating the plausibility thresholds (with speed)."""
    flagged = frame.copy()
    flagged["speed_mph"] = flagged["trip_distance"] / (
        flagged["trip_duration_minutes"] / 60.0
    )
    return flagged[_flag_mask(frame, thresholds, high)]


def _plausible_share(frame: pd.DataFrame) -> float:
    """Share of the extremes that passes the plausibility rules.

    The statistic recomputes the flags from the raw columns (never a
    precomputed flag), so the block-shuffled null actually breaks the joint
    that produces a flag: under a shuffled duration only chance-level
    plausibility survives.
    """
    top = frame["top"].to_numpy(dtype=bool)
    flags = np.concatenate([
        _flag_mask(frame[~top], BOTTOM_SUSPECT, high=False),
        _flag_mask(frame[top], TOP_SUSPECT, high=True),
    ])
    return float(1.0 - flags.mean())


def _combined_evidence(
    bottom_desc: pd.DataFrame,
    top_desc: pd.DataFrame,
    bottom_flagged: pd.DataFrame,
    top_flagged: pd.DataFrame,
) -> pd.DataFrame:
    """Join both describes and both flagged sets into ONE evidence table.

    Index = describe stat labels / suspicious row numbers; ``group``
    distinguishes the sections; ``speed_mph`` is NaN on describe rows.
    """
    rows: list[dict[str, object]] = []
    for label, desc in (("bottom_describe", bottom_desc), ("top_describe", top_desc)):
        for stat, row in desc.iterrows():
            rows.append({
                "group": label, "stat": stat,
                "fare_amount": row["fare_amount"],
                "trip_distance": row["trip_distance"],
                "trip_duration_minutes": row["trip_duration_minutes"],
                "speed_mph": np.nan,
            })
    for label, flagged in (("suspicious_bottom", bottom_flagged),
                           ("suspicious_top", top_flagged)):
        for i, (_, row) in enumerate(flagged.iterrows(), 1):
            rows.append({
                "group": label, "stat": i,
                "fare_amount": row["fare_amount"],
                "trip_distance": row["trip_distance"],
                "trip_duration_minutes": row["trip_duration_minutes"],
                "speed_mph": row["speed_mph"],
            })
    return pd.DataFrame(rows).reset_index(drop=True)


def plot_extremes(df: pd.DataFrame, bottom: pd.DataFrame, top: pd.DataFrame) -> object:
    """Two-panel figure: bottom-vs-top fare boxplot plus fare-density comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    extremes = pd.concat([
        bottom.assign(group="bottom 1%"),
        top.assign(group="top 1%"),
    ])
    sns.boxplot(
        x="group", y="fare_amount", data=extremes, log_scale=True,
        whis=[0, 100], showmeans=True, ax=axes[0],
        meanprops={"marker": "o", "markerfacecolor": "#d62728",
                   "markeredgecolor": "#d62728", "markersize": 5},
    )
    axes[0].yaxis.set_major_formatter(
        FuncFormatter(lambda v, _p: UNIT_FMT["fare_amount"].format(v))
    )
    axes[0].set_title("bottom vs top 1% fares")
    axes[0].grid(True, alpha=0.3, axis="y")
    sns.kdeplot(df["fare_amount"], label="all trips", color="#7f7f7f", ax=axes[1])
    sns.kdeplot(bottom["fare_amount"], label="bottom 1%", color="#4c72b0", ax=axes[1])
    sns.kdeplot(top["fare_amount"], label="top 1%", color="#d62728", ax=axes[1])
    axes[1].set_xscale("log")
    axes[1].set_xlabel("fare_amount")
    axes[1].set_ylabel("density")
    axes[1].legend(fontsize=7)
    axes[1].set_title("fare density: all vs bottom/top 1%")
    axes[1].grid(True, alpha=0.3, axis="y")
    return fig


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    sample = read_named_sample(SAMPLE_NAME)
    df = sample.df

    q01 = float(df["fare_amount"].quantile(BOTTOM_Q))
    q99 = float(df["fare_amount"].quantile(TOP_Q))
    bottom = df[df["fare_amount"] <= q01][[*PROFILE_COLS, "pickup_datetime"]]
    top = df[df["fare_amount"] >= q99][[*PROFILE_COLS, "pickup_datetime"]]
    print(f"Bottom 1% (fare <= ${q01:,.2f}): {len(bottom):,} rows")
    print(f"Top 1% (fare >= ${q99:,.2f}): {len(top):,} rows")

    bottom_flagged = _flagged(bottom, BOTTOM_SUSPECT, high=False)
    top_flagged = _flagged(top, TOP_SUSPECT, high=True)
    print(f"Bottom flagged: {len(bottom_flagged)}")
    print(f"Top flagged: {len(top_flagged)}")

    frame = pd.concat([
        bottom.assign(top=False),
        top.assign(top=True),
    ]).assign(day=lambda part: part["pickup_datetime"].dt.floor("D"))
    analysis = frame.sample(n=min(ANALYSIS_SAMPLE, len(frame)), random_state=SEED)
    rule = load_decision_rule(STEM)
    spec = MeasurementSpec(
        bootstrap="block",
        block_column=BLOCK,
        null="block_shuffle",
        null_column="trip_duration_minutes",
        iterations=200,
        seed=SEED,
    )
    measurement = MeasurementEstimator().estimate(
        frame, _plausible_share, spec, name=rule.metric, unit="share"
    )
    n = measurement.sample_size
    bottom_analysis = analysis[~analysis["top"]]
    top_analysis = analysis[analysis["top"]]
    half = frame.sample(n=max(2, len(analysis) // 2), random_state=SEED + 1)
    evidence = Evidence(
        baseline="plausible extremes (no implausible trip in the bottom/top 1%)",
        segments=[
            SegmentEvidence(
                segment="bottom_1pct",
                value=float(1.0 - _flag_mask(
                    bottom_analysis, BOTTOM_SUSPECT, high=False
                ).mean()),
                sample_size=len(bottom_analysis),
            ),
            SegmentEvidence(
                segment="top_1pct",
                value=float(1.0 - _flag_mask(
                    top_analysis, TOP_SUSPECT, high=True
                ).mean()),
                sample_size=len(top_analysis),
            ),
        ],
        sensitivity={
            f"plausible_share_at_40mph_cap_n{n}": float(1.0 - np.concatenate([
                _flag_mask(bottom_analysis, {**BOTTOM_SUSPECT, "max_speed": ALTERNATE_SPEED_CAP}, high=False),
                _flag_mask(top_analysis, {**TOP_SUSPECT, "max_speed": ALTERNATE_SPEED_CAP}, high=True),
            ]).mean()),
            f"plausible_share_second_seed_n{len(half)}": _plausible_share(half),
        },
        falsifier="the plausible-share interval falls below the 95% floor",
        provenance={
            "artifact": sample.provenance["name"],
            "artifact_sha256": sample.provenance["artifact_sha256"],
            "seed": str(measurement.seed),
            "analysis_sample": str(measurement.sample_size),
            "block": BLOCK,
            "resampling": measurement.resampling,
            "exclusions": "bottom/top 1% quantile cut at the measured fare",
        },
    )
    decision = measurement.to_decision(
        rule,
        question="Are the fare extremes plausible after the policy?",
        evidence=evidence,
    )
    output = StepOutput(
        results_dir=RESULTS, stem=STEM, decision=decision, source=Path(__file__)
    )
    output.add_table(
        "describe",
        _combined_evidence(
            _profile(bottom, PROFILE_COLS), _profile(top, PROFILE_COLS),
            bottom_flagged[FLAG_COLS], top_flagged[FLAG_COLS],
        ),
    )
    output.save_figure(plot_extremes(analysis, bottom, top))
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
