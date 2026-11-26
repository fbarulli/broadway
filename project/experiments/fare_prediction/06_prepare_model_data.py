"""06: chronological train/val/test split, temporal features, route frequency, raw location ids as categorical, log1p target.

The model feature set is SAFE_FEATURES (pre-trip only — no distance/duration/
speed, which are post-trip leakage). The only fitted encoding is the route
frequency, computed on the TRAIN slice only (unseen routes map to 0); raw
zone ids stay categorical and are handled natively by the model. No
target-derived column enters the feature set.

The step answers whether the chronological split leaves the test slice inside
the train-derived route frequency: the share of test routes unseen in train.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import (
    DATETIME_SRC,
    RESULTS,
    SAMPLE_NAME,
    SEED,
    build_temporal_features,
    load_decision_rule,
)

from broadway.features.transformers import FrequencyEncoding
from broadway.reports.decision import Evidence, SegmentEvidence
from broadway.reports.measurement import MeasurementEstimator, MeasurementSpec
from broadway.reports.step_output import StepOutput
from broadway.samples import read_named_sample

TRAIN_VAL_TEST = (0.8, 0.1, 0.1)
TARGET = "fare_amount"
TARGET_LOG = "fare_amount_log"
PU_COL = "pickup_location_id"
DO_COL = "dropoff_location_id"
ROUTE_COL = "route_id"
FREQUENCY_COL = "route_frequency"
PREPARED_DIR = RESULTS / "prepared"
SPLIT_CSV = RESULTS / "06_prepare_model_data_split.csv"
STEM = "06_prepare_model_data"
ANALYSIS_SAMPLE = 50_000
BLOCK = "day"

LOCATION_COLS = (PU_COL, DO_COL)


def time_split(
    df: pd.DataFrame, fractions: tuple[float, float, float]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/val/test split: train = earliest, test = latest."""
    ordered = df.sort_values(DATETIME_SRC).reset_index(drop=True)
    n = len(ordered)
    n_val = int(n * fractions[1])
    n_test = int(n * fractions[2])
    train = ordered.iloc[: n - n_val - n_test]
    val = ordered.iloc[n - n_val - n_test : n - n_test]
    test = ordered.iloc[n - n_test :]
    if any(part.empty for part in (train, val, test)):
        raise ValueError(f"empty split for fractions {fractions} over {n} rows")
    return train, val, test


def _prepare_split(df: pd.DataFrame) -> pd.DataFrame:
    """Temporal features, route id, categorical ids, log1p target."""
    df = build_temporal_features(df)
    df[ROUTE_COL] = df[PU_COL].astype(str) + "_" + df[DO_COL].astype(str)
    for col in LOCATION_COLS:
        df[col] = df[col].astype("category")
    df[TARGET_LOG] = np.log1p(df[TARGET])
    return df


def _apply_frequency(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Train-only route frequency; unseen routes map to 0 (never a target value)."""
    encoder = FrequencyEncoding(
        columns=[ROUTE_COL], feature_name=FREQUENCY_COL, normalize=False
    ).fit(splits["train"])
    return {name: encoder.transform(frame) for name, frame in splits.items()}


def _known_routes(train: pd.DataFrame) -> set[tuple[int, int]]:
    """The (pickup, dropoff) route pairs the train slice covers."""
    return set(zip(train[PU_COL].tolist(), train[DO_COL].tolist()))


def _unseen_share(known: set[tuple[int, int]], other: pd.DataFrame) -> float:
    """Share of ``other`` rows whose route id never appears in ``train``."""
    pairs = zip(other[PU_COL].tolist(), other[DO_COL].tolist())
    return float(np.mean([pair not in known for pair in pairs]))


def _evidence_rows(
    splits: dict[str, pd.DataFrame], unseen: dict[str, float]
) -> list[dict[str, object]]:
    """One evidence row per split: size, date range, unseen-route share."""
    return [
        {
            "split": name,
            "n_rows": len(df),
            "date_min": str(df[DATETIME_SRC].min()),
            "date_max": str(df[DATETIME_SRC].max()),
            "unseen_route_share": round(unseen[name], 6),
        }
        for name, df in splits.items()
    ]


def plot_split(splits: dict[str, pd.DataFrame], unseen: dict[str, float]) -> object:
    """Chronological split on a time axis plus the unseen-route share per split."""
    colours = {"train": "#4c78a8", "val": "#f58518", "test": "#e45756"}
    figure, axes = plt.subplots(2, 1, figsize=(10, 5.2), constrained_layout=True)
    axis, bars = axes
    for row, (name, part) in enumerate(splits.items()):
        start = part[DATETIME_SRC].min()
        end = part[DATETIME_SRC].max()
        hours = (end - start).total_seconds() / 3600
        axis.barh(row, hours, left=start, height=0.5, color=colours.get(name, "#888"))
        axis.text(start, row + 0.34, f"{name}: {len(part):,} rows", fontsize=9)
    axis.set_yticks(range(len(splits)), list(splits))
    axis.invert_yaxis()
    axis.set_xlabel("time")
    axis.set_title("chronological train/val/test split")
    names = list(unseen)
    bars.bar(names, [unseen[name] for name in names],
             color=[colours.get(name, "#888") for name in names])
    for index, name in enumerate(names):
        bars.text(index, unseen[name], f"{unseen[name]:.2%}", ha="center", va="bottom", fontsize=9)
    bars.set_ylabel("unseen route share")
    bars.set_title("routes unseen in train — unseen routes get a zero route frequency")
    bars.grid(True, alpha=0.3, axis="y")
    return figure


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    sample = read_named_sample(SAMPLE_NAME)
    train, val, test = time_split(sample.df, TRAIN_VAL_TEST)

    splits = {
        "train": _prepare_split(train),
        "val": _prepare_split(val),
        "test": _prepare_split(test),
    }
    splits = _apply_frequency(splits)
    for name, df in splits.items():
        out = PREPARED_DIR / f"{name}.parquet"
        df.to_parquet(out, index=False)
        print(f"wrote {out}")

    known = _known_routes(train)
    unseen = {
        name: (0.0 if name == "train" else _unseen_share(known, part))
        for name, part in (("train", train), ("val", val), ("test", test))
    }
    analysis = test.sample(n=min(ANALYSIS_SAMPLE, len(test)), random_state=SEED).assign(
        day=lambda part: part[DATETIME_SRC].dt.floor("D")
    )
    rule = load_decision_rule(STEM)
    spec = MeasurementSpec(
        bootstrap="block",
        block_column=BLOCK,
        iterations=200,
        seed=SEED,
    )
    measurement = MeasurementEstimator().estimate(
        analysis,
        lambda part: _unseen_share(known, part),
        spec,
        name=rule.metric,
        unit="share",
    )
    n = measurement.sample_size
    route_counts = train.groupby([PU_COL, DO_COL]).size()
    pairs = list(zip(analysis[PU_COL].tolist(), analysis[DO_COL].tolist()))
    counts = np.array([int(route_counts.get(pair, 0)) for pair in pairs])
    repeated, rare = counts >= 2, counts < 2
    half = test.sample(n=max(2, n // 2), random_state=SEED + 1)
    evidence = Evidence(
        baseline="every test route seen in train (unseen share of zero)",
        segments=[
            SegmentEvidence(
                segment="repeated_train_routes",
                value=_unseen_share(known, analysis[repeated]),
                sample_size=int(repeated.sum()),
            ),
            SegmentEvidence(
                segment="rare_or_unseen_train_routes",
                value=_unseen_share(known, analysis[rare]),
                sample_size=int(rare.sum()),
            ),
        ],
        sensitivity={
            f"val_unseen_route_share_n{len(val)}": unseen["val"],
            f"unseen_route_share_second_seed_n{len(half)}": _unseen_share(known, half),
        },
        falsifier="the unseen-route interval exceeds the 5% ceiling",
        provenance={
            "artifact": sample.provenance["name"],
            "artifact_sha256": sample.provenance["artifact_sha256"],
            "version": str(sample.provenance["version"]),
            "seed": str(measurement.seed),
            "analysis_sample": str(measurement.sample_size),
            "block": BLOCK,
            "resampling": measurement.resampling,
            "exclusions": "route frequency fitted on the chronological train slice only",
        },
    )
    decision = measurement.to_decision(
        rule,
        evidence=evidence,
        question="Does the chronological split leave the test slice inside the train route frequency?",
    )
    evidence_table = pd.DataFrame(
        _evidence_rows({"train": train, "val": val, "test": test}, unseen)
    )
    output = StepOutput(
        results_dir=RESULTS, stem=STEM, decision=decision, source=Path(__file__)
    )
    output.add_table("split", evidence_table)
    output.save_figure(plot_split({"train": train, "val": val, "test": test}, unseen))
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
