"""01: generate the named fare_prediction_1m sample (sampling + policy applied once)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import RESULTS, SAMPLE_NAME, SEED, UNIT_FMT, load_decision_rule
from scipy.stats import ks_2samp

from broadway.lineage.sample import load_sample
from broadway.reports.decision import Decision, Evidence, SegmentEvidence
from broadway.reports.step_output import StepOutput
from broadway.samples import generate_sample

COMPARED = ("fare_amount", "trip_distance", "trip_duration_minutes")
STEM = "01_generate_sample"
COMPARE_SAMPLE = 50_000
BOOTSTRAP_ITERATIONS = 200


def _provenance(artifact: Path) -> dict:
    return json.loads(artifact.with_suffix(".json").read_text(encoding="utf-8"))


def _ecdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values.dropna().to_numpy(dtype=float))
    return ordered, np.linspace(0.0, 1.0, len(ordered))


def _null_ks(source_values: pd.Series, sample_values: pd.Series) -> list[float]:
    """KS under the no-effect permutation at this sample size (the null band)."""
    pool = np.concatenate([source_values.to_numpy(), sample_values.to_numpy()])
    source_n = len(source_values)
    rng = np.random.default_rng(SEED)
    null = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        permuted = rng.permutation(pool)
        null.append(float(ks_2samp(permuted[:source_n], permuted[source_n:]).statistic))
    return null


def _drift(
    source: pd.DataFrame, frame: pd.DataFrame
) -> tuple[float, tuple[float, float], list[float]]:
    """KS distance of fare with a bootstrap 95% interval (the decision metric)."""
    column = "fare_amount"
    source_values = source[column].dropna().sample(
        n=min(COMPARE_SAMPLE, len(source)), random_state=SEED
    )
    sample_values = frame[column].dropna().sample(
        n=min(COMPARE_SAMPLE, len(frame)), random_state=SEED
    )
    point = float(ks_2samp(source_values, sample_values).statistic)
    draws = []
    for iteration in range(BOOTSTRAP_ITERATIONS):
        resample = sample_values.sample(
            n=len(sample_values), replace=True, random_state=SEED + iteration
        )
        draws.append(float(ks_2samp(source_values, resample).statistic))
    low, high = np.percentile(draws, [2.5, 97.5])
    return point, (min(float(low), point), max(float(high), point)), _null_ks(
        source_values, sample_values
    )


def _figure(source: pd.DataFrame, frame: pd.DataFrame, sampled: int) -> object:
    """ECDF overlays (the sample vs its source) plus the row funnel."""
    figure, axes = plt.subplots(1, len(COMPARED) + 1, figsize=(18, 4), constrained_layout=True)
    for axis, column in zip(axes, COMPARED):
        for values, label, colour in (
            (source[column], "source", "#9ecae1"),
            (frame[column], "sample", "#08519c"),
        ):
            xs, ys = _ecdf(values)
            axis.plot(xs, ys, label=label, color=colour, linewidth=1.4)
        axis.set_title(f"{column}: ECDF")
        axis.set_xlabel(UNIT_FMT.get(column, column))
        axis.set_ylabel("cumulative share")
        low, high = np.nanpercentile(source[column].to_numpy(dtype=float), [0.5, 99.5])
        axis.set_xlim(low, high)
        axis.legend(loc="lower right", fontsize=8)
    funnel = axes[-1]
    counts = {"source": len(source), "sampled": sampled, "filtered": len(frame)}
    funnel.bar(list(counts), list(counts.values()), color=["#9ecae1", "#6baed6", "#08519c"])
    for index, count in enumerate(counts.values()):
        funnel.text(index, count, f"{count:,}", ha="center", va="bottom", fontsize=9)
    funnel.set_title("row funnel")
    funnel.set_ylabel("rows")
    funnel.tick_params(axis="x", labelsize=8)
    return figure


def _segments(source: pd.DataFrame, frame: pd.DataFrame) -> list[SegmentEvidence]:
    """Re-measure the KS inside weekday/weekend slices (a failing slice must show)."""
    evidence = []
    for label, weekend in (("weekday", False), ("weekend", True)):
        source_mask = source["pickup_datetime"].dt.dayofweek >= 5
        frame_mask = frame["pickup_datetime"].dt.dayofweek >= 5
        if weekend:
            source_slice, frame_slice = source[source_mask], frame[frame_mask]
        else:
            source_slice, frame_slice = source[~source_mask], frame[~frame_mask]
        n = min(COMPARE_SAMPLE, len(source_slice), len(frame_slice))
        if n == 0:
            continue
        src = source_slice["fare_amount"].sample(n=n, random_state=SEED)
        smp = frame_slice["fare_amount"].sample(n=n, random_state=SEED)
        evidence.append(
            SegmentEvidence(
                segment=label,
                value=float(ks_2samp(src, smp).statistic),
                sample_size=n,
            )
        )
    return evidence


def _columns_table(frame: pd.DataFrame) -> pd.DataFrame:
    missing = frame.isna().sum()
    return pd.DataFrame(
        {
            "column": list(frame.columns),
            "dtype": [str(frame[column].dtype) for column in frame.columns],
            "missing": [int(missing[column]) for column in frame.columns],
        }
    )


def main() -> None:
    spec = load_sample(SAMPLE_NAME)
    artifact = Path(spec.path)
    if artifact.exists():
        print(f"artifact exists: {artifact}")
    else:
        artifact = generate_sample(SAMPLE_NAME)
        print(f"artifact: {artifact}")
    provenance = _provenance(artifact)
    frame = pd.read_parquet(artifact)
    source = pd.read_parquet(
        provenance["source"]["path"], columns=[*COMPARED, "pickup_datetime"]
    )
    RESULTS.mkdir(parents=True, exist_ok=True)

    value, interval, null = _drift(source, frame)
    rule = load_decision_rule(STEM)
    half = frame["fare_amount"].sample(frac=0.5, random_state=SEED)
    source_half = source["fare_amount"].sample(n=len(half), random_state=SEED)
    evidence = Evidence(
        baseline="no-effect permutation (sample labels shuffled at this n)",
        segments=_segments(source, frame),
        sensitivity={
            "ks_half_sample": float(ks_2samp(source_half, half).statistic),
            "ks_seed_7": float(
                ks_2samp(
                    source["fare_amount"].sample(n=COMPARE_SAMPLE, random_state=SEED),
                    frame["fare_amount"].sample(n=COMPARE_SAMPLE, random_state=SEED),
                ).statistic
            ),
        },
        falsifier=(
            "the KS interval no longer clears the null band once the fare-policy "
            "exclusions are relaxed"
        ),
        provenance={
            "artifact": provenance["name"],
            "artifact_sha256": provenance["artifact_sha256"],
            "source": provenance["source"]["path"],
            "seed": str(SEED),
            "draw_seed": str(SEED),
            "exclusions": "fare<=4.40 with long distance/duration/speed",
        },
    )
    decision = Decision.evaluate(
        rule=rule,
        value=value,
        interval=interval,
        sample_size=min(COMPARE_SAMPLE, len(frame)),
        null_values=null,
        evidence=evidence,
        question="Did sampling plus the fare policy preserve the source population?",
    )
    output = StepOutput(results_dir=RESULTS, stem=STEM, decision=decision)
    output.add_table("columns", _columns_table(frame))
    output.add_table(
        "funnel",
        pd.DataFrame(
            {
                "stage": ["source", "sampled", "filtered"],
                "rows": [
                    len(source),
                    provenance["size"],
                    provenance["row_count"],
                ],
            }
        ),
    )
    output.save_figure(_figure(source, frame, provenance["size"]))
    plt.close("all")
    report, plot, decision_path = output.write()
    print(f"report: {report}")
    print(f"plot: {plot}")
    print(f"decision: {decision_path} -> {decision.verdict} ({rule.metric} {value:.4f})")


if __name__ == "__main__":
    main()
