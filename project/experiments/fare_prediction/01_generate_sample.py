"""01: generate the named fare_prediction_1m sample (sampling + policy applied once)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import RESULTS, SAMPLE_NAME, UNIT_FMT

from broadway.lineage.sample import load_sample
from broadway.samples import generate_sample

COMPARED = ("fare_amount", "trip_distance", "trip_duration_minutes")


def _provenance(artifact: Path) -> dict:
    return json.loads(artifact.with_suffix(".json").read_text(encoding="utf-8"))


def _ecdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values.dropna().to_numpy(dtype=float))
    return ordered, np.linspace(0.0, 1.0, len(ordered))


def _profile_markdown(artifact: Path, provenance: dict) -> str:
    frame = pd.read_parquet(artifact)
    lines = [
        f"# 01 — {provenance['name']} ({provenance['version']})",
        "",
        f"- artifact: `{artifact}`",
        f"- rows: {len(frame):,} · columns: {len(frame.columns)}",
        f"- source: `{provenance['source']['path']}` (seed {provenance['seed']}, size {provenance['size']})",
        f"- definition_sha256: `{provenance['definition_sha256']}`",
        f"- artifact_sha256: `{provenance['artifact_sha256']}`",
        "",
        "## columns",
        "",
        "| column | dtype | missing | missing % |",
        "| --- | --- | ---: | ---: |",
    ]
    missing = frame.isna().sum()
    for column in frame.columns:
        count = int(missing[column])
        share = 100.0 * count / len(frame) if len(frame) else 0.0
        lines.append(f"| {column} | {frame[column].dtype} | {count:,} | {share:.2f}% |")
    numeric = frame.select_dtypes("number")
    if not numeric.empty:
        lines += [""]
        lines += ["## numeric summary", ""]
        summary = numeric.describe().T[["mean", "std", "min", "50%", "max"]]
        lines += [summary.to_markdown(floatfmt=".4f"), ""]
    return "\n".join(lines)


def _plot_sample_vs_source(
    frame: pd.DataFrame, provenance: dict, out_path: Path
) -> None:
    """Plot what 01 actually did: the sample against its source, plus the row funnel.

    The step's question is "did sampling + policy keep the population shape?" — a
    sample-only histogram cannot answer it, so the panels overlay the source's
    empirical CDF on the artifact's and show the rows each stage removed.
    """
    source = pd.read_parquet(provenance["source"]["path"], columns=list(COMPARED))
    source = source.sample(n=min(len(source), 200_000), random_state=provenance["seed"])
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
    stages = ["source", "sampled", "filtered"]
    counts = [len(source), provenance["size"], provenance["row_count"]]
    funnel.bar(stages, counts, color=["#9ecae1", "#6baed6", "#08519c"])
    for index, count in enumerate(counts):
        funnel.text(index, count, f"{count:,}", ha="center", va="bottom", fontsize=9)
    funnel.set_title("row funnel")
    funnel.set_ylabel("rows")
    funnel.tick_params(axis="x", labelsize=8)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def main() -> None:
    spec = load_sample(SAMPLE_NAME)
    artifact = Path(spec.path)
    if artifact.exists():
        print(f"artifact exists: {artifact}")
    else:
        artifact = generate_sample(SAMPLE_NAME)
        print(f"artifact: {artifact}")
    provenance = _provenance(artifact)
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = RESULTS / "01_generate_sample.md"
    report.write_text(_profile_markdown(artifact, provenance), encoding="utf-8")
    print(f"report: {report}")
    figure = RESULTS / "01_generate_sample.png"
    _plot_sample_vs_source(pd.read_parquet(artifact), provenance, figure)
    print(f"plot: {figure}")


if __name__ == "__main__":
    main()
