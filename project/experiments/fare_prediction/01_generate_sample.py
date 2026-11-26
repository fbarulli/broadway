"""01: generate the named fare_prediction_1m sample (sampling + policy applied once)."""

import json
from pathlib import Path

import pandas as pd
from _common import RESULTS, SAMPLE_NAME

from broadway.lineage.sample import load_sample
from broadway.samples import generate_sample


def _provenance(artifact: Path) -> dict:
    return json.loads(artifact.with_suffix(".json").read_text(encoding="utf-8"))


def _profile(artifact: Path, provenance: dict) -> str:
    """Render the sample's output as aggregates: shape, dtypes, numeric summary."""
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
        lines.append("")
        summary = numeric.describe().T[["mean", "std", "min", "50%", "max"]]
        lines += ["## numeric summary", "", summary.to_markdown(floatfmt=".4f"), ""]
    if provenance.get("derived"):
        lines += [f"derived: {', '.join(provenance['derived'])}", ""]
    return "\n".join(lines)


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
    output = RESULTS / "01_generate_sample.md"
    output.write_text(_profile(artifact, provenance), encoding="utf-8")
    print(f"profile: {output}")


if __name__ == "__main__":
    main()
