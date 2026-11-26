"""01: generate the named fare_prediction_1m sample (sampling + policy applied once)."""

import json
from pathlib import Path

import pandas as pd
from _common import RESULTS, SAMPLE_NAME

from broadway.lineage.sample import load_sample
from broadway.samples import generate_sample


def _provenance(artifact: Path) -> dict:
    return json.loads(artifact.with_suffix(".json").read_text(encoding="utf-8"))


def main() -> None:
    spec = load_sample(SAMPLE_NAME)
    artifact = Path(spec.path)
    if artifact.exists():
        print(f"artifact exists: {artifact}")
    else:
        artifact = generate_sample(SAMPLE_NAME)
        print(f"artifact: {artifact}")
    provenance = _provenance(artifact)
    summary = pd.DataFrame(
        [
            {
                "artifact": str(artifact),
                "rows": provenance["row_count"],
                "columns": len(provenance["columns"]),
                "version": provenance["version"],
                "definition_sha256": provenance["definition_sha256"],
                "artifact_sha256": provenance["artifact_sha256"],
            }
        ]
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / "01_generate_sample.csv"
    summary.to_csv(output, index=False)
    print(f"summary: {output}")


if __name__ == "__main__":
    main()
