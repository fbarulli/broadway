"""01: generate the named fare_prediction_1m sample (sampling + policy applied once)."""

import json

from broadway.samples import generate_sample

from _common import SAMPLE_NAME


def main() -> None:
    artifact = generate_sample(SAMPLE_NAME)
    provenance = json.loads(artifact.with_suffix(".json").read_text(encoding="utf-8"))
    print(f"artifact: {artifact}")
    print(f"rows: {provenance['row_count']}")


if __name__ == "__main__":
    main()
