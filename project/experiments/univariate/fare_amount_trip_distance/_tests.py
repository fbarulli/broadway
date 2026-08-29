"""Shared test-result persistence: one JSON per dataset, self-describing.

Each <dataset>.json (e.g. sample50k.json) carries the dataset name, row
count, source script, timestamp, the X/Y analyzed, and the transformations
that built the dataset — so every results file is identifiable and
trackable on its own. The dataset name is the filename; the experiment
folder is the context, so no tests_ prefix is needed.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from _common import DATASET_META, RESULTS


def tests_path_for(dataset: Path) -> Path:
    return RESULTS / f"{dataset.stem}.json"


def write_tests_json(
    dataset: Path,
    results: dict,
    source_script: str,
    n_rows: int,
) -> Path:
    """Write/merge test results for a dataset; returns the written path.

    Merges into an existing file so multiple test scripts can append.
    """
    out = tests_path_for(dataset)
    existing = {}
    if out.exists():
        existing = json.loads(out.read_text()).get("results", {})
    existing.update(results)

    meta = DATASET_META[dataset.stem]
    payload = {
        "dataset": dataset.name,
        "n_rows": n_rows,
        "source_script": source_script,
        "created_at": datetime.now(UTC).isoformat(),
        "x_columns": meta["x_columns"],
        "y_column": meta["y_column"],
        "transformations": meta["transformations"],
        "results": existing,
    }
    out.write_text(json.dumps(payload, indent=2))
    return out
