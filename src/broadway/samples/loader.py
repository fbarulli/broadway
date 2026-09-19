"""Load a named sample: resolve name → immutable artifact, then validate its
provenance (integrity, definition digest, row count, schema) before returning.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pandera as pa

from broadway.contracts.pandera import pandera_dtype
from broadway.lineage.sample import load_sample
from broadway.samples.generate import (
    SAMPLES_DIR,
    _canonical_spec_sha256,
    _file_sha256,
)
from broadway.samples.models import Sample

_CHECK_BUILDERS: dict[str, Callable[..., pa.Check]] = {
    ">": pa.Check.gt,
    "<": pa.Check.lt,
    ">=": pa.Check.ge,
    "<=": pa.Check.le,
    "==": pa.Check.equal_to,
    "!=": pa.Check.not_equal_to,
}


def _build_schema(schema: dict[str, Any]) -> pa.DataFrameSchema:
    """Build a pandera schema from the ``schema`` block of a sample definition."""
    columns: dict[str, pa.Column] = {}
    for col, entry in schema.items():
        nullable = entry.get("nullable")
        checks = [
            _CHECK_BUILDERS[check["op"]](check["value"])
            for check in entry.get("checks", [])
        ]
        columns[col] = pa.Column(
            pandera_dtype(entry["dtype"]),
            nullable=True if nullable is None else nullable,
            checks=checks,
        )
    return pa.DataFrameSchema(columns)


def read_named_sample(
    name: str, version: str | None = None, samples_dir: Path | None = None
) -> Sample:
    """Resolve ``name`` to its immutable artifact and validate it before returning.

    ``version`` defaults to the version declared in the sample definition;
    an explicit version resolves to exactly that artifact (never "whatever
    file exists").
    """
    spec = load_sample(name)
    resolved_version = version or spec.version
    target_dir = samples_dir or SAMPLES_DIR
    artifact = target_dir / f"{name}@{resolved_version}.parquet"
    provenance_path = artifact.with_suffix(".json")
    if not artifact.exists():
        raise FileNotFoundError(f"sample artifact missing: {artifact}")
    if not provenance_path.exists():
        raise FileNotFoundError(f"sample provenance missing: {provenance_path}")

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if _file_sha256(artifact) != provenance["artifact_sha256"]:
        raise ValueError(f"artifact digest mismatch (corrupted?): {artifact}")
    if _canonical_spec_sha256(spec) != provenance["definition_sha256"]:
        raise ValueError(
            f"sample definition changed since v{resolved_version} was generated; "
            "bump version in config and regenerate"
        )

    df = pd.read_parquet(artifact)
    if len(df) != provenance["row_count"]:
        raise ValueError(
            f"row count mismatch for {name}@v{resolved_version}: "
            f"expected {provenance['row_count']}, got {len(df)}"
        )
    if spec.schema is not None:
        schema = _build_schema(spec.schema)
        schema.validate(df, lazy=True)  # raises SchemaErrors with the report on failure
    return Sample(df=df, spec=spec, provenance=provenance)
