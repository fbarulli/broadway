"""Generate an immutable named-sample artifact (``<name>@v<version>.parquet``)
plus its provenance sidecar from a ``configs/sample/<name>.yaml`` definition.
"""

from __future__ import annotations

import hashlib
import json
import operator
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from broadway.config.loader import CONFIGS_DIR
from broadway.lineage.models import FilterSpec, SampleSpec
from broadway.lineage.sample import load_sample

SAMPLES_DIR = Path(CONFIGS_DIR).parent / "data" / "samples"

_OPERATORS = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def _canonical_spec_sha256(spec: SampleSpec) -> str:
    """Deterministic digest of the sample definition (config → model dump).

    ``model_dump(mode="json")`` + ``sort_keys`` yields a canonical payload, so
    the digest is stable across generate/read and independent of key order.
    """
    payload = json.dumps(spec.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _apply_filters(df: pd.DataFrame, filters: list[FilterSpec]) -> pd.DataFrame:
    """Apply FilterSpec masks; all filters AND together."""
    mask = pd.Series(True, index=df.index)
    for spec in filters:
        mask &= _OPERATORS[spec.op](df[spec.column], spec.value)
    return df[mask]


def generate_sample(name: str, samples_dir: Path | None = None) -> Path:
    """Generate the immutable artifact for ``name``; raises if it already exists.

    Reads the source declared in the sample definition, applies column
    selection, deterministic sampling, and filters once, then writes the
    artifact + provenance sidecar.
    """
    spec = load_sample(name)
    if spec.source is None or spec.seed is None or spec.size is None:
        raise ValueError(
            f"sample config '{name}' must declare source, seed, and size to generate"
        )
    target_dir = samples_dir or SAMPLES_DIR
    artifact = target_dir / f"{name}@{spec.version}.parquet"
    if artifact.exists():
        raise FileExistsError(
            "immutable artifact exists; bump `version` in the sample config to "
            f"regenerate: {artifact}"
        )

    source_path = (Path(CONFIGS_DIR).parent / spec.source.path).resolve()
    df = pd.read_parquet(source_path)
    if spec.columns is not None:
        df = df[spec.columns]
    df = df.sample(n=spec.size, random_state=spec.seed)
    if spec.filters:
        df = _apply_filters(df, spec.filters)
    df = df.reset_index(drop=True)

    target_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(artifact, index=False)
    provenance = {
        "name": name,
        "version": spec.version,
        "source": {"name": spec.source.name, "path": spec.source.path},
        "seed": spec.seed,
        "size": spec.size,
        "row_count": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "filters": [f.model_dump() for f in (spec.filters or [])],
        "definition_sha256": _canonical_spec_sha256(spec),
        "artifact_sha256": _file_sha256(artifact),
        "created_at": datetime.now(UTC).isoformat(),
    }
    artifact.with_suffix(".json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return artifact
