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
from broadway.features.builders import BUILDERS
from broadway.lineage.models import DerivedSpec, FilterSpec, SampleSpec
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


def _apply_derived(df: pd.DataFrame, derived: list[DerivedSpec]) -> pd.DataFrame:
    """Compute shared-registry derived columns; unknown formulas raise.

    Formulas resolve against the shared transform registry
    (``broadway.features.builders.BUILDERS``) — the same implementation
    functions the feature pipeline uses. Dataset configs declare which
    columns feed each formula role via ``DerivedSpec.columns``.
    """
    for spec in derived:
        func = BUILDERS.get(spec.formula)
        if func is None:
            raise ValueError(
                f"unknown derived formula '{spec.formula}' "
                f"(registry: {sorted(BUILDERS)})"
            )
        df = df.assign(**{spec.name: func(df, None, columns=spec.columns)})
    return df


def _apply_exclude_any(
    df: pd.DataFrame, groups: list[list[FilterSpec]]
) -> pd.DataFrame:
    """Drop rows matching ANY group; each group's conditions AND together."""
    drop = pd.Series(False, index=df.index)
    for group in groups:
        mask = pd.Series(True, index=df.index)
        for spec in group:
            if spec.column not in df.columns:
                raise ValueError(
                    f"exclude_any condition references missing column "
                    f"'{spec.column}' (have: {sorted(df.columns)})"
                )
            mask &= _OPERATORS[spec.op](df[spec.column], spec.value)
        drop |= mask
    return df[~drop]


def generate_sample(name: str, samples_dir: Path | None = None) -> Path:
    """Generate the immutable artifact for ``name``; raises if it already exists.

    Reads the source declared in the sample definition, applies column
    selection, deterministic sampling, derived columns, filters, and
    OR-of-AND exclusions once, then writes the artifact + provenance sidecar.
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
    if spec.derived:
        df = _apply_derived(df, spec.derived)
    if spec.filters:
        df = _apply_filters(df, spec.filters)
    if spec.exclude_any:
        df = _apply_exclude_any(df, spec.exclude_any)
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
        "derived": [d.name for d in (spec.derived or [])],
        "exclude_any": [
            [f.model_dump() for f in group] for group in (spec.exclude_any or [])
        ],
        "definition_sha256": _canonical_spec_sha256(spec),
        "artifact_sha256": _file_sha256(artifact),
        # DOCUMENTED SILENCE (determinism ledger d): wall-clock bytes — this
        # timestamp makes provenance JSON run-unique by design; pinned only
        # once a freeze flag (pinned-timestamp config/env) exists.
        "created_at": datetime.now(UTC).isoformat(),
    }
    artifact.with_suffix(".json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return artifact
