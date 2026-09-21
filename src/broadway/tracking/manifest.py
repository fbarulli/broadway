"""Data manifest writer — sha256 + row counts + timestamps + schema version.

Reads the canonical parquet hash, row counts from ``TransformAudit``, timestamp
bounds from ``DatasetProfile`` (falling back to the in-memory frame when no
profile exists), and the schema version from the experiment's declared
``data_source.schema_contract``. Writes ``artifacts/tracking/data_manifest.json``.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from broadway.config.schema import DatasetContract
from broadway.discover.profile import DatasetProfile
from broadway.lineage.models import TransformAudit

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "data_manifest.json"
_PROFILE_RELATIVE = Path("discover") / "profile.json"
_CHUNK_SIZE = 65536


class DataManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str
    canonical_path: str
    sha256: str
    rows_in: int
    rows_out: int
    timestamp_min: str | None
    timestamp_max: str | None
    schema_contract: str


def tracking_dir(root: Path | None = None) -> Path:
    """Resolve the tracking bundle dir; explicit ``root`` wins over the env."""
    if root is not None:
        return root
    return Path(os.getenv("BROADWAY_TRACKING_DIR", "artifacts/tracking"))


def manifest_path(root: Path | None = None) -> Path:
    """Resolve the ``data_manifest.json`` path under the tracking dir."""
    return tracking_dir(root) / MANIFEST_FILENAME


def default_profile_path(artifacts_dir: Path | None = None) -> Path:
    """Resolve ``artifacts/discover/profile.json``; explicit dir wins over env."""
    base = artifacts_dir
    if base is None:
        base = Path(os.getenv("BROADWAY_ARTIFACTS_DIR", "artifacts"))
    return base / _PROFILE_RELATIVE


def sha256_of_file(path: Path) -> str:
    """Hash a file chunk-wise; missing paths fail loud via ``open``."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamps_from_profile(profile: DatasetProfile) -> tuple[str | None, str | None]:
    """Aggregate min/max over every column's datetime bounds (data-agnostic)."""
    starts = sorted(c.datetime_min for c in profile.columns.values() if c.datetime_min)
    ends = sorted(c.datetime_max for c in profile.columns.values() if c.datetime_max)
    low = starts[0] if starts else None
    high = ends[-1] if ends else None
    return low, high


def timestamps_from_frame(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Min/max over datetime64 columns of a frame; (None, None) when absent."""
    bounds: list[tuple[str, str]] = []
    for name in df.columns:
        series = df[name]
        if not pd.api.types.is_datetime64_any_dtype(series):
            continue
        trimmed = series.dropna()
        if trimmed.empty:
            continue
        low = pd.Timestamp(trimmed.min()).isoformat()
        high = pd.Timestamp(trimmed.max()).isoformat()
        bounds.append((low, high))
    if not bounds:
        return None, None
    lows, highs = zip(*bounds)
    return min(lows), max(highs)


def load_profile(profile_file: Path | None = None) -> DatasetProfile | None:
    """Load the discover profile; None (with a log line) when not yet written."""
    resolved = profile_file if profile_file is not None else default_profile_path()
    if not resolved.exists():
        logger.info(f"manifest: no profile at {resolved} — timestamps fall back to frame")
        return None
    return DatasetProfile.model_validate_json(resolved.read_text(encoding="utf-8"))


def build_manifest(
    *,
    dataset: DatasetContract,
    audit: TransformAudit,
    canonical_path: Path,
    schema_contract: str,
    timestamp_min: str | None,
    timestamp_max: str | None,
) -> DataManifest:
    """Assemble a manifest; hashing a missing parquet fails loud."""
    return DataManifest(
        dataset=dataset.name,
        canonical_path=str(canonical_path),
        sha256=sha256_of_file(canonical_path),
        rows_in=audit.rows_in,
        rows_out=audit.rows_out,
        timestamp_min=timestamp_min,
        timestamp_max=timestamp_max,
        schema_contract=schema_contract,
    )


def write_manifest(manifest: DataManifest, root: Path | None = None) -> Path:
    """Persist a manifest to the tracking dir; returns the written path."""
    out = manifest_path(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"manifest: wrote {manifest.dataset} ({manifest.rows_out} rows) to {out}")
    return out


def write_etl_manifest(
    *,
    dataset: DatasetContract,
    audit: TransformAudit,
    canonical_path: Path,
    schema_contract: str,
    df: pd.DataFrame | None = None,
    profile: DatasetProfile | None = None,
    tracking_root: Path | None = None,
    profile_file: Path | None = None,
) -> Path:
    """Build and write the manifest for one etl run; returns the written path."""
    resolved = profile if profile is not None else load_profile(profile_file)
    if resolved is not None:
        timestamp_min, timestamp_max = timestamps_from_profile(resolved)
    elif df is not None:
        timestamp_min, timestamp_max = timestamps_from_frame(df)
    else:
        timestamp_min, timestamp_max = None, None
    manifest = build_manifest(
        dataset=dataset,
        audit=audit,
        canonical_path=canonical_path,
        schema_contract=schema_contract,
        timestamp_min=timestamp_min,
        timestamp_max=timestamp_max,
    )
    return write_manifest(manifest, tracking_root)
