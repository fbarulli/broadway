"""Drift baseline stats writer + data-card renderer (STATE-20260921-002).

Consumes the training-truth snapshot — the canonical parquet frame (or an
in-memory frame already read from it) plus :class:`DatasetProfile` and
:class:`TransformAudit` evidence — and produces the two README-promised
``artifacts/tracking/`` files this lane owns:

- ``baseline_stats.json`` — per-feature moments plus distribution bins for
  future PSI/CSI comparison. The bins stored here are the *expected*
  (training-time) distribution; a future production frame recomputes the
  same binning and compares against these counts.
- ``data_card.md`` — date range, biases, exclusions, and missingness rates.

Data-agnostic: no dataset or column names appear below. All numeric tuning
(``n_bins``, ``top_k``, ``smoothing``, ``missing_flag_rate``) arrives via
explicit params (:class:`BaselineOptions` / :class:`DataCardOptions`), never
as inline literals at the decision site.

Deliberate non-overlap with STATE-20260921-001 (manifest lane): nothing here
computes hashes, row-count manifests, schema versions, or timestamps of the
parquet file itself — those belong to the manifest writer. This module only
reads feature distributions and rendersMarkdown from already-built profile
and audit models.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from broadway.discover.profile import DatasetProfile
from broadway.lineage.models import TransformAudit

FeatureKind = Literal["numeric", "categorical", "datetime", "other"]


class BaselineOptions(BaseModel):
    """Tunable knobs for baseline construction. No dataset knowledge here."""

    model_config = ConfigDict(extra="forbid")

    n_bins: int = Field(default=10, ge=1)
    top_k: int = Field(default=10, ge=1)
    smoothing: float = Field(default=0.0, ge=0.0)


class DataCardOptions(BaseModel):
    """Tunable knobs for data-card rendering. No dataset knowledge here."""

    model_config = ConfigDict(extra="forbid")

    biases: list[str] = Field(default_factory=list)
    missing_flag_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    generated_at: str | None = None


class FeatureBaseline(BaseModel):
    """Training-time distribution snapshot for one feature."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: FeatureKind
    dtype: str
    count: int
    missing_count: int
    missing_rate: float
    mean: float | None = None
    std: float | None = None
    min_value: str | None = None
    max_value: str | None = None
    bin_edges: list[float] = Field(default_factory=list)
    bin_counts: list[int] = Field(default_factory=list)
    bin_labels: list[str] = Field(default_factory=list)


class BaselineStats(BaseModel):
    """Per-feature training truth for future PSI/CSI drift comparison."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    row_count: int
    n_bins: int
    top_k: int
    smoothing: float
    features: dict[str, FeatureBaseline]


def _classify(series: pd.Series) -> FeatureKind:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def _stringify(value: object) -> str | None:
    return None if value is None else str(value)


def _numeric_baseline(
    name: str, series: pd.Series, *, n_bins: int, row_count: int
) -> FeatureBaseline:
    non_null = pd.to_numeric(series, errors="coerce").dropna()
    missing = int(series.isna().sum())
    base = FeatureBaseline(
        name=name,
        kind="numeric",
        dtype=str(series.dtype),
        count=int(non_null.size),
        missing_count=missing,
        missing_rate=(missing / row_count) if row_count else 0.0,
    )
    if non_null.empty:
        return base
    base.mean = float(non_null.mean())
    base.std = float(non_null.std(ddof=1)) if non_null.size > 1 else 0.0
    base.min_value = _stringify(non_null.min())
    base.max_value = _stringify(non_null.max())
    lo, hi = float(non_null.min()), float(non_null.max())
    if lo == hi:
        base.bin_edges = [lo, hi]
        base.bin_counts = [int(non_null.size)]
        return base
    import numpy as np

    counts, edges = np.histogram(non_null.to_numpy(dtype=float), bins=n_bins)
    base.bin_edges = [float(e) for e in edges]
    base.bin_counts = [int(c) for c in counts]
    return base


def _categorical_baseline(
    name: str, series: pd.Series, *, top_k: int, row_count: int
) -> FeatureBaseline:
    missing = int(series.isna().sum())
    counts = series.dropna().astype(str).value_counts().head(top_k)
    return FeatureBaseline(
        name=name,
        kind="categorical",
        dtype=str(series.dtype),
        count=int(series.dropna().shape[0]),
        missing_count=missing,
        missing_rate=(missing / row_count) if row_count else 0.0,
        bin_labels=[str(label) for label in counts.index],
        bin_counts=[int(c) for c in counts.to_numpy()],
    )


def _datetime_baseline(name: str, series: pd.Series, *, row_count: int) -> FeatureBaseline:
    missing = int(series.isna().sum())
    non_null = series.dropna()
    base = FeatureBaseline(
        name=name,
        kind="datetime",
        dtype=str(series.dtype),
        count=int(non_null.size),
        missing_count=missing,
        missing_rate=(missing / row_count) if row_count else 0.0,
    )
    if not non_null.empty:
        base.min_value = str(non_null.min())
        base.max_value = str(non_null.max())
    return base


def build_feature_baseline(
    name: str, series: pd.Series, *, options: BaselineOptions, row_count: int
) -> FeatureBaseline:
    """Build the baseline snapshot for one column (data-agnostic)."""
    kind = _classify(series)
    if kind == "numeric":
        return _numeric_baseline(name, series, n_bins=options.n_bins, row_count=row_count)
    if kind == "datetime":
        return _datetime_baseline(name, series, row_count=row_count)
    if kind == "categorical":
        return _categorical_baseline(name, series, top_k=options.top_k, row_count=row_count)
    missing = int(series.isna().sum())
    return FeatureBaseline(
        name=name,
        kind="other",
        dtype=str(series.dtype),
        count=int(series.dropna().shape[0]),
        missing_count=missing,
        missing_rate=(missing / row_count) if row_count else 0.0,
    )


def build_baseline_stats(
    df: pd.DataFrame, *, dataset: str, options: BaselineOptions | None = None
) -> BaselineStats:
    """Compute per-feature moments + PSI/CSI bins over the training frame."""
    opts = options or BaselineOptions()
    row_count = len(df)
    features = {
        str(col): build_feature_baseline(str(col), df[col], options=opts, row_count=row_count)
        for col in df.columns
    }
    return BaselineStats(
        dataset=dataset,
        row_count=row_count,
        n_bins=opts.n_bins,
        top_k=opts.top_k,
        smoothing=opts.smoothing,
        features=features,
    )


def build_baseline_stats_from_parquet(
    parquet_path: str | Path, *, dataset: str, options: BaselineOptions | None = None
) -> BaselineStats:
    """Read the canonical parquet once and build its baseline stats."""
    df = pd.read_parquet(parquet_path)
    return build_baseline_stats(df, dataset=dataset, options=options)


def write_baseline_stats(stats: BaselineStats, path: str | Path) -> Path:
    """Persist ``baseline_stats.json``; returns the written path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(stats.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return out


def _date_range(profile: DatasetProfile | None) -> str:
    if profile is None:
        return "unknown (no DatasetProfile provided)"
    mins = [c.datetime_min for c in profile.columns.values() if c.datetime_min]
    maxs = [c.datetime_max for c in profile.columns.values() if c.datetime_max]
    if not mins or not maxs:
        return "unknown (profile carries no datetime bounds)"
    return f"{min(mins)} .. {max(maxs)}"


def _missingness_lines(profile: DatasetProfile | None, *, missing_flag_rate: float) -> list[str]:
    if profile is None:
        return ["unknown (no DatasetProfile provided)"]
    lines = []
    for name, col in profile.columns.items():
        rate = (col.null_count / profile.row_count) if profile.row_count else 0.0
        flag = " (above missing_flag_rate)" if rate > missing_flag_rate else ""
        lines.append(f"- {name}: {col.null_count}/{profile.row_count} ({rate:.4f}){flag}")
    return lines or ["no columns"]


def _exclusion_lines(audits: Sequence[TransformAudit]) -> list[str]:
    if not audits:
        return ["none recorded"]
    lines: list[str] = []
    for audit in audits:
        lines.append(
            f"- rows_in={audit.rows_in} rows_out={audit.rows_out} "
            f"dropped_total={audit.rows_dropped_total} "
            f"dropped_unexplained={audit.rows_dropped_unexplained}"
        )
        lines.extend(f"  - reason: {reason}" for reason in audit.reasons)
        if audit.columns_removed:
            lines.append(f"  - columns_removed: {', '.join(audit.columns_removed)}")
    return lines


def _baseline_lines(baseline: BaselineStats | None) -> list[str]:
    if baseline is None:
        return ["not computed"]
    lines = [
        (
            f"dataset={baseline.dataset} rows={baseline.row_count} "
            f"n_bins={baseline.n_bins} top_k={baseline.top_k}"
        )
    ]
    for name, feat in baseline.features.items():
        if feat.kind == "numeric":
            lines.append(
                f"- {name} (numeric, n={feat.count}): mean={feat.mean} std={feat.std} "
                f"bins={len(feat.bin_counts)}"
            )
        elif feat.kind == "categorical":
            lines.append(f"- {name} (categorical, n={feat.count}): top_bins={len(feat.bin_counts)}")
        else:
            lines.append(f"- {name} ({feat.kind}, n={feat.count})")
    return lines


def render_data_card(
    *,
    dataset: str,
    profile: DatasetProfile | None = None,
    audits: Sequence[TransformAudit] = (),
    baseline: BaselineStats | None = None,
    options: DataCardOptions | None = None,
) -> str:
    """Render ``data_card.md`` from profile + audit evidence.

    Biases are caller-supplied (this module never invents them); when empty
    the card records that no biases were declared.
    """
    opts = options or DataCardOptions()
    biases = list(opts.biases) or ["not declared"]
    generated = opts.generated_at or "unrecorded"
    lines = [
        f"# Data Card — {dataset}",
        "",
        f"Generated at: {generated}",
        "",
        "## Date range",
        "",
        _date_range(profile),
        "",
        "## Biases",
        "",
        *(f"- {bias}" for bias in biases),
        "",
        "## Exclusions",
        "",
        *_exclusion_lines(audits),
        "",
        "## Missingness",
        "",
        *_missingness_lines(profile, missing_flag_rate=opts.missing_flag_rate),
        "",
        "## Training baseline",
        "",
        *_baseline_lines(baseline),
        "",
    ]
    return "\n".join(lines)


def write_data_card(markdown: str, path: str | Path) -> Path:
    """Persist ``data_card.md``; returns the written path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown if markdown.endswith("\n") else markdown + "\n", encoding="utf-8")
    return out


def load_baseline_stats(path: str | Path) -> BaselineStats:
    """Reload a persisted ``baseline_stats.json`` (drift-compare entry point)."""
    return BaselineStats.model_validate_json(Path(path).read_text(encoding="utf-8"))


def dump_baseline_counts(path: str | Path) -> dict[str, list[int]]:
    """Lightweight accessor: feature name -> expected bin counts.

    Stored separately from :func:`load_baseline_stats` so future PSI/CSI code
    can fetch just the expected distribution without importing the models.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    features = raw.get("features", {})
    return {name: list(feat.get("bin_counts", [])) for name, feat in features.items()}
