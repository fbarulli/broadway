"""Detect format (csv/parquet/excel) → load → optional lookup join."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import polars as pl
from pandera.errors import SchemaError

from broadway.config.schema import DatasetContract, EnvironmentConfig
from broadway.contracts.pandera import build_raw_schema
from broadway.data.join_audit import JoinAudit, audit_join
from broadway.data.lookup_value_audit import LookupValueAudit, audit_lookup_values

READERS = {
    ".csv": pd.read_csv,
    ".parquet": pd.read_parquet,
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
}

MERGE_HOW = "left"

# Single source of the collision-suffix rule: shared by the rename in
# merged_lookup_column_names and by the merge call's pandas ``suffixes``.
_LOOKUP_SUFFIX = "_lookup"


def merged_lookup_column_names(
    existing_columns: set[str], lookup_columns: Iterable[str]
) -> dict[str, str]:
    """Map each lookup column to its post-merge name; collisions get ``_lookup``.

    The single implementation of the ``_lookup`` suffix rule — consumed by the
    loader's ``merged_names`` audit dict and by the joined schema module, so
    the rule cannot drift between the loader and the schema (Decision 6).
    Shares ``_LOOKUP_SUFFIX`` with the merge call in ``load_with_audit``, so
    this rename rule and the pandas ``suffixes`` argument cannot drift apart.
    """
    return {c: (c if c not in existing_columns else c + _LOOKUP_SUFFIX) for c in lookup_columns}


def _assert_unique_merged_labels(
    dataset: DatasetContract,
    df: pd.DataFrame,
    left_key: str,
    right_key: str,
    merged_names: dict[str, str],
) -> None:
    """Reject a lookup merge that would produce duplicate column labels.

    Closes the 2026-08-23 incident class at the joined-loader boundary (FIX_4
    G2): ``merged_lookup_column_names`` only checks whether a lookup column
    name collides with an existing column — it never checks whether the
    *renamed result* (``<name>_lookup``) itself already exists. A raw frame
    carrying both ``Borough`` and ``Borough_lookup`` while the lookup
    contributes ``Borough`` therefore merges to TWO ``Borough_lookup`` labels,
    which today survives the (non-strict) schemas and only trips an accidental
    deep error inside ``audit_lookup_values``. This guard raises ``SchemaError``
    before the merge, naming every duplicated label and its provenance.

    The check is list-level (``pd.Index(...).is_unique``) on the ACTUAL
    produced label list — existing labels plus every value of ``merged_names``,
    preserving multiplicity — because a pure set-size equation collapses a
    genuine lookup-vs-lookup rename collision (e.g. existing ``{Borough}`` with
    lookup columns ``{Borough, Borough_lookup}`` maps BOTH to
    ``Borough_lookup``).
    """
    # pandas keeps the right join key in the output; when the join is on the
    # same name (left_key == right_key) the single key column the merge emits
    # is the existing one and the SSOT's "<key>_lookup" mapping is a phantom
    # the merge never produces — exclude that one name from the produced list.
    phantom = {merged_names[left_key]} if left_key == right_key else set()
    produced = list(df.columns)
    produced += [name for name in merged_names.values() if name not in phantom]
    if pd.Index(produced).is_unique:
        return

    existing_columns = set(df.columns)
    counts = Counter(produced)
    duplicated = sorted({label for label, count in counts.items() if count > 1})
    parts: list[str] = []
    for label in duplicated:
        provenance: list[str] = []
        if label in existing_columns:
            provenance.append("pre-existing column")
        renamed_from = sorted(c for c, m in merged_names.items() if m == label and c != label)
        if renamed_from:
            quoted = ", ".join(f"'{c}'" for c in renamed_from)
            provenance.append(
                f"lookup column(s) {quoted} renamed by the '{_LOOKUP_SUFFIX}' suffix rule"
            )
        if any(c == label for c, m in merged_names.items() if m == label):
            provenance.append(f"lookup's own column '{label}'")
        parts.append(f"'{label}' ({'; '.join(provenance)})")
    raise SchemaError(
        schema=build_raw_schema(dataset),
        data=df,
        message=(
            f"duplicate column label(s) after lookup merge (left key '{left_key}', "
            f"right key '{right_key}'): {', '.join(parts)} — refusing the merge; "
            "the merged frame would carry these labels multiple times"
        ),
        failure_cases=duplicated,
        check="unique_column_labels",
    )


def canonical_path(dataset: DatasetContract, environment: EnvironmentConfig) -> Path:
    return (
        Path(environment.data_dir)
        / environment.processed_subdir
        / f"{dataset.name}_canonical.parquet"
    )


def load(dataset: DatasetContract) -> pd.DataFrame:
    return load_with_audit(dataset)[0]


def load_with_audit(dataset: DatasetContract) -> tuple[pd.DataFrame, list[JoinAudit], list[LookupValueAudit]]:
    path = Path(dataset.path)
    ext = path.suffix.lower()
    if ext not in READERS:
        raise ValueError(f"unsupported format: {ext}")
    df = READERS[ext](path)
    audits: list[JoinAudit] = []
    value_audits: list[LookupValueAudit] = []
    for col, lookup in dataset.lookup_tables.items():
        right_on = lookup.key
        lookup_df = pd.read_csv(lookup.path, keep_default_na=False, na_values=lookup.na_values)
        audit = audit_join(df, col, lookup, lookup_df)
        audits.append(audit)
        merged_names = merged_lookup_column_names(set(df.columns), lookup_df.columns)
        _assert_unique_merged_labels(dataset, df, col, right_on, merged_names)
        df = df.merge(lookup_df, left_on=col, right_on=right_on, how=MERGE_HOW, suffixes=("", _LOOKUP_SUFFIX))
        value_audits.append(
            audit_lookup_values(
                df_merged=df,
                left_key=col,
                lookup=lookup,
                lookup_df=lookup_df,
                merged_names=merged_names,
                matched=audit.matched,
            )
        )
    return df, audits, value_audits


def read_sample(
    dataset: DatasetContract,
    sample: int | None = None,
    seed: int | None = None,
    columns: list[str] | None = None,
    *,
    full: bool = False,
) -> pd.DataFrame:
    """Seeded random sample of ``dataset.path`` via a lazy scan.

    Draws directly from the dataset's raw parquet — NOT the dev/live mode
    caches. Optional ``columns`` prunes to those columns only. ``sample=None``
    requires ``full=True`` (loading the full dataset is deliberate). ``seed`` is
    passed through unchanged; callers own reproducibility.
    """
    if sample is None and not full:
        raise ValueError("pass sample=<n>, or full=True to load the full dataset")
    lf = pl.scan_parquet(dataset.path)
    if columns:
        lf = lf.select(columns)
    df = lf.collect()
    if sample is not None:
        df = df.sample(n=sample, seed=seed)
    return df.to_pandas()
