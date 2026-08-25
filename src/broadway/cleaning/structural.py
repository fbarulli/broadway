"""Primitives for structural cleaning: datetime parsing and missing encoding.

Also the single owner of the coercion-evidence record (FIX_4 G1 / Option E):
``parse_numeric``'s astype-back-to-declared branch records every coercion event
into a caller-supplied collector instead of silently restoring the dtype, and
etl surfaces the collected records with the run's evidence (same channel family
as JoinAudit/LookupValueAudit). Fractional values bound for an integer dtype
refuse the repair entirely (FX-A05): they are recorded as a ``ParseFailure``
through the same channel as unparseable values and the astype-back is skipped,
leaving the column float64 so the downstream dtype gate fails loud.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict

from broadway.cleaning.models import ParseFailure


class CoercionRecord(BaseModel):
    """Evidence of one astype-back-to-declared event inside ``parse_numeric``.

    The record means "a coercion happened" (the arriving read-time dtype
    differed from the declared contract dtype and the column was cast back to
    the declared dtype) — NOT "something drifted". Pandas routinely hands
    float64 for perfectly healthy int columns during CSV inference, and
    ``parse_numeric`` casts those back silently today; FIX_4 makes that repair
    observable instead of silent. Distinguishing drift from benign parsing is
    an ingest-boundary concern (contract C), not this record.
    """

    model_config = ConfigDict(extra="forbid")

    column: str
    declared_dtype: str      # declared contract dtype the column is cast back to
    arriving_dtype: str      # read-time dtype before to_numeric/astype
    rows_affected: int       # rows rewritten by the astype-back


class CoercionAuditReport(BaseModel):
    """Run-level container mirroring ``JoinAuditReport``/``LookupValueAuditReport``."""

    model_config = ConfigDict(extra="forbid")

    coercions: list[CoercionRecord]


def parse_datetime(series: pd.Series, column: str) -> tuple[pd.Series, ParseFailure | None]:
    coerced = pd.to_datetime(series, errors="coerce")
    failed = series.notna() & coerced.isna()
    failure = None
    if failed.any():
        examples = [str(v) for v in series[failed].dropna().unique()[:5]]
        failure = ParseFailure(
            column=column,
            count=int(failed.sum()),
            examples=examples,
            target_dtype="datetime",
        )
    return coerced, failure


def _fractional_refusal(
    series: pd.Series,
    coerced: pd.Series,
    column: str,
    target_dtype: str,
) -> ParseFailure | None:
    """Failure for non-integral values an integer astype would silently truncate."""
    fractional = coerced.notna() & (coerced % 1 != 0)
    if not fractional.any():
        return None
    examples = [str(v) for v in series[fractional].dropna().unique()[:5]]
    return ParseFailure(
        column=column,
        count=int(fractional.sum()),
        examples=examples,
        target_dtype=target_dtype,
    )


def parse_numeric(
    series: pd.Series,
    column: str,
    target_dtype: str,
    coercions: list[CoercionRecord] | None = None,
) -> tuple[pd.Series, ParseFailure | None]:
    coerced = pd.to_numeric(series, errors="coerce")
    failed = series.notna() & coerced.isna()
    failure = None
    if failed.any():
        examples = [str(v) for v in series[failed].dropna().unique()[:5]]
        failure = ParseFailure(
            column=column,
            count=int(failed.sum()),
            examples=examples,
            target_dtype=target_dtype,
        )
    if target_dtype in ("int8", "int16", "int32", "int64") and not coerced.isna().any():
        arriving_dtype = str(series.dtype)
        if arriving_dtype != target_dtype:
            refusal = _fractional_refusal(series, coerced, column, target_dtype)
            if refusal is not None:
                return coerced, refusal
            coerced = coerced.astype(target_dtype)
            if coercions is not None:
                coercions.append(
                    CoercionRecord(
                        column=column,
                        declared_dtype=target_dtype,
                        arriving_dtype=arriving_dtype,
                        rows_affected=len(coerced),
                    )
                )
    return coerced, failure


def standardize_missing(
    series: pd.Series, column: str, encodings: list[str]
) -> tuple[pd.Series, list[str]]:
    observed: list[str] = []
    for value in series.unique():
        if value in encodings and value not in observed:
            observed.append(value)
    cleaned = series.replace(encodings, None)
    return cleaned, observed
