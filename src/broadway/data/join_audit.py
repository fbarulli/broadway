from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict

from broadway.config.schema import LookupSpec


class JoinAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lookup: str            # left dataset column, e.g. "location_id"
    lookup_path: str
    left_key: str
    right_key: str
    rows_attempted: int
    matched: int
    unmatched: int
    null_keys: int
    unmatched_rate: float


class JoinAuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    joins: list[JoinAudit]


def audit_join(df: pd.DataFrame, left_key: str, lookup: LookupSpec, lookup_df: pd.DataFrame) -> JoinAudit:
    rows_attempted = len(df)
    null_keys = int(df[left_key].isna().sum())
    matched = int(df[left_key].isin(lookup_df[lookup.key]).sum())
    unmatched = rows_attempted - matched - null_keys
    non_null_attempted = matched + unmatched
    rate = round(unmatched / non_null_attempted, 6) if non_null_attempted else 0.0
    return JoinAudit(
        lookup=left_key,
        lookup_path=lookup.path,
        left_key=left_key,
        right_key=lookup.key,
        rows_attempted=rows_attempted,
        matched=matched,
        unmatched=unmatched,
        null_keys=null_keys,
        unmatched_rate=rate,
    )
