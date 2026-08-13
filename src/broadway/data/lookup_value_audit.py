from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict

from broadway.config.schema import LookupSpec


class LookupColumnValueAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str                      # MERGED column name in the data
    null_count: int                  # MATCHED rows whose value is null
    sentinel_counts: dict[str, int]
    affected_rows: int               # null_count + sum(sentinel_counts)
    affected_rate: float             # affected_rows / matched
    affected_lookup_keys: list[str]  # distinct RIGHT keys whose value is null/sentinel


class LookupValueAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lookup: str                      # left key, e.g. "pickup_location_id"
    lookup_path: str
    matched: int
    columns: list[LookupColumnValueAudit]


class LookupValueAuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lookups: list[LookupValueAudit]


def audit_lookup_values(
    df_merged: pd.DataFrame,
    left_key: str,
    lookup: LookupSpec,
    lookup_df: pd.DataFrame,
    merged_names: dict[str, str],
    matched: int,
) -> LookupValueAudit:
    matched_mask = df_merged[merged_names[lookup.key]].notna()
    columns: list[LookupColumnValueAudit] = []
    for c in lookup_df.columns:
        if c == lookup.key:
            continue
        merged = merged_names[c]
        null_count = int((matched_mask & df_merged[merged].isna()).sum())
        sentinels = lookup.value_policies.get(c).sentinel_values if c in lookup.value_policies else []
        sentinel_counts = {
            s: int((matched_mask & (df_merged[merged] == s)).sum()) for s in sentinels
        }
        affected_rows = null_count + sum(sentinel_counts.values())
        affected_rate = round(affected_rows / matched, 6) if matched else 0.0
        bad = lookup_df[c].isna() | (lookup_df[c].isin(sentinels) if sentinels else False)
        affected_lookup_keys = sorted(lookup_df.loc[bad, lookup.key].astype(str).tolist())
        columns.append(
            LookupColumnValueAudit(
                column=merged,
                null_count=null_count,
                sentinel_counts=sentinel_counts,
                affected_rows=affected_rows,
                affected_rate=affected_rate,
                affected_lookup_keys=affected_lookup_keys,
            )
        )
    return LookupValueAudit(
        lookup=left_key,
        lookup_path=lookup.path,
        matched=matched,
        columns=columns,
    )
