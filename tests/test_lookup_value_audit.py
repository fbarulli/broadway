from __future__ import annotations

import pandas as pd
import pytest

from broadway.config.schema import LookupSpec, LookupValuePolicy
from broadway.data.lookup_value_audit import audit_lookup_values


def _lookup_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "LocationID": [1, 2, 3, 4],
            "Borough": ["Manhattan", "Unknown", None, "Brooklyn"],
            "Zone": ["a", "b", "c", "d"],
        }
    )


def _spec(sentinels: list[str] | None = None) -> LookupSpec:
    value_policies = {}
    if sentinels is not None:
        value_policies = {"Borough": LookupValuePolicy(sentinel_values=sentinels)}
    return LookupSpec(path="lookup.csv", key="LocationID", value_policies=value_policies)


def _audit(sentinels: list[str] | None = None):
    df = pd.DataFrame({"pickup_location_id": [1, 2, 3, 4, 99]})
    lookup_df = _lookup_df()
    merged_names = {c: c for c in lookup_df.columns}
    merged = df.merge(lookup_df, left_on="pickup_location_id", right_on="LocationID", how="left")
    audit = audit_lookup_values(
        df_merged=merged,
        left_key="pickup_location_id",
        lookup=_spec(sentinels),
        lookup_df=lookup_df,
        merged_names=merged_names,
        matched=4,
    )
    return audit


def _column(audit, name: str):
    return next(c for c in audit.columns if c.column == name)


def test_unmatched_rows_excluded_from_null_count() -> None:
    audit = _audit()
    borough = _column(audit, "Borough")
    assert borough.null_count == 1


def test_null_count_counts_only_matched_null_values() -> None:
    audit = _audit()
    borough = _column(audit, "Borough")
    assert borough.null_count == 1
    zone = _column(audit, "Zone")
    assert zone.null_count == 0


def test_sentinel_counts_only_configured_sentinels() -> None:
    audit = _audit(sentinels=["Unknown"])
    borough = _column(audit, "Borough")
    assert borough.sentinel_counts == {"Unknown": 1}


def test_affected_rate_is_affected_rows_over_matched() -> None:
    audit = _audit(sentinels=["Unknown"])
    borough = _column(audit, "Borough")
    assert borough.affected_rows == 2
    assert borough.affected_rate == pytest.approx(2 / 4)


def test_affected_lookup_keys_come_from_right_key() -> None:
    audit = _audit(sentinels=["Unknown"])
    borough = _column(audit, "Borough")
    assert borough.affected_lookup_keys == ["2", "3"]


def test_no_sentinels_yields_empty_counts() -> None:
    audit = _audit()
    borough = _column(audit, "Borough")
    assert borough.sentinel_counts == {}
    assert borough.affected_lookup_keys == ["3"]


def test_every_non_key_column_gets_an_entry() -> None:
    audit = _audit()
    assert {c.column for c in audit.columns} == {"Borough", "Zone"}
