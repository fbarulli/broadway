from __future__ import annotations

from pathlib import Path

import pandas as pd

from broadway.config.schema import (
    ColumnRole,
    ColumnSchema,
    DatasetContract,
    LookupSpec,
    LookupValuePolicy,
    TaskType,
)
from broadway.data.loader import load_with_audit
from broadway.data.lookup_value_audit import audit_lookup_values


def _lookup_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "LocationID": [1, 2, 3, 4],
            "district": ["North", "Unknown", None, "South"],
            "Zone": ["a", "b", "c", "d"],
        }
    )


def _spec(sentinels: list[str] | None = None) -> LookupSpec:
    value_policies = {}
    if sentinels is not None:
        value_policies = {"district": LookupValuePolicy(sentinel_values=sentinels)}
    return LookupSpec(path="lookup.csv", key="LocationID", value_policies=value_policies)


def _audit(sentinels: list[str] | None = None):
    df = pd.DataFrame({"location_id": [1, 2, 3, 4, 99]})
    lookup_df = _lookup_df()
    merged_names = {c: c for c in lookup_df.columns}
    merged = df.merge(lookup_df, left_on="location_id", right_on="LocationID", how="left")
    audit = audit_lookup_values(
        df_merged=merged,
        left_key="location_id",
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
    district = _column(audit, "district")

    # Fixture hand-count: matched rows are location_id 1-4 and only id 3
    # carries a null district. Row 99 has no lookup hit, so its post-merge
    # NaN must NOT register -- a bare isna().sum() over the merged frame
    # would report 2.
    assert district.null_count == 1


def test_null_count_counts_only_matched_null_values() -> None:
    audit = _audit()
    district = _column(audit, "district")
    zone = _column(audit, "Zone")

    # Matched districts are North, Unknown, None, South -> exactly one
    # matched null; every matched Zone value is populated.
    assert district.null_count == 1
    assert zone.null_count == 0


def test_sentinel_counts_only_configured_sentinels() -> None:
    audit = _audit(sentinels=["Unknown", "NeverSeen"])
    district = _column(audit, "district")
    zone = _column(audit, "Zone")

    # Matched districts: North, Unknown, None(null), South. "Unknown" hits
    # exactly one row; configured-but-absent "NeverSeen" stays at zero;
    # ordinary values must never grow sentinel buckets. Zone has no policy,
    # so its bucket set stays empty.
    assert district.sentinel_counts == {"Unknown": 1, "NeverSeen": 0}
    assert zone.sentinel_counts == {}


def test_affected_rate_is_affected_rows_over_matched() -> None:
    audit = _audit(sentinels=["Unknown"])
    district = _column(audit, "district")

    # district: 1 matched null (id 3) + 1 sentinel hit (id 2) = 2 affected
    # rows out of matched=4 -> rate 0.5. Dividing by all 5 merged rows
    # would give 0.4, so the matched-only denominator is pinned too.
    assert district.affected_rows == 2
    assert district.affected_rate == 0.5


def test_affected_lookup_keys_come_from_right_key() -> None:
    lookup_df = pd.DataFrame(
        {"Code": ["R1", "R2", "R3"], "district": ["North", "Bad", None]}
    )
    merged = pd.DataFrame(
        {
            "row_id": [10.0, 20.0, 30.0, 99.0],
            "district": ["North", "Bad", None, None],
        }
    )
    spec = LookupSpec(
        path="lookup.csv",
        key="Code",
        value_policies={"district": LookupValuePolicy(sentinel_values=["Bad"])},
    )
    audit = audit_lookup_values(
        df_merged=merged,
        left_key="row_id",
        lookup=spec,
        lookup_df=lookup_df,
        merged_names={"Code": "row_id", "district": "district"},
        matched=3,
    )
    district = _column(audit, "district")

    # Affected right-side rows: R2 (sentinel) and R3 (null). Left ids run
    # 10..30, so echoing LEFT keys would yield ["20", "30"] and fail --
    # the disjoint key spaces make the provenance observable.
    assert district.affected_lookup_keys == ["R2", "R3"]


def test_no_sentinels_yields_empty_counts() -> None:
    audit = _audit()
    district = _column(audit, "district")
    zone = _column(audit, "Zone")

    # Without a policy only the matched null (id 3) registers: empty
    # sentinel bucket set, affected_rows == 1 (rate 0.25 over matched=4).
    # The null's right key still appears -- null coverage does not depend
    # on any sentinel policy; the unconfigured Zone column stays at zero.
    assert district.sentinel_counts == {}
    assert district.affected_rows == 1
    assert district.affected_lookup_keys == ["3"]
    assert zone.affected_rows == 0
    assert zone.affected_lookup_keys == []


def test_every_non_key_column_gets_an_entry() -> None:
    audit = _audit()
    assert {c.column for c in audit.columns} == {"district", "Zone"}


def _na_dataset(tmp_path: Path, na_values: list[str]) -> tuple[DatasetContract, Path]:
    lookup_csv = tmp_path / "lookup.csv"
    lookup_csv.write_text(
        "LocationID,district,Zone\n1,N/A,\n2,,N/A\n",
        encoding="utf-8",
    )
    raw_csv = tmp_path / "raw.csv"
    pd.DataFrame({"area": [1, 2]}).to_csv(raw_csv, index=False)
    dataset = DatasetContract(
        name="test",
        path=str(raw_csv),
        target="area",
        task=TaskType.REGRESSION,
        datetime_column=None,
        columns={"area": ColumnSchema(dtype="int64", null_count=0, role=ColumnRole.FEATURE)},
        lookup_tables={
            "area": LookupSpec(path=str(lookup_csv), key="LocationID", na_values=na_values)
        },
    )
    return dataset, lookup_csv


def test_keep_default_na_false_preserves_literal_tokens(tmp_path: Path) -> None:
    dataset, _ = _na_dataset(tmp_path, na_values=[])
    _, _, value_audits = load_with_audit(dataset)
    audit = value_audits[0]

    assert audit.na_values == []
    assert _column(audit, "district").null_count == 0
    assert _column(audit, "Zone").null_count == 0


def test_na_values_only_converts_authored_tokens(tmp_path: Path) -> None:
    dataset, _ = _na_dataset(tmp_path, na_values=["N/A"])
    _, _, value_audits = load_with_audit(dataset)
    audit = value_audits[0]

    assert audit.na_values == ["N/A"]
    assert _column(audit, "district").null_count == 1
    assert _column(audit, "Zone").null_count == 1
