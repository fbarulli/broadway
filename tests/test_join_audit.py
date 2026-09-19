from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from broadway.config.schema import (
    ColumnRole,
    ColumnSchema,
    DatasetContract,
    LookupSpec,
    TaskType,
)
from broadway.data.join_audit import JoinAuditReport, audit_join
from broadway.data.loader import load, load_with_audit


def _lookup_spec(path: str) -> LookupSpec:
    return LookupSpec(path=path, key="LocationID")


def test_audit_join_matched_unmatched_null_keys() -> None:
    df = pd.DataFrame(
        {
            "location_id": [1, 2, 3, None, 5],
        }
    )
    lookup_df = pd.DataFrame(
        {
            "LocationID": [1, 2, 3],
            "zone": ["a", "b", "c"],
        }
    )
    audit = audit_join(df, "location_id", _lookup_spec("lookup.csv"), lookup_df)

    assert audit.matched == 3
    assert audit.unmatched == 1
    assert audit.null_keys == 1
    assert audit.rows_attempted == 5
    assert audit.rows_attempted == audit.matched + audit.unmatched + audit.null_keys
    assert audit.unmatched_rate == pytest.approx(audit.unmatched / (audit.matched + audit.unmatched))


def test_audit_join_all_matched() -> None:
    df = pd.DataFrame({"key": [1, 2, 3]})
    lookup_df = pd.DataFrame({"LocationID": [1, 2, 3], "zone": ["a", "b", "c"]})
    audit = audit_join(df, "key", _lookup_spec("lookup.csv"), lookup_df)

    assert audit.matched == 3
    assert audit.unmatched == 0
    assert audit.null_keys == 0
    assert audit.unmatched_rate == 0.0


def test_audit_join_no_non_null_keys() -> None:
    df = pd.DataFrame({"key": [None, None]})
    lookup_df = pd.DataFrame({"LocationID": [1, 2], "zone": ["a", "b"]})
    audit = audit_join(df, "key", _lookup_spec("lookup.csv"), lookup_df)

    assert audit.matched == 0
    assert audit.unmatched == 0
    assert audit.null_keys == 2
    assert audit.unmatched_rate == 0.0


def _dataset(path: str, lookup_csv: str | None = None) -> DatasetContract:
    return DatasetContract(
        name="test",
        path=path,
        target="price",
        task=TaskType.REGRESSION,
        datetime_column=None,
        columns={
            "area": ColumnSchema(dtype="int64", null_count=0, role=ColumnRole.FEATURE),
            "price": ColumnSchema(dtype="int64", null_count=0, role=ColumnRole.TARGET),
        },
        lookup_tables={} if lookup_csv is None else {"area": _lookup_spec(lookup_csv)},
    )


def test_load_with_audit_returns_df_and_audits(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.csv"
    pd.DataFrame({"area": [1, 2, 3], "price": [10, 20, 30]}).to_csv(raw_path, index=False)

    lookup_csv = tmp_path / "lookup.csv"
    pd.DataFrame({"LocationID": [1, 2], "zone": ["a", "b"]}).to_csv(lookup_csv, index=False)

    df, audits, value_audits = load_with_audit(_dataset(str(raw_path), str(lookup_csv)))

    assert "zone" in df.columns
    assert len(audits) == 1
    assert len(value_audits) == 1
    audit = audits[0]
    assert audit.rows_attempted == 3
    assert audit.matched == 2
    assert audit.unmatched == 1
    assert audit.rows_attempted == audit.matched + audit.unmatched + audit.null_keys


def test_load_returns_only_df(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.csv"
    pd.DataFrame({"area": [1, 2, 3], "price": [10, 20, 30]}).to_csv(raw_path, index=False)

    df = load(_dataset(str(raw_path)))

    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"area", "price"}


def test_join_audit_report_round_trip(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.csv"
    pd.DataFrame({"area": [1, 2, 3], "price": [10, 20, 30]}).to_csv(raw_path, index=False)

    lookup_csv = tmp_path / "lookup.csv"
    pd.DataFrame({"LocationID": [1, 2], "zone": ["a", "b"]}).to_csv(lookup_csv, index=False)

    _, audits, _ = load_with_audit(_dataset(str(raw_path), str(lookup_csv)))
    report = JoinAuditReport(joins=audits)
    parsed = JoinAuditReport.model_validate_json(report.model_dump_json())
    assert parsed == report
