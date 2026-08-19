from __future__ import annotations

from pathlib import Path

import pandas as pd

import broadway.etl.module as etl_module
from broadway.cleaning.models import StructuralCleanResult
from broadway.cleaning.structural import (
    parse_datetime,
    parse_numeric,
    standardize_missing,
)
from broadway.config.loader import load_config
from broadway.config.schema import ColumnRole, ColumnSchema, LookupSpec
from broadway.contracts.pandera import is_numeric_dtype
from broadway.data.cleaner import canonicalize
from broadway.lineage import records
from broadway.lineage.ids import node_id
from broadway.lineage.models import LineageRecord


def test_parse_datetime_records_failures() -> None:
    series = pd.Series(["2024-01-01", "not-a-date", "2024-01-02"])
    parsed, failure = parse_datetime(series, "dt")
    assert parsed.isna().tolist() == [False, True, False]
    assert failure is not None
    assert failure.column == "dt"
    assert failure.count == 1
    assert failure.examples == ["not-a-date"]


def test_parse_datetime_no_drop() -> None:
    series = pd.Series(["2024-01-01", "not-a-date", "2024-01-02"])
    parsed, _ = parse_datetime(series, "dt")
    assert len(parsed) == len(series)


def test_standardize_missing_observed_only() -> None:
    series = pd.Series(["a", "NA", "", "b"])
    cleaned, observed = standardize_missing(series, "col", ["", "NA", "null"])
    assert cleaned.isna().tolist() == [False, True, True, False]
    assert observed == ["NA", ""]


def test_is_numeric_dtype() -> None:
    assert is_numeric_dtype("int64") is True
    assert is_numeric_dtype("float64") is True
    assert is_numeric_dtype("object") is False
    assert is_numeric_dtype("datetime64[us]") is False


def test_parse_numeric_records_failures() -> None:
    series = pd.Series(["1", "2", "abc", "3"])
    coerced, failure = parse_numeric(series, "num", "int64")
    assert coerced.isna().tolist() == [False, False, True, False]
    assert failure is not None
    assert failure.column == "num"
    assert failure.count == 1
    assert failure.examples == ["abc"]
    assert len(coerced) == len(series)


def test_parse_numeric_clean_int_stays_int() -> None:
    series = pd.Series(["1", "2", "3"])
    coerced, failure = parse_numeric(series, "num", "int64")
    assert coerced.dtype == "int64"
    assert failure is None


def test_parse_numeric_failure_stays_float() -> None:
    series = pd.Series(["1", "x", "3"])
    coerced, failure = parse_numeric(series, "num", "int64")
    assert coerced.dtype.kind == "f"
    assert failure is not None


def test_canonicalize_coerces_numeric() -> None:
    df = pd.DataFrame(
        {
            "feature_1": ["1", "2", "x", "4"],
            "target": [10.0, 20.0, 30.0, 40.0],
        }
    )
    out, _, parse_failures, _ = canonicalize(
        df,
        target="target",
        datetime_columns=[],
        numeric_columns={"feature_1": "int64"},
        missing_encodings=[],
    )
    assert len(parse_failures) == 1
    assert parse_failures[0].column == "feature_1"
    assert parse_failures[0].count == 1
    assert parse_failures[0].examples == ["x"]
    assert pd.api.types.is_numeric_dtype(out["feature_1"])
    assert out["feature_1"].isna().tolist() == [False, False, True, False]


def test_canonicalize_order() -> None:
    df = pd.DataFrame(
        {
            "dt": ["2024-01-01", "NA", "2024-01-02"],
            "target": [1.0, 2.0, 3.0],
        }
    )
    out, _, parse_failures, observed = canonicalize(
        df,
        target="target",
        datetime_columns=["dt"],
        missing_encodings=["NA", ""],
        numeric_columns={},
    )
    assert parse_failures == []
    assert observed == {"dt": ["NA"]}
    assert out["dt"].isna().sum() == 1


def test_canonicalize_duplicates_before_normalization() -> None:
    df = pd.DataFrame(
        {
            "a": ["x", "x", "y"],
            "target": [1.0, 1.0, 2.0],
        }
    )
    out, reasons, _, _ = canonicalize(
        df, target="target", datetime_columns=[], missing_encodings=[""], numeric_columns={}
    )
    assert len(out) == 2
    assert any("duplicates" in r for r in reasons)


def test_canonicalize_target_null() -> None:
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "target": [10.0, None, "NA", 40.0],
        }
    )
    out, reasons, _, _ = canonicalize(
        df, target="target", datetime_columns=[], missing_encodings=["NA"], numeric_columns={}
    )
    assert len(out) == 2
    assert any("null target" in r for r in reasons)


def test_etl_module_writes_canonical_and_result(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = load_config("etl", dataset="test", experiment="baseline")
    cfg = cfg.model_copy(
        update={
            "environment": cfg.environment.model_copy(
                update={"data_dir": str(tmp_path)}
            )
        }
    )
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")

    columns = dict(cfg.dataset.columns)
    columns["listed_at"] = ColumnSchema(
        dtype="datetime64[us]", null_count=0, role=ColumnRole.DATETIME
    )
    cfg = cfg.model_copy(
        update={"dataset": cfg.dataset.model_copy(update={"columns": columns})}
    )

    df = pd.DataFrame(
        {
            "feature_2": [100, 100, 150, 200],
            "feature_3": ["a", "a", "b", "c"],
            "target": [10, 10, 30, 40],
            "feature_1": [2, 2, 3, 4],
            "listed_at": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
            ],
        }
    )
    monkeypatch.setattr(etl_module, "load_with_audit", lambda dataset: (df.copy(), [], []))

    etl_module.run(cfg)

    out_dir = Path(cfg.environment.data_dir) / cfg.environment.processed_subdir
    canonical_path = out_dir / "test_canonical.parquet"
    result_path = out_dir / "test_clean.json"
    assert canonical_path.exists()
    assert result_path.exists()

    canonical = pd.read_parquet(canonical_path)
    assert pd.api.types.is_datetime64_any_dtype(canonical["listed_at"])

    result = StructuralCleanResult.model_validate_json(
        result_path.read_text(encoding="utf-8")
    )
    assert result.audit.rows_in == 4
    assert result.audit.rows_out == 3
    assert result.canonical_path == str(canonical_path)
    assert result.parse_failures == []


def _etl_record_path(tmp_path: Path) -> Path:
    return tmp_path / "lineage" / "records" / "etl_test.json"


def _simple_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_2": [100, 101],
            "feature_3": ["a", "b"],
            "target": [10, 20],
            "feature_1": [2, 3],
        }
    )


def test_etl_parent_defaults_to_dataset(tmp_path: Path, monkeypatch) -> None:
    cfg = load_config("etl", dataset="test", experiment="baseline")
    cfg = cfg.model_copy(
        update={
            "environment": cfg.environment.model_copy(update={"data_dir": str(tmp_path)})
        }
    )
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")
    monkeypatch.setattr(etl_module, "load_with_audit", lambda dataset: (_simple_df().copy(), [], []))

    etl_module.run(cfg)

    record = LineageRecord.model_validate_json(
        _etl_record_path(tmp_path).read_text(encoding="utf-8")
    )
    assert record.parents == ["dataset:test"]


def test_etl_parent_ingest_when_present(tmp_path: Path, monkeypatch) -> None:
    cfg = load_config("etl", dataset="test", experiment="baseline")
    cfg = cfg.model_copy(
        update={
            "environment": cfg.environment.model_copy(update={"data_dir": str(tmp_path)})
        }
    )
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")
    records.write_record(
        node_id("ingest", "test"),
        "ingest",
        "data/processed/training_data.parquet",
        [node_id("dataset", "test")],
    )
    monkeypatch.setattr(etl_module, "load_with_audit", lambda dataset: (_simple_df().copy(), [], []))

    etl_module.run(cfg)

    record = LineageRecord.model_validate_json(
        _etl_record_path(tmp_path).read_text(encoding="utf-8")
    )
    assert record.parents == ["ingest:test"]


def test_etl_ci_sampling_gated(tmp_path: Path, monkeypatch) -> None:
    cfg = load_config("etl", dataset="test", experiment="baseline")
    cfg = cfg.model_copy(
        update={
            "environment": cfg.environment.model_copy(update={"data_dir": str(tmp_path)}),
            "etl": cfg.etl.model_copy(update={"ci_sample_size": 2}),
        }
    )
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")

    df = pd.DataFrame(
        {
            "feature_2": [100, 101, 102, 103, 104, 105],
            "feature_3": ["a", "b", "c", "d", "e", "f"],
            "target": [10, 20, 30, 40, 50, 60],
            "feature_1": [2, 3, 2, 4, 3, 5],
        }
    )
    monkeypatch.setattr(etl_module, "load_with_audit", lambda dataset: (df.copy(), [], []))

    canonical_path = (
        Path(cfg.environment.data_dir)
        / cfg.environment.processed_subdir
        / "test_canonical.parquet"
    )

    monkeypatch.setenv("CI", "true")
    etl_module.run(cfg)
    assert len(pd.read_parquet(canonical_path)) == 2

    monkeypatch.delenv("CI", raising=False)
    etl_module.run(cfg)
    assert len(pd.read_parquet(canonical_path)) == 6


def test_etl_with_lookups_writes_join_audit(tmp_path: Path, monkeypatch) -> None:
    cfg = load_config("etl", dataset="test", experiment="baseline")
    cfg = cfg.model_copy(
        update={
            "environment": cfg.environment.model_copy(update={"data_dir": str(tmp_path)})
        }
    )
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")

    raw_path = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "feature_2": [100, 150, 200, 999],
            "feature_3": ["a", "b", "c", "d"],
            "target": [10, 20, 30, 40],
            "feature_1": [2, 3, 4, 5],
        }
    ).to_csv(raw_path, index=False)

    lookup_path = tmp_path / "lookup.csv"
    pd.DataFrame(
        {
            "LocationID": [100, 150, 200],
            "zone": ["Alpha", "Beta", "Gamma"],
        }
    ).to_csv(lookup_path, index=False)

    dataset = cfg.dataset.model_copy(
        update={
            "path": str(raw_path),
            "lookup_tables": {
                "feature_2": LookupSpec(path=str(lookup_path), key="LocationID")
            },
        }
    )
    cfg = cfg.model_copy(update={"dataset": dataset})

    etl_module.run(cfg)

    out_dir = Path(cfg.environment.data_dir) / cfg.environment.processed_subdir
    join_audit_path = out_dir / "test_join_audit.json"
    assert join_audit_path.exists()

    join_record = LineageRecord.model_validate_json(
        (tmp_path / "lineage" / "records" / "join_test.json").read_text(encoding="utf-8")
    )
    assert join_record.parents == ["dataset:test"]

    value_audit_path = out_dir / "test_lookup_value_audit.json"
    assert value_audit_path.exists()

    lookup_value_record = LineageRecord.model_validate_json(
        (tmp_path / "lineage" / "records" / "lookup_value_test.json").read_text(encoding="utf-8")
    )
    assert lookup_value_record.parents == ["join:test"]

    etl_record = LineageRecord.model_validate_json(
        _etl_record_path(tmp_path).read_text(encoding="utf-8")
    )
    assert etl_record.parents == ["join:test"]


def test_column_schema_normalizes_datetime_dtype() -> None:
    col = ColumnSchema(dtype="datetime64[us]", null_count=0, role=ColumnRole.DATETIME)
    assert col.dtype == "datetime64"
