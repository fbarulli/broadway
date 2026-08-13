from __future__ import annotations

from pathlib import Path

import pandas as pd

import broadway.etl.module as etl_module
from broadway.cleaning.models import StructuralCleanResult
from broadway.cleaning.structural import parse_datetime, standardize_missing
from broadway.config.loader import load_config
from broadway.config.schema import ColumnRole, ColumnSchema
from broadway.data.cleaner import canonicalize
from broadway.lineage import records


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


def test_canonicalize_order() -> None:
    df = pd.DataFrame(
        {
            "dt": ["2024-01-01", "NA", "2024-01-02"],
            "target": [1.0, 2.0, 3.0],
        }
    )
    out, _, parse_failures, observed = canonicalize(
        df, target="target", datetime_columns=["dt"], missing_encodings=["NA", ""]
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
        df, target="target", datetime_columns=[], missing_encodings=[""]
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
        df, target="target", datetime_columns=[], missing_encodings=["NA"]
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
        dtype="object", null_count=0, role=ColumnRole.DATETIME
    )
    cfg = cfg.model_copy(
        update={"dataset": cfg.dataset.model_copy(update={"columns": columns})}
    )

    df = pd.DataFrame(
        {
            "area": [100, 100, 150, 200],
            "neighborhood": ["a", "a", "b", "c"],
            "price": [10, 10, 30, 40],
            "rooms": [2, 2, 3, 4],
            "listed_at": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
            ],
        }
    )
    monkeypatch.setattr(etl_module, "load", lambda dataset: df.copy())

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
