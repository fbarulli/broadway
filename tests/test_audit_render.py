from __future__ import annotations

from types import SimpleNamespace

import pytest

from broadway.cleaning.models import ParseFailure, StructuralCleanResult
from broadway.data.join_audit import JoinAudit, JoinAuditReport
from broadway.data.lookup_value_audit import (
    LookupColumnValueAudit,
    LookupValueAudit,
    LookupValueAuditReport,
)
from broadway.discover.profile import ColumnProfile, DatasetProfile
from broadway.lineage.models import TransformAudit
from broadway.reports import audit


def _audit(rows_in=100, rows_out=100, dropped=0, unexplained=0, reasons=None, parse_failures=None):
    return StructuralCleanResult(
        audit=TransformAudit(
            rows_in=rows_in,
            rows_out=rows_out,
            rows_dropped_total=dropped,
            rows_dropped_unexplained=unexplained,
            reasons=reasons if reasons is not None else [],
            columns_before=["a"],
            columns_after=["a"],
            columns_added=[],
            columns_removed=[],
        ),
        parse_failures=parse_failures if parse_failures is not None else [],
        missing_encodings={},
        canonical_path="data/processed/taxi_canonical.parquet",
    )


def _join_report(unmatched=0):
    return JoinAuditReport(
        joins=[
            JoinAudit(
                lookup="pickup_location_id",
                lookup_path="data/raw/taxi_zone_lookup.csv",
                left_key="pickup_location_id",
                right_key="LocationID",
                rows_attempted=5,
                matched=5 - unmatched,
                unmatched=unmatched,
                null_keys=0,
                unmatched_rate=round(unmatched / 5, 6),
            )
        ]
    )


def _lookup_report(affected_rows=0):
    return LookupValueAuditReport(
        lookups=[
            LookupValueAudit(
                lookup="pickup_location_id",
                lookup_path="data/raw/taxi_zone_lookup.csv",
                matched=100,
                na_values=[],
                columns=[
                    LookupColumnValueAudit(
                        column="Borough",
                        null_count=0,
                        sentinel_counts={"Unknown": affected_rows},
                        affected_rows=affected_rows,
                        affected_rate=round(affected_rows / 100, 6),
                        affected_lookup_keys=["264"] if affected_rows else [],
                    )
                ],
            )
        ]
    )


def _profile(identifier_score=0.5):
    return DatasetProfile(
        name="taxi",
        path="data/processed/training_data.parquet",
        row_count=10,
        columns={
            "id": ColumnProfile(
                dtype="int64",
                null_count=0,
                cardinality=10,
                min="1",
                max="10",
                datetime_min=None,
                datetime_max=None,
                identifier_score=identifier_score,
            )
        },
    )


def test_render_join_all_matched_notes_value_usability() -> None:
    md = audit.render_join(_join_report(unmatched=0))
    assert "Rows evaluated" in md
    assert "Matched join-key events" in md
    assert "key matching only" in md
    assert "matched values were usable" in md


def test_render_join_distinguishes_rows_from_events() -> None:
    report = JoinAuditReport(
        joins=[
            JoinAudit(
                lookup="pickup_location_id",
                lookup_path="data/raw/taxi_zone_lookup.csv",
                left_key="pickup_location_id",
                right_key="LocationID",
                rows_attempted=5,
                matched=5,
                unmatched=0,
                null_keys=0,
                unmatched_rate=0.0,
            ),
            JoinAudit(
                lookup="dropoff_location_id",
                lookup_path="data/raw/taxi_zone_lookup.csv",
                left_key="dropoff_location_id",
                right_key="LocationID",
                rows_attempted=5,
                matched=5,
                unmatched=0,
                null_keys=0,
                unmatched_rate=0.0,
            ),
        ]
    )
    md = audit.render_join(report)
    assert "Rows evaluated: 5" in md
    assert "Lookup joins checked: 2" in md
    assert "Matched join-key events: 10" in md


def test_join_counts_raises_on_divergent_row_counts() -> None:
    report = JoinAuditReport(
        joins=[
            JoinAudit(
                lookup="pickup_location_id",
                lookup_path="data/raw/taxi_zone_lookup.csv",
                left_key="pickup_location_id",
                right_key="LocationID",
                rows_attempted=5,
                matched=5,
                unmatched=0,
                null_keys=0,
                unmatched_rate=0.0,
            ),
            JoinAudit(
                lookup="dropoff_location_id",
                lookup_path="data/raw/taxi_zone_lookup.csv",
                left_key="dropoff_location_id",
                right_key="LocationID",
                rows_attempted=4,
                matched=4,
                unmatched=0,
                null_keys=0,
                unmatched_rate=0.0,
            ),
        ]
    )
    with pytest.raises(ValueError):
        audit._join_counts(report)


def test_render_lookup_values_affected_mentions_deficiency_and_no_redefinition() -> None:
    md = audit.render_lookup_values(_lookup_report(affected_rows=5))
    assert "deficient values" in md
    assert "did not exclude these rows or redefine the population" in md


def test_render_transform_no_domain_cleaning_sentence() -> None:
    md = audit.render_transform(_audit())
    assert "No domain or outlier cleaning was performed here." in md


def test_render_profile_high_identifier() -> None:
    md = audit.render_profile(_profile(identifier_score=1.0))
    assert "behaves like an identifier" in md


def test_run_writes_five_markdown_files(
    tmp_path, monkeypatch
) -> None:
    clean_path = tmp_path / "taxi_clean.json"
    join_path = tmp_path / "taxi_join_audit.json"
    lookup_path = tmp_path / "taxi_lookup_value_audit.json"
    profile_path = tmp_path / "profile.json"

    clean_path.write_text(_audit().model_dump_json(), encoding="utf-8")
    join_path.write_text(_join_report().model_dump_json(), encoding="utf-8")
    lookup_path.write_text(_lookup_report(affected_rows=5).model_dump_json(), encoding="utf-8")
    profile_path.write_text(_profile().model_dump_json(), encoding="utf-8")

    fake_cfg = SimpleNamespace(
        dataset=SimpleNamespace(name="taxi"),
        environment=SimpleNamespace(data_dir=str(tmp_path), processed_subdir="processed"),
    )
    audit_dir = tmp_path / "audit"
    paths = {
        "clean": clean_path,
        "join": join_path,
        "lookup": lookup_path,
        "profile": profile_path,
    }
    monkeypatch.setattr(audit, "load_config", lambda *a, **k: fake_cfg)
    monkeypatch.setattr(audit, "_evidence_paths", lambda cfg: paths)
    monkeypatch.setattr(audit, "AUDIT_DIR", audit_dir)

    audit.run("taxi", None, "development")

    for name in ("index.md", "profile.md", "transform.md", "join.md", "lookup_values.md"):
        assert (audit_dir / name).exists(), name
    index_md = (audit_dir / "index.md").read_text(encoding="utf-8")
    assert "Data Audit" in index_md
