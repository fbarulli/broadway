from __future__ import annotations

from types import SimpleNamespace

import pytest

from broadway.cleaning.models import StructuralCleanResult
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
        canonical_path="data/processed/test_canonical.parquet",
    )


def _join_report(unmatched=0):
    return JoinAuditReport(
        joins=[
            JoinAudit(
                lookup="location_id",
                lookup_path="data/raw/test_lookup.csv",
                left_key="location_id",
                right_key="location_id",
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
                lookup="location_id",
                lookup_path="data/raw/test_lookup.csv",
                matched=100,
                na_values=[],
                columns=[
                    LookupColumnValueAudit(
                        column="district",
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
        name="test",
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
                lookup="location_id",
                lookup_path="data/raw/test_lookup.csv",
                left_key="location_id",
                right_key="location_id",
                rows_attempted=5,
                matched=5,
                unmatched=0,
                null_keys=0,
                unmatched_rate=0.0,
            ),
            JoinAudit(
                lookup="location_id",
                lookup_path="data/raw/test_lookup.csv",
                left_key="location_id",
                right_key="location_id",
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
                lookup="location_id",
                lookup_path="data/raw/test_lookup.csv",
                left_key="location_id",
                right_key="location_id",
                rows_attempted=5,
                matched=5,
                unmatched=0,
                null_keys=0,
                unmatched_rate=0.0,
            ),
            JoinAudit(
                lookup="location_id",
                lookup_path="data/raw/test_lookup.csv",
                left_key="location_id",
                right_key="location_id",
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
    clean_path = tmp_path / "test_clean.json"
    join_path = tmp_path / "test_join_audit.json"
    lookup_path = tmp_path / "test_lookup_value_audit.json"
    profile_path = tmp_path / "profile.json"

    clean_path.write_text(_audit().model_dump_json(), encoding="utf-8")
    join_path.write_text(_join_report().model_dump_json(), encoding="utf-8")
    lookup_path.write_text(_lookup_report(affected_rows=5).model_dump_json(), encoding="utf-8")
    profile_path.write_text(_profile().model_dump_json(), encoding="utf-8")

    fake_cfg = SimpleNamespace(
        dataset=SimpleNamespace(name="test"),
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

    audit.run("test", None, "development")

    for name in ("index.md", "profile.md", "transform.md", "join.md", "lookup_values.md"):
        assert (audit_dir / name).exists(), name
    index_md = (audit_dir / "index.md").read_text(encoding="utf-8")
    assert "Data Audit" in index_md


# --- summary/state helper coverage (index-page assembly) ------------------- #


def test_changed_items_labels_known_kinds_and_skips_zero_counts() -> None:
    """_changed_items maps audit reasons to human labels; zero counts skip."""
    result = _audit(
        dropped=8,
        reasons=["duplicates: -5 rows", "null target: -3 rows", "CI sampling: -0 rows"],
    )
    items = audit._changed_items(result)
    assert items == [
        "exact-duplicate rows dropped: 5",
        "target-missing rows dropped: 3",
    ]


def test_changed_items_parse_failures_counted_by_dtype_kind() -> None:
    from broadway.cleaning.models import ParseFailure

    result = _audit(
        parse_failures=[
            ParseFailure(column="d", target_dtype="datetime64[ns]", count=2, examples=[]),
            ParseFailure(column="n", target_dtype="float64", count=0, examples=[]),
        ],
    )
    items = audit._changed_items(result)
    assert items == ["datetime parse failures in d: 2"]


def test_changed_items_none_result_is_empty() -> None:
    assert audit._changed_items(None) == []


def test_state_helpers_none_means_incomplete() -> None:
    assert audit._transform_state(None) == audit.INCOMPLETE
    assert audit._join_state(None) == audit.INCOMPLETE
    assert audit._lookup_state(None) == audit.INCOMPLETE
    assert audit._profile_state(None) == audit.INCOMPLETE


def test_state_helpers_pass_and_warning_paths() -> None:
    clean = _audit()  # no failures, no unexplained drops
    assert audit._transform_state(clean) == audit.PASS
    dirty = _audit(unexplained=1, dropped=1)
    assert audit._transform_state(dirty) == audit.WARNING
    assert audit._join_state(_join_report(unmatched=0)) == audit.PASS
    assert audit._join_state(_join_report(unmatched=4)) == audit.WARNING
    assert audit._lookup_state(_lookup_report(affected_rows=0)) == audit.PASS
    assert audit._lookup_state(_lookup_report(affected_rows=2)) == audit.WARNING


def test_summary_helpers_none_and_populated() -> None:
    assert audit._join_summary(None) == "no join audit available"
    assert audit._lookup_summary(None) == "no lookup value audit available"
    assert "0 unmatched" in audit._join_summary(_join_report(unmatched=0))
    assert "unmatched key event" in audit._join_summary(_join_report(unmatched=7))
    assert "all matched" in audit._lookup_summary(_lookup_report(affected_rows=0))
    assert "missing or sentinel" in audit._lookup_summary(_lookup_report(affected_rows=9))
