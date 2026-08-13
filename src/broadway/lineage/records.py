from __future__ import annotations

import os
from pathlib import Path

from broadway.lineage.models import LineageRecord, SampleRole, TransformAudit

LINEAGE_DIR = Path(os.getenv("BROADWAY_LINEAGE_DIR", "artifacts/lineage"))
REPORTS_DIR = Path(os.getenv("BROADWAY_REPORTS_DIR", "reports"))


def records_dir() -> Path:
    return LINEAGE_DIR / "records"


def decisions_dir() -> Path:
    return LINEAGE_DIR / "decisions"


def write_record(
    node_id: str,
    kind: str,
    artifact: str,
    parents: list[str],
    audit: TransformAudit | None = None,
    sample_name: str | None = None,
    sample_role: SampleRole | None = None,
) -> None:
    record = LineageRecord(
        node_id=node_id,
        kind=kind,
        artifact=artifact,
        parents=parents,
        audit=audit,
        sample_name=sample_name,
        sample_role=sample_role,
    )
    out_dir = records_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = node_id.replace(":", "_") + ".json"
    (out_dir / filename).write_text(record.model_dump_json(indent=2), encoding="utf-8")


def enforce_drop_fraction(audit: TransformAudit, max_fraction: float) -> None:
    if audit.rows_in <= 0:
        return
    fraction = audit.rows_dropped_unexplained / audit.rows_in
    if fraction > max_fraction:
        raise ValueError(
            f"unexplained row loss {audit.rows_dropped_unexplained}/{audit.rows_in} "
            f"({fraction:.1%}) exceeds max_drop_fraction {max_fraction:.1%}"
        )
