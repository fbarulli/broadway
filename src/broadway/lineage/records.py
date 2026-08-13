from __future__ import annotations

import os
from pathlib import Path

from broadway.lineage.models import LineageRecord

LINEAGE_DIR = Path(os.getenv("BROADWAY_LINEAGE_DIR", "artifacts/lineage"))


def records_dir() -> Path:
    return LINEAGE_DIR / "records"


def decisions_dir() -> Path:
    return LINEAGE_DIR / "decisions"


def write_record(node_id: str, kind: str, artifact: str, parents: list[str]) -> None:
    record = LineageRecord(node_id=node_id, kind=kind, artifact=artifact, parents=parents)
    out_dir = records_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = node_id.replace(":", "_") + ".json"
    (out_dir / filename).write_text(record.model_dump_json(indent=2), encoding="utf-8")
