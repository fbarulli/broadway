from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from broadway.lineage.models import TransformAudit


class ParseFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    count: int
    examples: list[str]
    target_dtype: str


class StructuralCleanResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audit: TransformAudit
    parse_failures: list[ParseFailure]
    missing_encodings: dict[str, list[str]]
    canonical_path: str
