from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

SampleRole = Literal["diagnostic", "estimation"]


class DatasetRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    path: str
    row_count: int | None = None


class FilterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str
    op: Literal[">", "<", ">=", "<=", "==", "!="]
    value: float


class DerivedSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    # Resolved at runtime against the shared transform registry
    # (broadway.features.builders.BUILDERS) — same implementation functions
    # the feature pipeline uses.
    formula: str
    # Role → source-column mapping for the formula's inputs; defaults to
    # generic names ("distance", "duration_minutes"). Dataset-specific
    # configs can map their own column names onto the roles.
    columns: dict[str, str] = {}


class SampleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    role: SampleRole
    path: str
    description: str | None = None
    column_mapping: dict[str, str] = {}
    version: str = "v1"
    seed: int | None = None
    size: int | None = None
    source: DatasetRef | None = None
    columns: list[str] | None = None
    derived: list[DerivedSpec] | None = None
    filters: list[FilterSpec] | None = None
    exclude_any: list[list[FilterSpec]] | None = None
    schema: dict | None = None  # type: ignore[assignment]


class DatasetSlice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    dataset: str
    description: str
    filter_expression: str | None = None


class DecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    question: str
    outcome: str
    reason: list[str]
    status: Literal["open", "resolved"]
    parents: list[str]


class TransformAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows_in: int
    rows_out: int
    rows_dropped_total: int
    rows_dropped_unexplained: int
    reasons: list[str]
    columns_before: list[str]
    columns_after: list[str]
    columns_added: list[str]
    columns_removed: list[str]


class LineageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_id: str
    kind: str
    artifact: str
    parents: list[str]
    audit: TransformAudit | None = None
    sample_name: str | None = None
    sample_role: SampleRole | None = None


class LineageNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: str
    label: str
    artifact: str | None = None
    status: Literal[
        "produced", "not_yet_produced", "ran_but_output_missing", "referenced_not_found"
    ]
    sample_name: str | None = None
    sample_role: SampleRole | None = None


class LineageEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    target: str
    relation: str


class LineageGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nodes: list[LineageNode]
    edges: list[LineageEdge]


class RunState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal: str | None
    stage: str | None
    open_decisions: list[str]
    resolved_decisions: list[str]
    not_yet_run: list[str]
    ran_but_output_missing: list[str]
