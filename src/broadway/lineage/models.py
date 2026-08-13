from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class DatasetRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    path: str
    row_count: int | None = None


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


class LineageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_id: str
    kind: str
    artifact: str
    parents: list[str]


class LineageNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: str
    label: str
    artifact: str | None = None
    status: Literal["produced", "not_yet_produced"]


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
    not_yet_produced: list[str]
