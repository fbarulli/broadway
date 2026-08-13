from __future__ import annotations

import pytest
from pydantic import ValidationError

from broadway.lineage.models import (
    DatasetRef,
    DatasetSlice,
    DecisionRecord,
    LineageEdge,
    LineageGraph,
    LineageNode,
    LineageRecord,
)


def test_dataset_ref_requires_name_and_path() -> None:
    ref = DatasetRef(name="taxi", path="data/processed/training_data.parquet")
    assert ref.name == "taxi"
    assert ref.path == "data/processed/training_data.parquet"
    assert ref.row_count is None
    with pytest.raises(ValidationError):
        DatasetRef(name="taxi")
    with pytest.raises(ValidationError):
        DatasetRef(path="x.parquet")


def test_dataset_slice_requires_name_dataset_description() -> None:
    slice_ = DatasetSlice(name="airport", dataset="taxi", description="d")
    assert slice_.filter_expression is None
    with pytest.raises(ValidationError):
        DatasetSlice(name="airport", dataset="taxi")


def test_decision_record_requires_fields() -> None:
    decision = DecisionRecord(
        id="keep_outliers",
        question="q",
        outcome="o",
        reason=["r"],
        status="open",
        parents=["slice:airport"],
    )
    assert decision.status == "open"
    with pytest.raises(ValidationError):
        DecisionRecord(question="q", outcome="o", reason=["r"], status="open", parents=[])


def test_decision_record_rejects_bad_status() -> None:
    with pytest.raises(ValidationError):
        DecisionRecord(
            id="d",
            question="q",
            outcome="o",
            reason=["r"],
            status="bogus",
            parents=[],
        )


def test_lineage_record_requires_fields() -> None:
    rec = LineageRecord(node_id="baseline:taxi", kind="baseline", artifact="a.json", parents=[])
    assert rec.node_id == "baseline:taxi"
    with pytest.raises(ValidationError):
        LineageRecord(kind="baseline", artifact="a.json", parents=[])


def test_lineage_graph_json_round_trip() -> None:
    graph = LineageGraph(
        nodes=[
            LineageNode(id="dataset:taxi", kind="dataset", label="taxi", status="produced"),
        ],
        edges=[LineageEdge(source="dataset:taxi", target="baseline:taxi", relation="produced_by")],
    )
    loaded = LineageGraph.model_validate_json(graph.model_dump_json())
    assert loaded == graph
