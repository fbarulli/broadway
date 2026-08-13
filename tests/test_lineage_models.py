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
    SampleSpec,
    TransformAudit,
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


def test_transform_audit_and_lineage_record_round_trip() -> None:
    audit = TransformAudit(
        rows_in=100,
        rows_out=95,
        rows_dropped_total=5,
        rows_dropped_unexplained=0,
        reasons=["null target: -3 rows", "duplicates: -2 rows"],
        columns_before=["a", "b", "c"],
        columns_after=["a", "b", "d"],
        columns_added=["d"],
        columns_removed=["c"],
    )
    rec = LineageRecord(
        node_id="etl:taxi",
        kind="etl",
        artifact="data/processed/train.parquet",
        parents=["dataset:taxi"],
        audit=audit,
    )
    loaded = LineageRecord.model_validate_json(rec.model_dump_json())
    assert loaded == rec
    assert loaded.audit == audit

    without_audit = LineageRecord(
        node_id="baseline:taxi", kind="baseline", artifact="a.json", parents=[]
    )
    parsed = LineageRecord.model_validate_json(without_audit.model_dump_json())
    assert parsed.audit is None


def test_sample_spec_requires_name_role_path() -> None:
    spec = SampleSpec(name="taxi", role="diagnostic", path="results/sample.parquet")
    assert spec.role == "diagnostic"
    assert spec.description is None
    with pytest.raises(ValidationError):
        SampleSpec(name="taxi", path="results/sample.parquet")
    with pytest.raises(ValidationError):
        SampleSpec(name="taxi", role="bogus", path="results/sample.parquet")


def test_lineage_node_and_record_carry_sample() -> None:
    node = LineageNode(
        id="describe:taxi",
        kind="describe",
        label="describe:taxi",
        status="produced",
        sample_name="taxi_diagnostic",
        sample_role="diagnostic",
    )
    assert node.sample_name == "taxi_diagnostic"
    assert node.sample_role == "diagnostic"

    rec = LineageRecord(
        node_id="describe:taxi",
        kind="describe",
        artifact="describe.json",
        parents=[],
        sample_name="taxi_diagnostic",
        sample_role="diagnostic",
    )
    assert rec.sample_name == "taxi_diagnostic"
    assert rec.sample_role == "diagnostic"

    bare_node = LineageNode(id="a:b", kind="b", label="b", status="produced")
    assert bare_node.sample_name is None
    assert bare_node.sample_role is None
    bare_rec = LineageRecord(node_id="a:b", kind="b", artifact="x", parents=[])
    assert bare_rec.sample_name is None
    assert bare_rec.sample_role is None
