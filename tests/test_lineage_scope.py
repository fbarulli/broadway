from __future__ import annotations

from broadway.lineage.graph import LineageScopeError, scope_graph
from broadway.lineage.models import LineageEdge, LineageGraph, LineageNode


def _node(node_id: str) -> LineageNode:
    return LineageNode(
        id=node_id,
        kind=node_id.split(":")[0],
        label=node_id,
        artifact=None,
        status="produced",
    )


def _graph() -> LineageGraph:
    nodes = [
        _node("dataset:test"),
        _node("ingest:test"),
        _node("etl:test"),
        _node("describe:test_hypothesis"),
        _node("stats:test_hypothesis"),
        _node("analysis:test_hypothesis"),
        _node("analysis:test"),
        _node("training:test"),
        _node("slice:airport"),
        _node("decision:keep"),
    ]
    edges = [
        LineageEdge(source="dataset:test", target="ingest:test", relation="produces"),
        LineageEdge(source="ingest:test", target="etl:test", relation="produces"),
        LineageEdge(source="etl:test", target="describe:test_hypothesis", relation="produces"),
        LineageEdge(source="etl:test", target="stats:test_hypothesis", relation="produces"),
        LineageEdge(
            source="analysis:test_hypothesis", target="describe:test_hypothesis", relation="produces"
        ),
        LineageEdge(
            source="analysis:test_hypothesis", target="stats:test_hypothesis", relation="produces"
        ),
        LineageEdge(source="analysis:test", target="training:test", relation="produces"),
        LineageEdge(source="etl:test", target="training:test", relation="produces"),
        LineageEdge(source="slice:airport", target="dataset:test", relation="filters"),
        LineageEdge(source="slice:airport", target="decision:keep", relation="raises"),
    ]
    return LineageGraph(nodes=nodes, edges=edges)


def test_scope_graph_by_analysis_and_dataset() -> None:
    scoped = scope_graph(_graph(), analysis="test_hypothesis", dataset="test")
    node_ids = {n.id for n in scoped.nodes}

    for expected in (
        "dataset:test",
        "etl:test",
        "describe:test_hypothesis",
        "stats:test_hypothesis",
        "analysis:test_hypothesis",
        "slice:airport",
        "decision:keep",
    ):
        assert expected in node_ids

    assert "training:test" not in node_ids
    assert "analysis:test" not in node_ids


def test_scope_graph_by_dataset_only() -> None:
    scoped = scope_graph(_graph(), dataset="test")
    node_ids = {n.id for n in scoped.nodes}

    for expected in ("dataset:test", "ingest:test", "etl:test", "slice:airport", "decision:keep"):
        assert expected in node_ids

    for absent in (
        "analysis:test",
        "analysis:test_hypothesis",
        "describe:test_hypothesis",
        "stats:test_hypothesis",
        "training:test",
    ):
        assert absent not in node_ids


def test_scope_graph_wrong_dataset_raises() -> None:
    try:
        scope_graph(_graph(), analysis="test_hypothesis", dataset="test_other")
    except LineageScopeError:
        pass
    else:
        raise AssertionError("expected LineageScopeError")
