from __future__ import annotations

from broadway.lineage.models import DecisionRecord, LineageGraph, LineageNode
from broadway.lineage.state import current_state


def _graph(kinds: set[str]) -> LineageGraph:
    return LineageGraph(
        nodes=[LineageNode(id=k, kind=k, label=k, status="produced") for k in sorted(kinds)],
        edges=[],
    )


def _decisions() -> list[DecisionRecord]:
    return [
        DecisionRecord(id="a", question="q", outcome="o", reason=["r"], status="open", parents=[]),
        DecisionRecord(
            id="b", question="q", outcome="o", reason=["r"], status="resolved", parents=[]
        ),
    ]


def test_prediction_state() -> None:
    state = current_state(_graph({"profile", "baseline"}), "prediction", "g", _decisions())
    assert state.stage == "baseline"
    assert state.not_yet_run == ["etl", "features", "training", "evaluation"]
    assert state.ran_but_output_missing == []
    assert state.open_decisions == ["a"]
    assert state.resolved_decisions == ["b"]


def test_hypothesis_state() -> None:
    state = current_state(_graph({"profile", "baseline"}), "hypothesis", "g", [])
    assert state.stage == "baseline"
    assert state.not_yet_run == ["etl", "stats"]
    assert state.ran_but_output_missing == []


def test_non_lineage_kinds_never_listed() -> None:
    kinds = {"profile", "baseline", "etl", "eda", "contracts", "features"}
    state = current_state(_graph(kinds), "prediction", "g", [])
    for kind in ("etl", "eda", "contracts", "features"):
        assert kind not in state.not_yet_run
        assert kind not in state.ran_but_output_missing


def test_ran_but_output_missing_distinct_from_not_yet_run() -> None:
    graph = LineageGraph(
        nodes=[
            LineageNode(
                id="baseline:taxi", kind="baseline", label="baseline:taxi", status="ran_but_output_missing"
            ),
            LineageNode(id="profile:taxi", kind="profile", label="profile:taxi", status="produced"),
        ],
        edges=[],
    )
    state = current_state(graph, "prediction", "g", [])
    assert state.ran_but_output_missing == ["baseline"]
    assert "baseline" not in state.not_yet_run
    assert "etl" in state.not_yet_run
    assert state.stage == "profile"
