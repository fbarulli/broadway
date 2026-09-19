from __future__ import annotations

from broadway.lineage.models import DecisionRecord, LineageGraph, RunState

LINEAGE_STEPS = {
    "prediction": ["profile", "etl", "baseline", "features", "training", "evaluation"],
    "hypothesis": ["profile", "etl", "baseline", "stats"],
    "causal": ["profile", "etl", "baseline", "causal"],
}


def current_state(
    graph: LineageGraph, mode: str, goal: str, decisions: list[DecisionRecord]
) -> RunState:
    ordered = LINEAGE_STEPS.get(mode, [])
    status_by_kind: dict[str, list[str]] = {}
    for node in graph.nodes:
        status_by_kind.setdefault(node.kind, []).append(node.status)
    produced = [k for k in ordered if "produced" in status_by_kind.get(k, [])]
    stage = produced[-1] if produced else None
    not_yet_run = [k for k in ordered if k not in status_by_kind]
    ran_but_output_missing = [
        k for k in ordered if k in status_by_kind and "produced" not in status_by_kind[k]
    ]
    open_decisions = sorted(d.id for d in decisions if d.status == "open")
    resolved_decisions = sorted(d.id for d in decisions if d.status == "resolved")
    return RunState(
        goal=goal,
        stage=stage,
        open_decisions=open_decisions,
        resolved_decisions=resolved_decisions,
        not_yet_run=not_yet_run,
        ran_but_output_missing=ran_but_output_missing,
    )
