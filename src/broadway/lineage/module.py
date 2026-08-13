from __future__ import annotations

import logging

from broadway.config.loader import CONFIGS_DIR, load_config
from broadway.lineage.graph import build_graph, load_decisions
from broadway.lineage.mermaid import to_mermaid
from broadway.lineage.records import LINEAGE_DIR
from broadway.lineage.state import current_state
from broadway.reports.paths import REPORTS_DIR

logger = logging.getLogger(__name__)


def run(analysis: str, dataset: str) -> None:
    cfg = load_config("full", dataset=dataset, analysis=analysis)
    if cfg.analysis is None:
        raise ValueError("lineage requires an analysis contract")
    mode = cfg.analysis.mode.value
    graph = build_graph(CONFIGS_DIR, LINEAGE_DIR)
    decisions = load_decisions(LINEAGE_DIR)
    state = current_state(graph, mode, cfg.analysis.goal, decisions)

    out_dir = REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "graph.json").write_text(graph.model_dump_json(indent=2), encoding="utf-8")
    (out_dir / "graph.md").write_text("```mermaid\n" + to_mermaid(graph) + "\n```\n", encoding="utf-8")

    logger.info("lineage graph written to %s/graph.json and %s/graph.md", out_dir, out_dir)
    print(f"goal: {state.goal}")
    print(f"stage: {state.stage or 'not started'}")
    print(f"open decisions: {state.open_decisions}")
    print(f"resolved decisions: {state.resolved_decisions}")
    print(f"not yet run: {state.not_yet_run}")
    print(f"ran but output missing: {state.ran_but_output_missing}")
