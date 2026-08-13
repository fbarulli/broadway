from __future__ import annotations

from pathlib import Path

import yaml

from broadway.lineage.graph import build_graph
from broadway.lineage.models import DecisionRecord, LineageRecord


def test_build_graph_links_nodes_and_records(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    lineage = tmp_path / "lineage"

    (configs / "dataset").mkdir(parents=True)
    (configs / "analysis").mkdir(parents=True)
    (configs / "slice").mkdir(parents=True)

    (configs / "dataset" / "taxi.yaml").write_text(
        yaml.safe_dump(
            {"name": "taxi", "path": "data/processed/training_data.parquet", "row_count": 100}
        ),
        encoding="utf-8",
    )
    (configs / "analysis" / "taxi.yaml").write_text(
        yaml.safe_dump({"name": "taxi", "mode": "prediction"}),
        encoding="utf-8",
    )
    (configs / "slice" / "airport.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "airport",
                "dataset": "taxi",
                "description": "airport trips",
                "filter_expression": "pickup_location_id == 132",
            }
        ),
        encoding="utf-8",
    )

    artifact = tmp_path / "artifacts" / "baseline" / "baseline.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")

    records_dir = lineage / "records"
    records_dir.mkdir(parents=True)
    baseline_record = LineageRecord(
        node_id="baseline:taxi",
        kind="baseline",
        artifact=str(artifact),
        parents=["analysis:taxi", "profile:taxi"],
    )
    (records_dir / "baseline_taxi.json").write_text(
        baseline_record.model_dump_json(), encoding="utf-8"
    )

    missing_record = LineageRecord(
        node_id="stats:taxi",
        kind="stats",
        artifact=str(tmp_path / "does_not_exist.json"),
        parents=["baseline:taxi"],
    )
    (records_dir / "stats_taxi.json").write_text(
        missing_record.model_dump_json(), encoding="utf-8"
    )

    decisions_dir = lineage / "decisions"
    decisions_dir.mkdir(parents=True)
    decision = DecisionRecord(
        id="keep_outliers",
        question="keep outliers?",
        outcome="yes",
        reason=["r"],
        status="open",
        parents=["slice:airport"],
    )
    (decisions_dir / "keep_outliers.json").write_text(
        decision.model_dump_json(), encoding="utf-8"
    )

    graph = build_graph(configs, lineage)

    node_ids = {n.id for n in graph.nodes}
    assert "dataset:taxi" in node_ids
    assert "analysis:taxi" in node_ids
    assert "slice:airport" in node_ids
    assert "baseline:taxi" in node_ids
    assert "decision:keep_outliers" in node_ids
    assert "stats:taxi" not in node_ids

    edges = {(e.source, e.target) for e in graph.edges}
    assert ("slice:airport", "dataset:taxi") in edges
    assert ("analysis:taxi", "baseline:taxi") in edges
    assert ("slice:airport", "decision:keep_outliers") in edges
