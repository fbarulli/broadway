from __future__ import annotations

from pathlib import Path

import yaml

from broadway.lineage.graph import KIND_LABELS, build_graph
from broadway.lineage.models import DecisionRecord, LineageRecord


def test_kind_labels_include_etl() -> None:
    assert KIND_LABELS["etl"] == "ETL"


def test_build_graph_links_nodes_and_records(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    lineage = tmp_path / "lineage"

    (configs / "dataset").mkdir(parents=True)
    (configs / "analysis").mkdir(parents=True)
    (configs / "slice").mkdir(parents=True)

    (configs / "dataset" / "test.yaml").write_text(
        yaml.safe_dump(
            {"name": "test", "path": "data/processed/training_data.parquet", "row_count": 100}
        ),
        encoding="utf-8",
    )
    (configs / "analysis" / "test.yaml").write_text(
        yaml.safe_dump({"name": "test", "mode": "prediction"}),
        encoding="utf-8",
    )
    (configs / "slice" / "airport.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "airport",
                "dataset": "test",
                "description": "airport trips",
                "filter_expression": "location_id == 132",
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
        node_id="baseline:test",
        kind="baseline",
        artifact=str(artifact),
        parents=["analysis:test", "profile:test"],
    )
    (records_dir / "baseline_test.json").write_text(
        baseline_record.model_dump_json(), encoding="utf-8"
    )

    missing_record = LineageRecord(
        node_id="stats:test",
        kind="stats",
        artifact=str(tmp_path / "does_not_exist.json"),
        parents=["baseline:test"],
    )
    (records_dir / "stats_test.json").write_text(
        missing_record.model_dump_json(), encoding="utf-8"
    )

    describe_record = LineageRecord(
        node_id="describe:test",
        kind="describe",
        artifact=str(tmp_path / "does_not_exist.json"),
        parents=["analysis:test"],
        sample_name="test_diagnostic",
        sample_role="diagnostic",
    )
    (records_dir / "describe_test.json").write_text(
        describe_record.model_dump_json(), encoding="utf-8"
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
    assert "dataset:test" in node_ids
    assert "analysis:test" in node_ids
    assert "slice:airport" in node_ids
    assert "baseline:test" in node_ids
    assert "decision:keep_outliers" in node_ids
    assert "stats:test" in node_ids
    assert "describe:test" in node_ids

    sample_by_id = {n.id: (n.sample_name, n.sample_role) for n in graph.nodes}
    assert sample_by_id["describe:test"] == ("test_diagnostic", "diagnostic")
    assert sample_by_id["stats:test"] == (None, None)

    status_by_id = {n.id: n.status for n in graph.nodes}
    assert status_by_id["baseline:test"] == "produced"
    assert status_by_id["stats:test"] == "ran_but_output_missing"

    edges = {(e.source, e.target) for e in graph.edges}
    assert ("slice:airport", "dataset:test") in edges
    assert ("analysis:test", "baseline:test") in edges
    assert ("slice:airport", "decision:keep_outliers") in edges
    assert ("baseline:test", "stats:test") in edges


def test_record_edges_use_produces_relation(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    lineage = tmp_path / "lineage"
    (configs / "analysis").mkdir(parents=True)
    (configs / "analysis" / "test.yaml").write_text(
        yaml.safe_dump({"name": "test", "mode": "prediction"}), encoding="utf-8"
    )

    records_dir = lineage / "records"
    records_dir.mkdir(parents=True)
    baseline = LineageRecord(
        node_id="baseline:test",
        kind="baseline",
        artifact=str(tmp_path / "baseline.json"),
        parents=["analysis:test"],
    )
    (records_dir / "baseline_test.json").write_text(
        baseline.model_dump_json(), encoding="utf-8"
    )
    stats = LineageRecord(
        node_id="stats:test",
        kind="stats",
        artifact=str(tmp_path / "stats.json"),
        parents=["baseline:test"],
    )
    (records_dir / "stats_test.json").write_text(stats.model_dump_json(), encoding="utf-8")

    graph = build_graph(configs, lineage)

    relations = {(e.source, e.target): e.relation for e in graph.edges}
    assert relations[("baseline:test", "stats:test")] == "produces"


def test_dangling_parent_gets_placeholder_node(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    lineage = tmp_path / "lineage"
    (configs / "analysis").mkdir(parents=True)
    (configs / "analysis" / "test.yaml").write_text(
        yaml.safe_dump({"name": "test", "mode": "prediction"}), encoding="utf-8"
    )

    records_dir = lineage / "records"
    records_dir.mkdir(parents=True)
    rec = LineageRecord(
        node_id="stats:test",
        kind="stats",
        artifact=str(tmp_path / "stats.json"),
        parents=["missing:thing"],
    )
    (records_dir / "stats_test.json").write_text(rec.model_dump_json(), encoding="utf-8")

    graph = build_graph(configs, lineage)

    nodes_by_id = {n.id: n for n in graph.nodes}
    assert "missing:thing" in nodes_by_id
    assert nodes_by_id["missing:thing"].status == "referenced_not_found"
