"""Tests for broadway.lineage.module.run — the lineage report orchestration.

The graph/mermaid/state internals have their own test modules; this covers the
orchestration contract: the analysis-contract guard, and the two persisted
artifacts (graph.json / graph.md) plus the human summary printed to stdout,
with a seeded record chain so the analysis is derivable from its dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from broadway.lineage import module as lineage_module
from broadway.lineage.models import LineageRecord


def _seed_records(lineage_dir: Path, tmp_path: Path) -> None:
    """dataset:test → profile:test → baseline:test_hypothesis ← analysis.

    scope_graph's backward closure needs exactly this chain for
    analysis 'test_hypothesis' to be derived from dataset 'test'.
    """
    records_dir = lineage_dir / "records"
    records_dir.mkdir(parents=True)
    artifact = tmp_path / "profile.json"
    artifact.write_text("{}", encoding="utf-8")
    recs = [
        LineageRecord(
            node_id="profile:test", kind="profile", artifact=str(artifact),
            parents=["dataset:test"],
        ),
        LineageRecord(
            node_id="baseline:test_hypothesis", kind="baseline",
            artifact=str(artifact),
            parents=["profile:test", "analysis:test_hypothesis"],
        ),
    ]
    for i, r in enumerate(recs):
        (records_dir / f"rec_{i}.json").write_text(r.model_dump_json(), encoding="utf-8")


def test_lineage_run_requires_analysis_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lineage_module, "LINEAGE_DIR", tmp_path / "lineage")
    monkeypatch.setattr(lineage_module, "REPORTS_DIR", tmp_path / "reports")
    # a merged config without an analysis block (loader regression scenario)
    monkeypatch.setattr(
        lineage_module, "load_config",
        lambda *a, **k: SimpleNamespace(analysis=None),
    )

    def boom(*a: object, **k: object) -> None:  # must never be reached
        raise AssertionError("build_graph called without analysis contract")

    monkeypatch.setattr(lineage_module, "build_graph", boom)
    with pytest.raises(ValueError, match="lineage requires an analysis contract"):
        lineage_module.run("test_hypothesis", "test")


def test_lineage_run_writes_graph_artifacts_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    lineage_dir = tmp_path / "lineage"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(lineage_module, "LINEAGE_DIR", lineage_dir)
    monkeypatch.setattr(lineage_module, "REPORTS_DIR", reports_dir)
    _seed_records(lineage_dir, tmp_path)

    lineage_module.run("test_hypothesis", "test")

    out_dir = reports_dir / "lineage"
    graph_json = out_dir / "graph.json"
    graph_md = out_dir / "graph.md"
    assert graph_json.exists()
    assert graph_md.exists()

    doc = json.loads(graph_json.read_text(encoding="utf-8"))
    node_ids = {n["id"] for n in doc["nodes"]}
    assert "analysis:test_hypothesis" in node_ids
    assert "dataset:test" in node_ids  # derivation held — dataset in scope

    md = graph_md.read_text(encoding="utf-8")
    assert md.startswith("# Lineage — test_hypothesis")
    assert "Read arrows left-to-right" in md
    assert "```mermaid" in md and "```" in md.split("```mermaid")[1]

    out = capsys.readouterr().out
    assert out.startswith("goal: ")
    assert "stage:" in out
    assert "open decisions:" in out
    assert "resolved decisions:" in out


def test_lineage_run_undecodable_derivation_raises_scope_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No records at all: the analysis exists but is not derived from the
    dataset — the orchestration surfaces the scope error loudly."""
    monkeypatch.setattr(lineage_module, "LINEAGE_DIR", tmp_path / "lineage")
    monkeypatch.setattr(lineage_module, "REPORTS_DIR", tmp_path / "reports")
    with pytest.raises(ValueError, match="not derived from dataset"):
        lineage_module.run("test_hypothesis", "test")
