"""Canvas spec builders — synthetic only (no node, no skill install)."""

from __future__ import annotations

import json

import pytest

from broadway.reports import canvas


def test_new_spec_shape() -> None:
    spec = canvas.new_spec("Demo")
    assert spec == {"title": "Demo", "nodes": [], "edges": []}


def test_series_spec_two_tracks() -> None:
    spec = canvas.experiment_series_spec({"alpha": ["01_a", "02_b"], "beta": ["01_c"]})
    ids = [n["id"] for n in spec["nodes"]]
    assert "dashboard" in ids and "track-alpha" in ids and "alpha/01_a" in ids
    assert len(spec["edges"]) == 2 * 3


def test_blast_radius_spec_fan_out() -> None:
    spec = canvas.blast_radius_spec("Foo", [("bar()", "x.py:1"), ("baz()", "y.py:2")])
    assert spec["title"] == "Foo blast radius"
    assert len([n for n in spec["nodes"] if n["group"] == "consumers"]) == 2


def test_write_spec_round_trip(tmp_path) -> None:
    spec = canvas.new_spec("T")
    canvas.add_node(spec, "a", "A label", "g")
    canvas.add_edge(spec, "a", "a", "self")
    path = tmp_path / "spec.json"
    canvas.write_spec(spec, path)
    assert json.loads(path.read_text(encoding="utf-8"))["nodes"][0]["id"] == "a"


def test_emit_tldr_requires_node_and_skill(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(canvas.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="node is required"):
        canvas.emit_tldr(tmp_path / "s.json", tmp_path / "o.tldr")
