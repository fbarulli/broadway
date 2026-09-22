"""Canvas spec builders — synthetic only (no node, no skill install)."""

from __future__ import annotations

import importlib.util
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


def test_discover_series_lists_numbered_stems(tmp_path) -> None:
    spec = importlib.util.spec_from_file_location(
        "render_canvas_maps", "scripts/render_canvas_maps.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    (tmp_path / "alpha").mkdir()
    (tmp_path / "alpha" / "01_a.py").write_text('"""a"""', encoding="utf-8")
    (tmp_path / "alpha" / "notes.txt").write_text("x", encoding="utf-8")
    (tmp_path / "empty").mkdir()
    assert module._discover_series(tmp_path) == {"alpha": ["01_a"]}


def test_dashboard_launcher_enables_reload(monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location(
        "render_canvas_maps_reload", "scripts/render_canvas_maps.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    import uvicorn

    call = {}
    monkeypatch.setattr(uvicorn, "run", lambda target, **kwargs: call.update(target=target, **kwargs))
    monkeypatch.delenv("BROADWAY_EXPERIMENTS_ROOT", raising=False)
    monkeypatch.delenv("BROADWAY_OBSERVATIONS_DIR", raising=False)
    monkeypatch.delenv("BROADWAY_DIAGRAMS_DIR", raising=False)

    module.serve_dashboard(8123)

    assert call["target"] == "broadway.reports.experiments_dashboard:app"
    assert call["host"] == "127.0.0.1"
    assert call["port"] == 8123
    assert call["reload"] is True
    assert str(module.REPO_ROOT / "src") in call["reload_dirs"]


def test_dashboard_launcher_mounts_configured_results(monkeypatch, tmp_path) -> None:
    """Late env (maps built before serve) still serves /results.

    Regression: the dashboard read BROADWAY_* at first import, so building
    canvas specs before serve_dashboard() left /results unmounted (404).
    """
    from broadway.reports import experiments_dashboard as ui

    results = tmp_path / "results"
    results.mkdir()
    (results / "ping.txt").write_text("pong", encoding="utf-8")
    monkeypatch.setenv("BROADWAY_EXPERIMENTS_ROOT", str(tmp_path))
    monkeypatch.setenv("BROADWAY_DIAGRAMS_DIR", str(tmp_path / "missing-diagrams"))

    ui.configure_from_env()

    assert any(getattr(route, "path", "") == "/results" for route in ui.app.routes)
    routes_before = len(ui.app.routes)
    ui.configure_from_env()
    assert len(ui.app.routes) == routes_before


def test_dashboard_story_spec_chains_steps_with_artifacts() -> None:
    spec = canvas.dashboard_story_spec(
        "alpha",
        [("01_a", "Load it", ["01_a.csv", "01_a.png"]), ("02_b", "", [])],
    )
    assert spec["title"] == "alpha story"
    assert "01_a" in [n["id"] for n in spec["nodes"]]
    edges = [(e["from"], e["to"], e["label"]) for e in spec["edges"]]
    assert ("01_a", "02_b", "next") in edges
    assert ("01_a", "01_a/01_a.png", "produces") in edges
