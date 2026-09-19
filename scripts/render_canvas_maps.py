#!/usr/bin/env python3
"""Render canvas maps from live project surfaces, then serve the dashboard.

Single entry point: rebuilds every map in diagrams/ (specs via
src/broadway/reports/canvas.py builders, .tldr via the skill generator),
prints the URLs, and starts the web app as its final act — a running
dashboard is the expected output of this script, not files alone.

Usage:
    bash scripts/uv.sh run --extra dev python scripts/render_canvas_maps.py [--no-serve] [--port 8000]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from broadway.reports import canvas


def _discover_series(experiments_root: Path) -> dict[str, list[str]]:
    """Map each experiment series dir to its ordered numbered step stems."""
    series: dict[str, list[str]] = {}
    if not experiments_root.is_dir():
        return series
    for child in sorted(p for p in experiments_root.iterdir() if p.is_dir()):
        stems = sorted(
            p.stem for p in child.glob("*.py")
            if p.stem[:1].isdigit()
        )
        if stems:
            series[child.name] = stems
    return series


def build_all_maps(diagrams_dir: Path, experiments_root: Path) -> list[str]:
    """(Re)build every map spec + .tldr. Returns built map names."""
    built: list[str] = []
    series = _discover_series(experiments_root)
    if series:
        spec = canvas.experiment_series_spec(series)
        canvas.write_spec(spec, diagrams_dir / "results_producers.json")
        built.append("results_producers")
    for track, stems in series.items():
        steps = _dashboard_story_items(track, stems)
        story = canvas.dashboard_story_spec(track, steps)
        canvas.write_spec(story, diagrams_dir / f"story_{track}.json")
        built.append(f"story_{track}")
    for name in built:
        canvas.emit_tldr(diagrams_dir / f"{name}.json", diagrams_dir / f"{name}.tldr")
    return built


def _dashboard_story_items(track: str, stems: list[str]) -> list[tuple[str, str, list[str]]]:
    """(stem, question, artifact names) per step, from live dashboard data."""
    from broadway.reports import experiments_dashboard as ui

    root = REPO_ROOT / "project" / "experiments"
    if root.is_dir():
        ui.EXPERIMENTS_ROOT = root
    items: list[tuple[str, str, list[str]]] = []
    for stem in stems:
        try:
            script = ui._script_for(stem, track)
            question = ui._question_from_docstring(script) if script else ""
            artifacts = [f.name for f in ui._result_files(stem, track)][:6]
        except OSError:
            question, artifacts = "", []
        items.append((stem, question, artifacts))
    return items


def serve_dashboard(port: int) -> None:
    """Start the hosted dashboard (blocking): maps + results + canvas."""
    from project.paths import load_project_paths

    paths = load_project_paths()
    os.environ.setdefault("BROADWAY_EXPERIMENTS_ROOT", str(paths.experiments))
    os.environ.setdefault("BROADWAY_OBSERVATIONS_DIR", str(paths.observations))
    os.environ.setdefault("BROADWAY_DIAGRAMS_DIR", str(REPO_ROOT / "diagrams"))
    import uvicorn

    from broadway.reports.experiments_dashboard import app

    print(f"dashboard serving at http://127.0.0.1:{port}  (canvas: /canvas)")
    uvicorn.run(app, host="127.0.0.1", port=port)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="render_canvas_maps")
    parser.add_argument("--no-serve", action="store_true", help="rebuild maps only")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    diagrams_dir = REPO_ROOT / "diagrams"
    diagrams_dir.mkdir(parents=True, exist_ok=True)
    experiments_root = REPO_ROOT / "project" / "experiments"
    try:
        built = build_all_maps(diagrams_dir, experiments_root)
    except RuntimeError as exc:
        print(f"map build skipped: {exc}", file=sys.stderr)
        built = []
    print(f"maps ready: {', '.join(built) if built else 'specs only (no renderer)'}")
    if args.no_serve:
        return 0
    serve_dashboard(args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
