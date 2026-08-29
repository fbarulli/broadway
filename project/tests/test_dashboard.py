from __future__ import annotations

import os
from types import SimpleNamespace

import project.dashboard as dashboard


def test_main_wires_project_paths_into_generic_dashboard(monkeypatch, tmp_path) -> None:
    paths = SimpleNamespace(
        experiments=tmp_path / "experiments",
        observations=tmp_path / "observations",
    )
    run_calls: list[tuple[object, str, int]] = []
    monkeypatch.delenv("BROADWAY_EXPERIMENTS_ROOT", raising=False)
    monkeypatch.delenv("BROADWAY_OBSERVATIONS_DIR", raising=False)
    monkeypatch.setattr(dashboard, "load_project_paths", lambda: paths)
    monkeypatch.setattr(
        dashboard.uvicorn,
        "run",
        lambda app, host, port: run_calls.append((app, host, port)),
    )

    dashboard.main()

    assert run_calls[0][1:] == ("127.0.0.1", 8000)
    assert run_calls[0][0].title == "FastAPI"
    assert paths.experiments.as_posix() == os.environ["BROADWAY_EXPERIMENTS_ROOT"]
    assert paths.observations.as_posix() == os.environ["BROADWAY_OBSERVATIONS_DIR"]
