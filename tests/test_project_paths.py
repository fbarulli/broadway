"""Pins for the project-owned experiment path contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_project_paths_are_project_relative_and_results_are_derived() -> None:
    paths = _load_module(REPO / "project" / "paths.py", "project_paths_test").load_project_paths()
    assert paths.config == REPO / "project" / "config"
    assert paths.experiments == REPO / "project" / "experiments"
    assert paths.experiment_configs == REPO / "project" / "config" / "experiments"
    assert paths.results == paths.experiments / "results"


def test_project_paths_reject_a_layout_escape(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    (root / "config" / "layout.yaml").write_text(
        "config: config\nexperiments: ../outside\nexperiment_configs: config/experiments\nobservations: artifacts/experiments\n",
        encoding="utf-8",
    )
    paths_module = _load_module(REPO / "project" / "paths.py", "project_paths_escape_test")
    with pytest.raises(ValueError, match="must remain under"):
        paths_module.load_project_paths(root)


def test_path_checker_detects_a_retired_root_reference(tmp_path: Path) -> None:
    checker = _load_module(REPO / "scripts" / "check_project_paths.py", "project_paths_checker_test")
    source = tmp_path / "consumer.py"
    source.write_text('Path("experiments/multivariate/01_categorical_breakdown.py")', encoding="utf-8")
    assert checker.legacy_path_references(tmp_path, [source]) == ["consumer.py"]


def test_path_checker_detects_a_retired_config_root(tmp_path: Path) -> None:
    checker = _load_module(REPO / "scripts" / "check_project_paths.py", "project_paths_config_checker_test")
    source = tmp_path / "consumer.py"
    source.write_text('Path("configs/experiments/mlflow.yaml")', encoding="utf-8")
    assert checker.legacy_path_references(tmp_path, [source]) == ["consumer.py"]


def test_path_checker_detects_a_retired_taxi_config(tmp_path: Path) -> None:
    checker = _load_module(REPO / "scripts" / "check_project_paths.py", "project_paths_taxi_checker_test")
    source = tmp_path / "consumer.py"
    source.write_text('Path("configs/dataset/taxi.yaml")', encoding="utf-8")
    assert checker.legacy_path_references(tmp_path, [source]) == ["consumer.py"]


def test_worker_image_copies_the_project_path_import_closure() -> None:
    base = (REPO / "k8s" / "optuna" / "Dockerfile.base").read_text(encoding="utf-8")
    worker = (REPO / "k8s" / "optuna" / "Dockerfile.worker").read_text(encoding="utf-8")
    assert "COPY project/ /app/project/" in base
    assert "FROM broadway-base:latest" in worker


def test_main_day_sync_removes_project_surfaces_and_keeps_generic_sample() -> None:
    sync = (REPO / "scripts" / "main_day_sync.sh").read_text(encoding="utf-8")
    assert "git rm -r --ignore-unmatch project experiments configs/project configs/experiments" in sync
    assert "configs/sample/fare_prediction_1m.yaml" not in sync
    assert "configs/sample/demo.yaml" in sync


def test_ci_smoke_uses_the_moved_project_dispatcher() -> None:
    workflow = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "python project/experiments.py verify" in workflow
    assert "python experiments.py verify" not in workflow


def test_reusable_source_never_imports_the_project_layer() -> None:
    offenders = [
        path.relative_to(REPO)
        for path in (REPO / "src" / "broadway").rglob("*.py")
        if "from project" in path.read_text(encoding="utf-8")
        or "import project" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
