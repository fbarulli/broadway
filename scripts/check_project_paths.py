"""Fail closed when project experiment paths escape the project contract."""

from __future__ import annotations

import re
from pathlib import Path

from project.paths import load_project_paths

REPO = Path(__file__).resolve().parents[1]
LEGACY_PATH = re.compile(
    r"(?<!project/)configs/(?:experiments/|dataset/taxi\.yaml|analysis/taxi(?:_causal|_hypothesis)?\.yaml|experiment/taxi\.yaml|project/taxi\.yaml|sample/(?:fare_prediction_1m|taxi_diagnostic|taxi_estimation)\.yaml|slice/(?:airport|distance_duration_inconsistent|passenger_out_of_range|pre_2024)\.yaml)|(?<!project/)(?<!config/)(?:experiments/(?:results|mlflow|fare_prediction|more_modeling|multivariate|polynomial_regression_et_all|univariate)|/app/experiments/)"
)
RETIRED_TAXI_CONFIGS = (
    "configs/dataset/taxi.yaml",
    "configs/analysis/taxi.yaml",
    "configs/analysis/taxi_causal.yaml",
    "configs/analysis/taxi_hypothesis.yaml",
    "configs/experiment/taxi.yaml",
    "configs/project/taxi.yaml",
    "configs/sample/fare_prediction_1m.yaml",
    "configs/sample/taxi_diagnostic.yaml",
    "configs/sample/taxi_estimation.yaml",
    "configs/slice/airport.yaml",
    "configs/slice/distance_duration_inconsistent.yaml",
    "configs/slice/passenger_out_of_range.yaml",
    "configs/slice/pre_2024.yaml",
)
CHECKED_SUFFIXES = {".py", ".sh", ".yaml", ".yml"}
EXCLUDED_DIRS = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}


def runtime_files(root: Path) -> list[Path]:
    """Return working-tree runtime/config files that can contain filesystem paths."""
    files = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if EXCLUDED_DIRS.intersection(path.parts) or not path.is_file():
            continue
        if relative in {Path("scripts/check_project_paths.py"), Path("scripts/main_day_sync.sh")}:
            continue
        if relative.parts[:2] == ("project", "config"):
            continue
        if relative.name != ".gitignore" and relative.parts[0] not in {"src", "project", "scripts", "k8s", ".github"}:
            continue
        if path.suffix in CHECKED_SUFFIXES or path.name.startswith("Dockerfile") or path.name == ".gitignore":
            files.append(path)
    return files


def legacy_path_references(root: Path, paths: list[Path]) -> list[str]:
    """Return tracked runtime references to the retired root experiments tree."""
    return [str(path.relative_to(root)) for path in paths if LEGACY_PATH.search(path.read_text(encoding="utf-8"))]


def main() -> None:
    """Check layout targets and reject retired root-scoped project paths."""
    paths = load_project_paths()
    missing = [
        str(path)
        for path in (paths.config, paths.experiments, paths.experiment_configs)
        if not path.is_dir()
    ]
    legacy = legacy_path_references(REPO, runtime_files(REPO))
    retired_configs = [path for path in RETIRED_TAXI_CONFIGS if (REPO / path).exists()]
    if missing or legacy or retired_configs or (REPO / "experiments").exists():
        raise SystemExit(
            f"PROJECT PATH ERROR: missing={missing}; legacy={legacy}; "
            f"retired_configs={retired_configs}"
        )
    print(f"PROJECT PATHS OK: experiments={paths.experiments}; configs={paths.experiment_configs}")


if __name__ == "__main__":
    main()
