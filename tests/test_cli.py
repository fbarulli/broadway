from __future__ import annotations

import csv as csv_module
import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str, **kwargs: object) -> subprocess.CompletedProcess[str]:
    # timeout=150: a nested `uv run` must fail fast, never hang the suite
    # (uv editable-rebuild probe stall incident 2026-08-23).
    return subprocess.run(
        ["uv", "run", "ds-pipeline", *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=150,
        **kwargs,
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv_module.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _copy_viz_config(configs_dir: Path) -> None:
    (configs_dir / "step").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO_ROOT / "configs" / "step" / "viz.yaml", configs_dir / "step" / "viz.yaml")


class TestDiscoverCLI:
    def test_discover_parses_and_generates_yaml(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "test_data.csv"
        _write_csv(
            csv_path,
            [
                {"area": 1.5, "duration_min": 10.0, "rooms": 1, "price": 15.0},
                {"area": 3.2, "duration_min": 22.0, "rooms": 2, "price": 30.0},
                {"area": 0.8, "duration_min":  8.0, "rooms": 1, "price":  9.0},
            ],
        )

        configs_dir = tmp_path / "configs"
        dataset_dir = configs_dir / "dataset"
        _copy_viz_config(configs_dir)
        env = {
            **os.environ,
            "BROADWAY_CONFIGS_DIR": str(configs_dir),
            "BROADWAY_DATASET_DIR": "dataset",
            "BROADWAY_ARTIFACTS_DIR": str(tmp_path / "artifacts"),
            "BROADWAY_REPORTS_DIR": str(tmp_path / "reports"),
            "BROADWAY_LINEAGE_DIR": str(tmp_path / "lineage"),
        }

        result = _run(
            "discover",
            "--csv", str(csv_path),
            "--target", "price",
            "--task", "regression",
            env=env,
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"
        yaml_path = dataset_dir / f"{csv_path.stem}.yaml"
        assert yaml_path.exists()
        assert (tmp_path / "artifacts" / "discover" / "profile.json").exists()

    def test_discover_missing_csv_raises(self, tmp_path: Path) -> None:
        result = _run(
            "discover",
            "--csv", "/nonexistent/path.csv",
            "--target", "price",
            "--task", "regression",
        )
        assert result.returncode != 0
        assert "usage:" not in result.stderr.lower()

    def test_discover_with_optional_args(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        _write_csv(
            csv_path,
            [
                {"a": 1, "b": 2, "c": 3, "dt": "2024-01-01"},
                {"a": 4, "b": 5, "c": 6, "dt": "2024-01-02"},
            ],
        )

        configs_dir = tmp_path / "configs"
        dataset_dir = configs_dir / "dataset"
        _copy_viz_config(configs_dir)
        env = {
            **os.environ,
            "BROADWAY_CONFIGS_DIR": str(configs_dir),
            "BROADWAY_DATASET_DIR": "dataset",
            "BROADWAY_ARTIFACTS_DIR": str(tmp_path / "artifacts"),
            "BROADWAY_REPORTS_DIR": str(tmp_path / "reports"),
            "BROADWAY_LINEAGE_DIR": str(tmp_path / "lineage"),
        }

        real = Path("artifacts/discover/profile.json")
        before = real.read_bytes() if real.exists() else None

        result = _run(
            "discover",
            "--csv", str(csv_path),
            "--target", "a",
            "--task", "regression",
            "--datetime-column", "dt",
            "--ignore-columns", "c",
            env=env,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert dataset_dir.joinpath(f"{csv_path.stem}.yaml").exists()
        assert (tmp_path / "artifacts" / "discover" / "profile.json").exists()
        if before is not None:
            assert real.read_bytes() == before


class TestTrainCLI:
    def test_train_without_analysis_exits_1_naming_contract_requirement(self) -> None:
        # train requires a prediction-mode analysis contract: without --analysis
        # the step crashes (exit 1, not argparse exit 2) and names the missing
        # requirement. The audit (T-BUG-3) found the old test passed while the
        # command exited 1 — this test pins that contract explicitly.
        result = _run("train", "--dataset", "test", "--experiment", "baseline")
        assert result.returncode == 1, (
            f"expected exit 1 naming the missing analysis contract, "
            f"got {result.returncode}: {result.stderr}"
        )
        assert "requires an analysis contract (--analysis)" in result.stderr, (
            f"stderr should name the missing analysis contract: {result.stderr}"
        )
        assert "prediction" in result.stderr, (
            f"stderr should name the required mode 'prediction': {result.stderr}"
        )

    def test_train_without_dataset_exits_1_naming_missing_config(self) -> None:
        # Without --dataset no dataset config is merged, and train fails (exit 1,
        # not argparse exit 2) naming the missing config sections rather than
        # dispatching. Renamed from "still_dispatches": the command crashes.
        result = _run("train", "--experiment", "baseline")
        assert result.returncode == 1, (
            f"expected exit 1 naming the missing train config, "
            f"got {result.returncode}: {result.stderr}"
        )
        assert "requires dataset, experiment, train, and etl config" in result.stderr, (
            f"stderr should name the missing config sections: {result.stderr}"
        )


class TestWalkthroughCLI:
    def test_walkthrough_requires_dataset(self) -> None:
        result = _run("walkthrough", "--analysis", "test_hypothesis")
        assert result.returncode == 2
        assert "required" in result.stderr.lower()


class TestMissingSubcommand:
    def test_no_subcommand_raises_error(self) -> None:
        result = _run()
        assert result.returncode == 2
        assert "required" in result.stderr.lower()


class TestInvalidStep:
    def test_invalid_step_raises_error(self) -> None:
        result = _run("bogus")
        assert result.returncode == 2
        assert "invalid" in result.stderr.lower()
