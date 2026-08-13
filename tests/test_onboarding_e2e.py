"""End-to-end acceptance test: onboard + run a totally non-taxi dataset.

Proves the generic pipeline (init -> etl -> contracts -> baseline -> features
-> train -> evaluate) works for an arbitrary CSV without touching any taxi or
project-specific code.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from broadway.config import loader
from broadway.config.loader import load_config, resolve_full_steps
from broadway.discover import module as discover_module
from broadway.lineage import records
from broadway.onboard import module as onboard_module
from broadway.pipeline import run as run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_init_and_run_non_taxi_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. Isolate configs, artifacts, and lineage into tmp_path.
    configs_dir = tmp_path / "configs"
    shutil.copytree(REPO_ROOT / "configs" / "environment", configs_dir / "environment")
    shutil.copytree(REPO_ROOT / "configs" / "step", configs_dir / "step")
    shutil.copytree(REPO_ROOT / "configs" / "flow", configs_dir / "flow")

    monkeypatch.setattr(loader, "CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(onboard_module, "CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(discover_module, "CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(onboard_module, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(discover_module, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")

    # 2. Write a synthetic non-taxi CSV.
    rng = np.random.default_rng(42)
    n = 200
    listed_at = pd.date_range("2024-01-01", periods=n, freq="h").strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    rooms = rng.integers(1, 7, n)
    area = np.round(rng.uniform(30.0, 200.0, n), 1)
    neighborhood = rng.choice(["downtown", "suburbs", "historic", "riverside"], n)
    price = np.round(5000.0 * rooms + 1000.0 * area + rng.normal(0, 5000, n), 2)
    df = pd.DataFrame(
        {
            "id": list(range(n)),
            "listed_at": listed_at,
            "rooms": rooms,
            "area": area,
            "neighborhood": neighborhood,
            "price": price,
        }
    )
    csv_path = tmp_path / "houses.csv"
    df.to_csv(csv_path, index=False)

    # 3. Scaffold the dataset non-interactively.
    onboard_module.init(
        str(csv_path),
        "houses",
        target="price",
        task="regression",
        datetime_columns=["listed_at"],
        ignore_columns=["id"],
        split_column=None,
        mode="prediction",
        goal="predict house price",
        row_definition="one house listing",
        decision_moment="at listing time",
        available_info=["rooms", "area", "neighborhood", "listed_at"],
        leakage_notes=[],
        success_criterion="beat mean baseline",
    )
    assert (configs_dir / "dataset" / "houses.yaml").exists()
    assert (configs_dir / "analysis" / "houses.yaml").exists()
    assert (configs_dir / "experiment" / "houses.yaml").exists()

    # 4. Load the full config and redirect all writes into tmp_path.
    cfg = load_config("full", dataset="houses", analysis="houses", experiment="houses")
    cfg = cfg.model_copy(
        update={
            "environment": cfg.environment.model_copy(
                update={
                    "data_dir": str(tmp_path / "data"),
                    "mlflow_tracking_uri": str(tmp_path / "mlruns"),
                }
            )
        }
    )
    for attr, dirname in (
        ("baseline", "baseline"),
        ("train", "training"),
        ("evaluate", "evaluation"),
        ("stats", "stats"),
        ("causal", "causal"),
    ):
        step_cfg = getattr(cfg, attr)
        if step_cfg is not None:
            setattr(
                cfg,
                attr,
                step_cfg.model_copy(
                    update={"output_dir": str(tmp_path / "artifacts" / dirname)}
                ),
            )

    # 5. Run the full flow (skip the optional eda report; discover is skipped
    #    inside pipeline.run anyway).
    steps = resolve_full_steps(cfg)
    run_pipeline(cfg, [s for s in steps if s != "eda"])

    # 6. Assert every step produced its artifact and lineage sidecar.
    assert (tmp_path / "artifacts" / "baseline" / "baseline.json").exists()
    assert (tmp_path / "artifacts" / "training" / "training_result.json").exists()
    assert (tmp_path / "artifacts" / "evaluation" / "metrics.json").exists()

    records_dir = tmp_path / "lineage" / "records"
    for kind in ("profile", "baseline", "training", "evaluation"):
        assert (records_dir / f"{kind}_houses.json").exists()
