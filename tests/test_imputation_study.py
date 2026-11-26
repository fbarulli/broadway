"""Public behavior of the measured imputation/scaling impact study.

The study reads the persisted feature splits, injects MCAR churn into TRAIN
only, and scores every variant on the same clean holdout; these tests pin the
delta table it produces plus its fail-loud configuration guards.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from broadway.config.loader import load_config
from broadway.config.schema import ImputationStudyConfig, PipelineConfig
from broadway.selection.imputation import (
    ImpactRow,
    ImpactTable,
    ImputationImpactStudy,
)


def _config(tmp_path: Path, **study: object) -> PipelineConfig:
    cfg = load_config("features", dataset="test", experiment="baseline")
    assert cfg.features is not None and cfg.etl is not None
    values: dict[str, object] = {
        "strategies": ["median", "mean"],
        "scalings": ["none", "standard"],
        "metric_direction": "minimize",
        "sample_size": 10,
        "knn_neighbors": 2,
        "missing_injection": {"rate": 0.2, "columns": ["feature_1"]},
        "output_dir": str(tmp_path / "reports"),
        "output_file": "imputation_impact.json",
    }
    values.update(study)
    return cfg.model_copy(
        update={
            "environment": cfg.environment.model_copy(update={"data_dir": str(tmp_path)}),
            "features": cfg.features.model_copy(
                update={"imputation_study": ImputationStudyConfig(**values)}
            ),
        }
    )


def _write_splits(cfg: PipelineConfig, rows: int = 20) -> None:
    processed = Path(cfg.environment.data_dir) / cfg.environment.processed_subdir
    processed.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "feature_1": np.arange(rows, dtype=float) + 1.0,
            "feature_2": np.arange(rows, dtype=float) * 2.0,
            "target": np.arange(rows, dtype=float) * 3.0 + 10.0,
        }
    )
    assert cfg.etl is not None
    frame.to_parquet(processed / cfg.etl.train_features_file, index=False)
    frame.to_parquet(processed / cfg.etl.val_features_file, index=False)


def test_study_measures_every_variant_against_complete_case(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _write_splits(cfg)

    table = ImputationImpactStudy(cfg).run()

    assert table.metric == "rmse"
    assert len(table.rows) == 6
    assert [row.strategy for row in table.rows[:2]] == ["complete_case", "complete_case"]
    assert all(row.delta_vs_baseline == 0.0 for row in table.rows[:2])
    assert all(row.train_rows_kept <= 10 for row in table.rows)
    assert (tmp_path / "reports" / "imputation_impact.json").exists()
    assert "delta_vs_baseline" in table.to_frame().columns


def test_study_run_requires_the_persisted_feature_splits(tmp_path: Path) -> None:
    cfg = _config(tmp_path)

    with pytest.raises(FileNotFoundError, match="requires the feature split"):
        ImputationImpactStudy(cfg).run()


def test_study_rejects_injection_columns_outside_the_model_surface(tmp_path: Path) -> None:
    cfg = _config(tmp_path, missing_injection={"rate": 0.2, "columns": ["ghost"]})
    _write_splits(cfg)

    with pytest.raises(ValueError, match="not in the model surface"):
        ImputationImpactStudy(cfg).run()


def test_study_fails_loud_when_complete_case_leaves_no_rows(tmp_path: Path) -> None:
    cfg = _config(tmp_path, missing_injection={"rate": 0.999, "columns": ["feature_1"]})
    _write_splits(cfg)

    with pytest.raises(ValueError, match="leaves no rows"):
        ImputationImpactStudy(cfg).run()


def test_study_rejects_a_metric_compute_metrics_does_not_produce(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    assert cfg.experiment is not None
    cfg = cfg.model_copy(
        update={
            "experiment": cfg.experiment.model_copy(update={"target_metric": "not_a_metric"})
        }
    )
    _write_splits(cfg)

    with pytest.raises(ValueError, match="not produced by compute_metrics"):
        ImputationImpactStudy(cfg).run()


def test_study_constructor_requires_its_config_sections(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    assert cfg.features is not None

    with pytest.raises(ValueError, match="requires dataset, experiment, and features"):
        ImputationImpactStudy(cfg.model_copy(update={"features": None}))
    with pytest.raises(ValueError, match="requires the etl config"):
        ImputationImpactStudy(cfg.model_copy(update={"etl": None}))
    with pytest.raises(ValueError, match="no imputation_study block"):
        ImputationImpactStudy(
            cfg.model_copy(
                update={"features": cfg.features.model_copy(update={"imputation_study": None})}
            )
        )


def _row(strategy: str, scaling: str, delta: float) -> ImpactRow:
    return ImpactRow(
        strategy=strategy,
        scaling=scaling,
        value=1.0 + delta,
        delta_vs_baseline=delta,
        train_rows_kept=10,
        train_rows_dropped=0,
        val_rows_kept=5,
        val_rows_dropped=0,
    )


def test_impact_table_requires_rows_and_a_baseline_row() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        ImpactTable.assemble(
            metric="rmse", metric_direction="minimize", baseline="complete_case", rows=[]
        )
    with pytest.raises(ValueError, match="has no baseline"):
        ImpactTable.assemble(
            metric="rmse",
            metric_direction="minimize",
            baseline="complete_case",
            rows=[_row("median", "none", 0.0)],
        )
