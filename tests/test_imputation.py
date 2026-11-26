"""Public behavior of imputation, scaling, and the target-transform recipe.

One focused test per behavior: fit-on-train/apply-to-val semantics (including
the unseen-category fill), train-only scaling statistics, the difference and
BoxCox round trip through the recipe, and the impact table's ordering. All
frames are generated in-memory — no project data or config is read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from broadway.config.loader import load_config
from broadway.config.schema import PipelineConfig, PreprocessingStepConfig
from broadway.features.recipe import build_pipeline, build_target_transforms
from broadway.selection.imputation import ImpactRow, ImpactTable


def _recipe_cfg(
    preprocessing: list[dict[str, object]],
    target_transforms: list[dict[str, object]] | None = None,
) -> PipelineConfig:
    cfg = load_config("train", dataset="test", experiment="baseline")
    assert cfg.experiment is not None
    experiment = cfg.experiment.model_copy(
        update={
            "preprocessing": [PreprocessingStepConfig(**step) for step in preprocessing],
            "target_transforms": [
                PreprocessingStepConfig(**step) for step in (target_transforms or [])
            ],
        }
    )
    return cfg.model_copy(update={"experiment": experiment})


def test_imputer_fits_on_train_and_fills_val_including_unseen_categories() -> None:
    """Val missing values take the TRAIN statistic; an unseen category takes the fill."""
    cfg = _recipe_cfg(
        [
            {
                "type": "imputation",
                "columns": ["amount", "city"],
                "params": {"strategy": "mode"},
            }
        ]
    )
    train = pd.DataFrame(
        {"amount": [1.0, np.nan, 3.0, 1.0], "city": ["NYC", None, "NYC", "LA"]}
    )
    val = pd.DataFrame({"amount": [np.nan, 100.0], "city": ["PARIS", None]})

    pipeline = build_pipeline(cfg)
    pipeline.fit(train)
    out = pipeline.transform(val)

    # Train mode is 1.0; a val value the fit never saw is data, not a category.
    assert out["amount"].tolist() == [1.0, 100.0]
    # Unseen category and NaN both resolve to the train mode.
    assert out["city"].tolist() == ["NYC", "NYC"]


def test_scaler_uses_train_statistics_only() -> None:
    cfg = _recipe_cfg(
        [{"type": "scaling", "columns": ["amount"], "params": {"strategy": "standard"}}]
    )
    train = pd.DataFrame({"amount": [0.0, 2.0]})
    val = pd.DataFrame({"amount": [10.0, 20.0]})

    pipeline = build_pipeline(cfg)
    pipeline.fit(train)
    out = pipeline.transform(val)

    # mean=1, std=1 from TRAIN; val is transformed with those, never its own.
    assert out["amount"].tolist() == pytest.approx([9.0, 19.0])


def test_difference_round_trips_through_the_recipe() -> None:
    cfg = _recipe_cfg(
        [],
        [{"type": "difference", "columns": ["target"], "params": {"order": 2}}],
    )
    series = pd.Series([10.0, 12.0, 15.0, 19.0, 24.0, 30.0], name="target")

    chain = build_target_transforms(cfg)
    chain.fit(series)
    differenced = chain.transform(series)

    assert chain.n_dropped == 2
    assert len(differenced) == len(series) - 2
    restored = chain.inverse_transform(differenced)
    pd.testing.assert_series_equal(restored, series, check_names=False)


def test_boxcox_round_trips_through_the_recipe() -> None:
    cfg = _recipe_cfg([], [{"type": "boxcox", "columns": ["target"]}])
    series = pd.Series([2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0], name="target")

    chain = build_target_transforms(cfg)
    chain.fit(series)
    transformed = chain.transform(series)

    assert (transformed.diff().dropna() > 0).all()  # still monotone
    restored = chain.inverse_transform(transformed)
    pd.testing.assert_series_equal(restored, series, check_names=False, rtol=1e-10)


def test_impact_table_orders_complete_case_first_then_best_delta() -> None:
    def row(strategy: str, scaling: str, delta: float) -> ImpactRow:
        return ImpactRow(
            strategy=strategy,
            scaling=scaling,
            value=1.0 + delta,
            delta_vs_baseline=delta,
            train_rows_kept=100,
            train_rows_dropped=10,
            val_rows_kept=50,
            val_rows_dropped=0,
        )

    rows = [
        row("knn", "none", -0.5),
        row("median", "standard", 0.2),
        row("mean", "none", 0.1),
        row("complete_case", "none", 0.0),
        row("complete_case", "standard", 0.0),
    ]
    table = ImpactTable.assemble(
        metric="rmse", metric_direction="minimize", baseline="complete_case", rows=rows
    )

    assert [(r.strategy, r.scaling) for r in table.rows] == [
        ("complete_case", "none"),
        ("complete_case", "standard"),
        ("knn", "none"),
        ("mean", "none"),
        ("median", "standard"),
    ]
    assert "rmse" in table.to_frame().columns
