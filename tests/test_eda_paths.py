"""Public behavior of the EDA step along its non-happy paths.

The runners read the existing ETL parquets (never re-run etl): a missing val
split or canonical frame is a recorded skip, sampling keeps the report honest,
and the HTML report stays renderable when there is nothing to plot.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from broadway.config.loader import load_config
from broadway.config.schema import PipelineConfig, TaskType
from broadway.eda import module


def _config(tmp_path: Path, *, sample_size: int = 0, plot_format: str = "png") -> PipelineConfig:
    cfg = load_config("eda", dataset="test", experiment="baseline")
    assert cfg.eda is not None and cfg.etl is not None
    return cfg.model_copy(
        update={
            "environment": cfg.environment.model_copy(update={"data_dir": str(tmp_path)}),
            "eda": cfg.eda.model_copy(
                update={
                    "sample_size": sample_size,
                    "plot_format": plot_format,
                    "output_dir": str(tmp_path / "reports"),
                }
            ),
        }
    )


def _write(cfg: PipelineConfig, name: str, frame: pd.DataFrame) -> None:
    processed = Path(cfg.environment.data_dir) / cfg.environment.processed_subdir
    processed.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(processed / name, index=False)


def test_eda_profiles_the_train_split_when_val_and_canonical_are_absent(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path, sample_size=4, plot_format="svg")
    assert cfg.dataset is not None and cfg.etl is not None
    cfg = cfg.model_copy(
        update={
            "dataset": cfg.dataset.model_copy(
                update={
                    "task": TaskType.CLASSIFICATION,
                    "exclude_from_profiling": ["noise"],
                }
            )
        }
    )
    frame = pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, None],
            "feature_2": [10] * 8,
            "feature_3": ["a", "a", "b", "b", "c", "c", "a", None],
            "target": [0, 0, 0, 0, 0, 0, 0, 1],
            "flag": [True, False, True, False, True, False, True, False],
            "when": pd.to_datetime([f"2024-01-0{day}" for day in range(1, 9)]),
            "noise": list(range(100, 108)),
        }
    )
    _write(cfg, cfg.etl.train_file, frame)

    module.run(cfg)

    report = cfg.eda.output_path
    assert report.exists()
    html = report.read_text(encoding="utf-8")
    assert "Skipped:" in html
    assert "canonical frame not found" in html
    assert "sampled" in html
    assert "data:image/svg+xml;base64," in html
    assert "noise" not in html


def test_eda_report_renders_without_figures_and_with_an_etl_comparison(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    assert cfg.dataset is not None and cfg.etl is not None
    cfg = cfg.model_copy(
        update={"dataset": cfg.dataset.model_copy(update={"task": TaskType.CLASSIFICATION})}
    )
    _write(cfg, cfg.etl.train_file, pd.DataFrame({"category": ["a", "a"]}))
    _write(cfg, cfg.etl.val_file, pd.DataFrame({"category": ["a"]}))
    _write(
        cfg,
        f"{cfg.dataset.name}_canonical.parquet",
        pd.DataFrame({"category": ["a", "a", "a"], "legacy": [1, 2, 3]}),
    )

    module.run(cfg)

    html = cfg.eda.output_path.read_text(encoding="utf-8")
    assert "No figures were produced." in html
    assert "removed columns: legacy" in html
    assert "data:image" not in html


def test_eda_fails_loud_when_the_train_split_is_missing(tmp_path: Path) -> None:
    cfg = _config(tmp_path)

    with pytest.raises(FileNotFoundError, match="train split not found"):
        module.run(cfg)


def test_eda_requires_dataset_eda_and_etl_config(tmp_path: Path) -> None:
    cfg = _config(tmp_path).model_copy(update={"eda": None})

    with pytest.raises(ValueError, match="requires dataset, eda, and etl"):
        module.run(cfg)
