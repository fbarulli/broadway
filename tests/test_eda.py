"""Public behavior of the EDA step: counts, flags, comparison and report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from broadway.config.loader import load_config
from broadway.config.schema import TaskType
from broadway.eda import module
from broadway.eda.compare import EtlComparator
from broadway.eda.missing import MissingnessAnalyzer
from broadway.eda.quality import QualityAnalyzer
from broadway.eda.summary import SummaryAnalyzer


def test_summary_and_missingness_count_a_frame() -> None:
    frame = pd.DataFrame(
        {"n": [1.0, 2.0, 3.0, None, 5.0], "c": ["a", "a", "b", None, "b"]}
    )

    summary = SummaryAnalyzer(5).analyze(frame, {"n": "feature", "c": "feature"})
    missing = MissingnessAnalyzer(5).analyze(frame)

    assert summary.rows == 5
    assert summary.columns == 2
    assert summary.numeric_columns == 1
    assert summary.categorical_columns == 1
    numeric = next(item for item in summary.detail if item.name == "n")
    assert numeric.numeric is not None
    assert numeric.numeric.count == 4
    assert numeric.numeric.mean == 2.75
    assert numeric.role == "feature"

    assert missing.total_cells == 10
    assert missing.missing_cells == 2
    assert [column.column for column in missing.columns] == ["n", "c"]
    assert missing.columns[0].missing_rate == 0.2
    assert missing.pattern_count == 2
    assert missing.patterns[0].columns == []
    assert missing.patterns[0].row_count == 4
    assert missing.patterns[1].columns == ["n", "c"]
    assert missing.patterns[1].row_count == 1


def test_quality_flags_constants_outliers_and_imbalance() -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10000.0],
            "constant": [7] * 8,
            "target": ["a"] * 7 + ["b"],
        }
    )

    quality = QualityAnalyzer(1.5, 2.0, 0.2).analyze(
        frame, TaskType.CLASSIFICATION, "target"
    )

    assert quality.duplicate_rows >= 1
    assert "constant" in quality.constant_columns
    outliers = next(item for item in quality.outliers if item.column == "x")
    assert outliers.iqr_count == 1
    assert outliers.z_count == 1
    assert quality.imbalance.applicable
    assert quality.imbalance.flagged
    assert quality.imbalance.minority_share == 0.125


def test_etl_compare_records_skip_reason_and_row_delta() -> None:
    comparator = EtlComparator("before", "after")
    skipped = comparator.compare(None, pd.DataFrame({"a": [1, 2]}), 10, 2, "missing")
    assert not skipped.available
    assert skipped.skip_reason == "missing"
    assert skipped.before_rows is None

    compared = comparator.compare(
        pd.DataFrame({"a": [1.0, 2.0]}),
        pd.DataFrame({"a": [2.0, 3.0], "b": [4.0, 5.0]}),
        10,
        8,
        None,
    )
    assert compared.available
    assert compared.row_delta == -2
    assert compared.added_columns == ["b"]
    assert compared.numeric_shifts[0].mean_delta == 1.0


def test_eda_run_writes_self_contained_html_report(tmp_path: Path) -> None:
    cfg = load_config("eda", dataset="test", experiment="baseline")
    assert cfg.eda is not None
    assert cfg.etl is not None
    cfg = cfg.model_copy(
        update={
            "environment": cfg.environment.model_copy(
                update={"data_dir": str(tmp_path)}
            ),
            "eda": cfg.eda.model_copy(
                update={
                    "sample_size": 0,
                    "output_dir": str(tmp_path / "reports"),
                    "output_file": "eda.html",
                }
            ),
        }
    )
    processed = tmp_path / cfg.environment.processed_subdir
    processed.mkdir(parents=True)
    train = pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_2": [10, 20, 30, 40, 50],
            "feature_3": ["a", "b", "a", "b", "a"],
            "target": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    val = train.iloc[:2].copy()
    train.to_parquet(processed / cfg.etl.train_file, index=False)
    val.to_parquet(processed / cfg.etl.val_file, index=False)
    pd.concat([train, val], ignore_index=True).to_parquet(
        processed / "test_canonical.parquet", index=False
    )

    module.run(cfg)

    report = tmp_path / "reports" / "eda.html"
    assert report.exists()
    assert report.stat().st_size > 0
    html = report.read_text(encoding="utf-8")
    assert "Exploratory Data Analysis" in html
    assert "data:image/png;base64," in html
    assert "Before / after ETL" in html
    assert "test_canonical" not in html

    size_before = report.stat().st_size
    module.run(cfg)
    assert report.exists()
    assert report.stat().st_size == size_before
