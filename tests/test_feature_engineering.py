"""Public behavior of the feature-engineering engines.

One focused test per engine: output shape/columns, manifest entries, and
determinism. The engine libraries are optional (``feature_engineering`` extra),
so each test skips with a reason when its library is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from broadway.features.engineering import (
    DartsEngineConfig,
    FeatureEngineer,
    FeatureEngineeringConfig,
    FeatureEngineTabularConfig,
    FeatureEngineTransformerConfig,
    SktimeEngineConfig,
    TsfreshEngineConfig,
)
from broadway.features.plan import ColumnPlan

N_POINTS = 24
N_SERIES = 2


def _frame() -> pd.DataFrame:
    """Two deterministic hourly series with a trend and a daily wiggle."""
    rows: list[dict[str, object]] = []
    for series_id in range(N_SERIES):
        for step in range(N_POINTS):
            rows.append(
                {
                    "id": series_id,
                    "ts": pd.Timestamp("2024-01-01") + pd.to_timedelta(step, unit="h"),
                    "value": 100.0 + series_id * 10 + step + 5 * np.sin(step / 3),
                }
            )
    return pd.DataFrame(rows)


def _config(engine: object) -> FeatureEngineeringConfig:
    return FeatureEngineeringConfig.model_validate(
        {"time_column": "ts", "id_columns": ["id"], "engines": [engine]}
    )


def _plan(frame: pd.DataFrame) -> ColumnPlan:
    return ColumnPlan.from_frame_columns(frame, ["value"])


def _build_twice(config: FeatureEngineeringConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _frame()
    result = FeatureEngineer.build(frame, config, _plan(frame))
    repeated = FeatureEngineer.build(frame, config, _plan(frame))
    return result.frames[config.engines[0].id], repeated.frames[config.engines[0].id]


def _write_plan(path: Path, time_column: str, engine_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "time_column": time_column,
                "id_columns": [],
                "engines": [
                    {
                        "kind": "tsfresh",
                        "id": engine_id,
                        "feature_set": "minimal",
                        "n_jobs": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_from_yaml_prefers_the_overlay_and_an_explicit_path_still_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "configs" / "feature_engineering.yaml"
    overlay = tmp_path / "overlay" / "feature_engineering.yaml"
    _write_plan(base, "base_time", "base")
    _write_plan(overlay, "overlay_time", "overlay")
    monkeypatch.setenv("BROADWAY_CONFIGS_DIR", str(base.parent))
    monkeypatch.setenv("BROADWAY_CONFIG_OVERLAY_DIR", str(overlay.parent))

    resolved = FeatureEngineeringConfig.from_yaml()
    assert resolved.time_column == "overlay_time"
    assert resolved.engines[0].id == "overlay"

    explicit = FeatureEngineeringConfig.from_yaml(base)
    assert explicit.time_column == "base_time"
    assert explicit.engines[0].id == "base"


def test_tsfresh_produces_one_feature_row_per_series_and_a_matching_manifest() -> None:
    pytest.importorskip("tsfresh", reason="optional feature_engineering extra not installed")
    config = _config(
        TsfreshEngineConfig(
            kind="tsfresh",
            id="tsfresh",
            feature_set="minimal",
            n_jobs=1,
        )
    )
    frame = _frame()
    result = FeatureEngineer.build(frame, config, _plan(frame))
    output = result.frames["tsfresh"]

    assert output["id"].tolist() == [0, 1]
    assert output.shape[0] == N_SERIES
    assert output.shape[1] > 1
    assert {entry.name for entry in result.manifest} == set(output.columns) - {"id"}
    assert all(entry.kind == "tsfresh" for entry in result.manifest)
    assert all(entry.source_columns == ["value"] for entry in result.manifest)
    first, second = _build_twice(config)
    pd.testing.assert_frame_equal(first, second)


def test_feature_engine_imputes_and_winsorizes_in_config_order() -> None:
    pytest.importorskip("feature_engine", reason="optional feature_engineering extra not installed")
    frame = _frame()
    frame.loc[3, "value"] = np.nan
    config = _config(
        FeatureEngineTabularConfig(
            kind="feature_engine",
            id="feature_engine",
            transformers=[
                FeatureEngineTransformerConfig(
                    kind="MeanMedianImputer",
                    params={"imputation_method": "median"},
                ),
                FeatureEngineTransformerConfig(
                    kind="Winsorizer",
                    params={"capping_method": "quantiles", "tail": "both", "fold": 0.05},
                ),
            ],
        )
    )
    result = FeatureEngineer.build(frame, config, _plan(frame))
    output = result.frames["feature_engine"]

    assert output.shape == frame.shape
    assert output.columns.tolist() == ["id", "ts", "value"]
    assert not output["value"].isna().any()
    assert len(result.manifest) == 1
    entry = result.manifest[0]
    assert entry.name == "value"
    assert entry.source_columns == ["value"]
    assert [step["transformer"] for step in entry.params["steps"]] == [
        "MeanMedianImputer",
        "Winsorizer",
    ]
    first, second = _build_twice(config)
    pd.testing.assert_frame_equal(first, second)


def test_sktime_summarizes_each_series_with_manifest_entries() -> None:
    pytest.importorskip("sktime", reason="optional feature_engineering extra not installed")
    config = _config(
        SktimeEngineConfig(
            kind="sktime",
            id="sktime",
            transformer="SummaryTransformer",
            params={"summary_function": ["mean", "std", "min", "max"], "quantiles": None},
        )
    )
    frame = _frame()
    result = FeatureEngineer.build(frame, config, _plan(frame))
    output = result.frames["sktime"]

    assert output.shape == (N_SERIES, 5)
    assert output["id"].tolist() == [0, 1]
    assert [entry.name for entry in result.manifest] == ["mean", "std", "min", "max"]
    assert all(entry.kind == "sktime" for entry in result.manifest)
    assert all(entry.source_columns == ["value"] for entry in result.manifest)
    first, second = _build_twice(config)
    pd.testing.assert_frame_equal(first, second)


def test_darts_differences_each_series_and_records_the_transform() -> None:
    pytest.importorskip("darts", reason="optional feature_engineering extra not installed")
    config = _config(
        DartsEngineConfig(
            kind="darts",
            id="darts",
            transformer="Diff",
            freq=None,
            params={"lags": 1},
        )
    )
    frame = _frame()
    result = FeatureEngineer.build(frame, config, _plan(frame))
    output = result.frames["darts"]

    assert output.shape == (N_SERIES * (N_POINTS - 1), 3)
    assert output.columns.tolist() == ["id", "ts", "value"]
    assert sorted(output["id"].unique().tolist()) == [0, 1]
    assert len(result.manifest) == 1
    entry = result.manifest[0]
    assert entry.name == "value"
    assert entry.params == {"transformer": "Diff", "lags": 1}
    assert entry.source_columns == ["value"]
    first, second = _build_twice(config)
    pd.testing.assert_frame_equal(first, second)
