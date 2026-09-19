from __future__ import annotations

import types

import pandas as pd
import pytest

from broadway.config.schema import (
    BuilderParams,
    ColumnRole,
    ColumnSchema,
    DatasetContract,
    DerivedFeature,
    EncodingConfig,
    FeatureConfig,
    TaskType,
)
from broadway.features import builders
from broadway.features.builders import build_derived, load_custom_builders
from broadway.features.generic import build_generic_feature_specs
from broadway.features.pipeline import FeaturePipeline


class _FakeImportlib:
    def __init__(self, mod: types.ModuleType) -> None:
        self._mod = mod

    def import_module(self, name: str) -> types.ModuleType:
        return self._mod


def test_datetime_builders() -> None:
    df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01 09:30", "2024-02-03 18:00"])})
    feats = [
        DerivedFeature(name="h", func="datetime_hour", source="ts"),
        DerivedFeature(name="dw", func="datetime_dayofweek", source="ts"),
        DerivedFeature(name="m", func="datetime_month", source="ts"),
    ]
    result = build_derived(df, feats, "t")
    assert result["h"].tolist() == [9, 18]
    assert result["m"].tolist() == [1, 2]
    assert result["dw"].tolist() == [0, 5]


def test_source_copy_casts_to_declared_float64() -> None:
    # GAP-2: source_copy must honor the declared dtype (float64) regardless of
    # the source column's dtype — _BUILDER_DTYPES is the dtype SSOT.
    frames = {
        "int64": pd.DataFrame({"src": pd.Series([1, 2, 3], dtype="int64")}),
        "float64": pd.DataFrame({"src": pd.Series([1.5, 2.5, 3.5], dtype="float64")}),
    }
    for source_dtype, df in frames.items():
        result = build_derived(
            df, [DerivedFeature(name="copy", func="source_copy", source="src")], "t"
        )
        assert str(result["copy"].dtype) == "float64", f"source dtype {source_dtype}"
        assert result["copy"].notna().all()
        assert result["copy"].tolist() == df["src"].tolist()


def test_load_custom_builders_import_error() -> None:
    with pytest.raises(ValueError):
        load_custom_builders("does_not_exist_xyz")


def test_load_custom_builders_missing_builders(monkeypatch) -> None:
    monkeypatch.setattr(builders, "importlib", _FakeImportlib(types.SimpleNamespace()))
    with pytest.raises(ValueError):
        load_custom_builders("fake_module")


def test_load_custom_builders_collision(monkeypatch) -> None:
    mod = types.SimpleNamespace(BUILDERS={"datetime_hour": lambda df, src, **kw: 0})
    monkeypatch.setattr(builders, "importlib", _FakeImportlib(mod))
    with pytest.raises(ValueError, match="collides"):
        load_custom_builders("fake_module")


def test_load_custom_builders_valid(monkeypatch) -> None:
    mod = types.SimpleNamespace(BUILDERS={"my_feature": lambda df, src, **kw: pd.Series([1, 2])})
    monkeypatch.setattr(builders, "importlib", _FakeImportlib(mod))
    result = load_custom_builders("fake_module")
    assert "my_feature" in result


def test_build_generic_feature_specs() -> None:
    contract = DatasetContract(
        name="test",
        path="/tmp/test.csv",
        target="price",
        task=TaskType.REGRESSION,
        datetime_column="ts",
        columns={
            "rooms": ColumnSchema(dtype="int64", null_count=0, role=ColumnRole.FEATURE),
            "ts": ColumnSchema(dtype="datetime64[us]", null_count=0, role=ColumnRole.DATETIME),
            "hood": ColumnSchema(dtype="object", null_count=0, role=ColumnRole.FEATURE),
        },
        lookup_tables={},
    )
    features = FeatureConfig(
        include=["rooms"],
        exclude=[],
        derived=[DerivedFeature(name="ts_hour", func="datetime_hour", source="ts")],
        encodings=[EncodingConfig(type="target", columns=["hood"], smoothing=20)],
    )
    specs = build_generic_feature_specs(contract, features)
    assert set(specs) == {"rooms", "ts_hour", "hood_target_enc"}
    assert specs["ts_hour"].dtype == "int64"


def test_feature_pipeline_transform_with_custom_builder(tmp_path, monkeypatch) -> None:
    mod_file = tmp_path / "custom_builders_mod.py"
    mod_file.write_text(
        'BUILDERS = {\n    "double": lambda df, src, **kw: df[src] * 2,\n}\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    cfg = FeatureConfig(
        include=["x"],
        exclude=[],
        derived=[DerivedFeature(name="d", func="double", source="x")],
        encodings=[],
        builder_module="custom_builders_mod",
    )
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = FeaturePipeline(encodings=[]).transform(df, cfg, "t", 0)
    assert result["d"].tolist() == [2, 4, 6]


def test_feature_pipeline_same_group_with_builder_params() -> None:
    df = pd.DataFrame(
        {
            "Borough": ["Manhattan", "Queens", "Manhattan", None],
            "Borough_lookup": ["Manhattan", "Brooklyn", None, None],
            "pickup_location_id": [1, 2, 3, 4],
        }
    )
    cfg = FeatureConfig(
        include=[],
        exclude=[],
        derived=[
            DerivedFeature(name="same_borough", func="same_group", source="pickup_location_id")
        ],
        encodings=[],
        builder_params=BuilderParams(group_col="Borough", lookup_col="Borough_lookup"),
    )
    result = FeaturePipeline(encodings=[]).transform(df, cfg, "t", 0)
    expected = (df.Borough == df.Borough_lookup).astype(int).rename("same_borough")
    pd.testing.assert_series_equal(result["same_borough"], expected)


def test_feature_pipeline_same_group_without_builder_params_raises() -> None:
    df = pd.DataFrame(
        {
            "Borough": ["Manhattan"],
            "Borough_lookup": ["Manhattan"],
            "pickup_location_id": [1],
        }
    )
    cfg = FeatureConfig(
        include=[],
        exclude=[],
        derived=[
            DerivedFeature(name="same_borough", func="same_group", source="pickup_location_id")
        ],
        encodings=[],
    )
    with pytest.raises(ValueError, match="same_group requires column 'group'"):
        FeaturePipeline(encodings=[]).transform(df, cfg, "t", 0)


def test_feature_pipeline_absent_builder_params_is_inert() -> None:
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2024-01-01 09:30", "2024-02-03 18:00"]),
            "src": pd.Series([1.5, 2.5], dtype="float64"),
        }
    )
    cfg = FeatureConfig(
        include=[],
        exclude=[],
        derived=[
            DerivedFeature(name="h", func="datetime_hour", source="ts"),
            DerivedFeature(name="copy", func="source_copy", source="src"),
        ],
        encodings=[],
    )
    assert cfg.builder_params is None
    pipeline_out = FeaturePipeline(encodings=[]).transform(df, cfg, "t", 0)
    direct_out = build_derived(
        df, cfg.derived, "t", extra_builders=load_custom_builders(cfg.builder_module)
    )
    pd.testing.assert_frame_equal(pipeline_out, direct_out)
