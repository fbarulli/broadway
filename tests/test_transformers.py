"""Encoding transformer tests — synthetic frames, golden vs legacy free functions.

Single-column behavior is compared live against the legacy
``broadway.features.encodings`` free functions; the composite-key spec is
re-derived in-test via explicit groupby + formula; the FeaturePipeline column
names/values are the pre-change implementation's golden output.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.base import clone

from broadway.config.schema import EncodingConfig, FeatureConfig
from broadway.features.encodings import (
    fit_frequency_encoding,
    fit_target_encoding,
    transform_frequency_encoding,
    transform_target_encoding,
)
from broadway.features.pipeline import FeaturePipeline
from broadway.features.transformers import FrequencyEncoding, TargetEncoding


def _price_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "city": ["NYC", "LA", "NYC", "SF", "LA", "NYC", "SF", "SF", "NYC", "LA"],
            "zone": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
            "price": [100, 200, 150, 300, 250, 120, 280, 320, 130, 210],
        }
    )


def test_target_encoding_matches_legacy_single_column() -> None:
    frame = _price_frame()
    legacy_map = fit_target_encoding(frame, "city", "price", 10)
    legacy_out = transform_target_encoding(frame, "city", legacy_map)
    out = TargetEncoding(columns=["city"], target="price", smoothing=10).fit(frame).transform(frame)
    assert out.columns.tolist() == legacy_out.columns.tolist()
    pd.testing.assert_series_equal(out["city_target_enc"].round(10), legacy_out["city_target_enc"].round(10))


def test_target_encoding_multi_column_key_matches_formula() -> None:
    frame = _price_frame()
    smoothing = 10.0
    global_mean = frame["price"].mean()
    stats = frame.groupby(["city", "zone"])["price"].agg(["mean", "count"])
    expected = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
    keys = list(zip(frame["city"], frame["zone"]))
    expected_series = pd.Series([expected[key] for key in keys], index=frame.index, name="city_zone_target_enc")

    out = TargetEncoding(columns=["city", "zone"], target="price", smoothing=smoothing).fit(frame).transform(frame)

    assert out.columns.tolist() == ["city", "zone", "price", "city_zone_target_enc"]
    pd.testing.assert_series_equal(out["city_zone_target_enc"].round(10), expected_series.round(10))
    city_only = TargetEncoding(columns=["city"], target="price", smoothing=smoothing).fit(frame).transform(frame)
    zone_only = TargetEncoding(columns=["zone"], target="price", smoothing=smoothing).fit(frame).transform(frame)
    assert not out["city_zone_target_enc"].round(10).equals(city_only["city_target_enc"].round(10))
    assert not out["city_zone_target_enc"].round(10).equals(zone_only["zone_target_enc"].round(10))


def test_target_encoding_nan_category_falls_back_like_legacy() -> None:
    frame = pd.DataFrame(
        {"city": ["NYC", None, "LA", "SF", None, "NYC"], "price": [100, 200, 150, 300, 250, 120]}
    )
    legacy_map = fit_target_encoding(frame, "city", "price", 10)
    legacy_out = transform_target_encoding(frame, "city", legacy_map)
    out = TargetEncoding(columns=["city"], target="price", smoothing=10).fit(frame).transform(frame)
    pd.testing.assert_series_equal(out["city_target_enc"].round(10), legacy_out["city_target_enc"].round(10))
    assert out["city_target_enc"].iloc[1] == pytest.approx(186.66666666666666)
    assert out["city_target_enc"].iloc[4] == pytest.approx(186.66666666666666)


def test_frequency_encoding_matches_legacy() -> None:
    frame = _price_frame()
    legacy_map = fit_frequency_encoding(frame, "city")
    legacy_out = transform_frequency_encoding(frame, "city", legacy_map)
    out = FrequencyEncoding(columns=["city"]).fit(frame).transform(frame)
    assert out.columns.tolist() == legacy_out.columns.tolist()
    pd.testing.assert_series_equal(out["city_freq_enc"].round(10), legacy_out["city_freq_enc"].round(10))
    unseen = pd.DataFrame({"city": ["PARIS"]})
    filled = FrequencyEncoding(columns=["city"]).fit(frame).transform(unseen, fill=0.5)
    assert filled["city_freq_enc"].tolist() == [0.5]


def test_clone_refit_independence() -> None:
    frame_a = _price_frame()
    frame_b = pd.DataFrame({"city": ["NYC", "SF"], "price": [400, 500]})
    fitted = TargetEncoding(columns=["city"], target="price", smoothing=10).fit(frame_a)
    baseline = fitted.transform(frame_a)["city_target_enc"].round(10).tolist()
    cloned = clone(fitted)
    assert not hasattr(cloned, "_mapping")
    cloned.fit(frame_b)
    assert cloned._mapping != fitted._mapping
    assert fitted.transform(frame_a)["city_target_enc"].round(10).tolist() == baseline
    assert cloned.transform(frame_a)["city_target_enc"].round(10).tolist() != baseline


def test_get_params_set_params_round_trip() -> None:
    for transformer in (
        TargetEncoding(columns=["city"], target="price", smoothing=10),
        FrequencyEncoding(columns=["city"]),
    ):
        assert clone(transformer).get_params() == transformer.get_params()

    columns = ["city"]
    target = TargetEncoding(columns=columns, target="price", smoothing=10)
    columns.append("zone")
    assert target.columns == ["city"]

    target.set_params(columns=["zone"], smoothing=5)
    assert target.columns == ["zone"]
    assert target.smoothing == 5
    assert "zone_target_enc" in target.fit(_price_frame()).transform(_price_frame()).columns

    freq = FrequencyEncoding(columns=["city"])
    freq.set_params(columns=["zone"])
    assert freq.columns == ["zone"]
    assert "zone_freq_enc" in freq.fit(_price_frame()).transform(_price_frame()).columns


def test_index_and_row_order_preserved() -> None:
    frame = _price_frame().sample(frac=1, random_state=7)
    frame.index = frame.index + 100
    target = TargetEncoding(columns=["city"], target="price", smoothing=10).fit(frame)
    freq = FrequencyEncoding(columns=["city"]).fit(frame)
    for transformer, column, mapping in (
        (target, "city_target_enc", target._mapping),
        (freq, "city_freq_enc", freq._mapping),
    ):
        out = transformer.transform(frame)
        assert out.index.tolist() == frame.index.tolist()
        assert len(out) == len(frame)
        assert out[column].tolist() == [mapping[city] for city in frame["city"]]
        assert out.columns.tolist() == ["city", "zone", "price", column]
    assert frame.columns.tolist() == ["city", "zone", "price"]


def test_pipeline_reexpression_output_identical() -> None:
    frame = pd.DataFrame(
        {"x": [1, 2, 3, 4, 5], "city": ["NYC", "LA", "NYC", "SF", "LA"], "price": [100, 200, 150, 300, 250]}
    )
    cfg = FeatureConfig(
        include=["x"],
        exclude=[],
        derived=[],
        encodings=[
            EncodingConfig(type="frequency", columns=["city"], smoothing=None),
            EncodingConfig(type="target", columns=["city"], smoothing=10),
        ],
        builder_module=None,
    )
    pipeline = FeaturePipeline(encodings=cfg.encodings).fit(frame, "price", 10)
    out = pipeline.transform(frame, cfg, "price", 0)
    assert list(out.columns) == ["x", "city", "price", "city_freq_enc", "city_target_enc"]
    assert out["city_freq_enc"].tolist() == [0.4, 0.4, 0.4, 0.2, 0.4]
    assert out["city_target_enc"].round(10).tolist() == [187.5, 204.1666666667, 187.5, 209.0909090909, 204.1666666667]


def test_feature_name_param_overrides_platform_default() -> None:
    frame = _price_frame()
    custom_target = TargetEncoding(
        columns=["city"], target="price", smoothing=10, feature_name="custom_target"
    )
    custom_freq = FrequencyEncoding(columns=["city"], feature_name="custom_freq")
    assert "custom_target" in custom_target.fit(frame).transform(frame).columns
    assert "custom_freq" in custom_freq.fit(frame).transform(frame).columns
    assert "city_target_enc" not in custom_target.fit(frame).transform(frame).columns
    assert "city_freq_enc" not in custom_freq.fit(frame).transform(frame).columns

    default_target = TargetEncoding(columns=["city"], target="price", smoothing=10)
    default_freq = FrequencyEncoding(columns=["city"])
    assert "city_target_enc" in default_target.fit(frame).transform(frame).columns
    assert "city_freq_enc" in default_freq.fit(frame).transform(frame).columns
