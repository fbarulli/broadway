"""Encoding transformer tests — synthetic frames, golden-value pins.

Single-column frequency behavior is pinned to the exact outputs the legacy
``broadway.features.encodings`` free functions produced (values captured
before that module's deletion); the FeaturePipeline column names/values are
the pre-change implementation's golden output. Target encoding is not a
technique this project ships.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone

from broadway.features.pipeline import FeaturePipeline
from broadway.features.plan import ColumnOp, ColumnPlan
from broadway.features.transformers import FrequencyEncoding


def _price_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "city": ["NYC", "LA", "NYC", "SF", "LA", "NYC", "SF", "SF", "NYC", "LA"],
            "zone": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
            "price": [100, 200, 150, 300, 250, 120, 280, 320, 130, 210],
        }
    )


def test_frequency_encoding_golden_values() -> None:
    """Normalized frequencies value-pinned from the legacy reference at deletion time."""
    frame = _price_frame()
    out = FrequencyEncoding(columns=["city"]).fit(frame).transform(frame)
    assert out.columns.tolist() == ["city", "zone", "price", "city_freq_enc"]
    assert out["city_freq_enc"].round(10).tolist() == [
        0.4,
        0.3,
        0.4,
        0.3,
        0.3,
        0.4,
        0.3,
        0.3,
        0.4,
        0.3,
    ]
    unseen = pd.DataFrame({"city": ["PARIS"]})
    filled = FrequencyEncoding(columns=["city"]).fit(frame).transform(unseen, fill=0.5)
    assert filled["city_freq_enc"].tolist() == [0.5]


def test_clone_refit_independence() -> None:
    frame_a = _price_frame()
    frame_b = pd.DataFrame({"city": ["NYC", "SF"], "price": [400, 500]})
    fitted = FrequencyEncoding(columns=["city"]).fit(frame_a)
    baseline = fitted.transform(frame_a)["city_freq_enc"].round(10).tolist()
    cloned = clone(fitted)
    assert not hasattr(cloned, "mapping_")
    cloned.fit(frame_b)
    assert cloned.mapping_ != fitted.mapping_
    assert fitted.transform(frame_a)["city_freq_enc"].round(10).tolist() == baseline
    assert cloned.transform(frame_a)["city_freq_enc"].round(10).tolist() != baseline


def test_get_params_set_params_round_trip() -> None:
    transformer = FrequencyEncoding(columns=["city"])
    assert clone(transformer).get_params() == transformer.get_params()

    columns = ["city"]
    freq = FrequencyEncoding(columns=columns)
    columns.append("zone")
    assert freq.columns == ["city"]

    freq.set_params(columns=["zone"])
    assert freq.columns == ["zone"]
    assert "zone_freq_enc" in freq.fit(_price_frame()).transform(_price_frame()).columns


def test_index_and_row_order_preserved() -> None:
    frame = _price_frame().sample(frac=1, random_state=7)
    frame.index = frame.index + 100
    freq = FrequencyEncoding(columns=["city"]).fit(frame)
    out = freq.transform(frame)
    assert out.index.tolist() == frame.index.tolist()
    assert len(out) == len(frame)
    assert out["city_freq_enc"].tolist() == [freq.mapping_[city] for city in frame["city"]]
    assert out.columns.tolist() == ["city", "zone", "price", "city_freq_enc"]
    assert frame.columns.tolist() == ["city", "zone", "price"]


def test_pipeline_reexpression_output_identical() -> None:
    frame = pd.DataFrame(
        {"x": [1, 2, 3, 4, 5], "city": ["NYC", "LA", "NYC", "SF", "LA"], "price": [100, 200, 150, 300, 250]}
    )
    plan = ColumnPlan(
        target="price",
        datetime_column=None,
        columns=(
            ColumnOp(op="source", name="x", dtype="int64"),
            ColumnOp(
                op="encode",
                name="city_freq_enc",
                dtype="float64",
                sources=("city",),
                encoder="frequency",
            ),
        ),
    )
    pipeline = FeaturePipeline().fit(frame, plan, "price")
    out = pipeline.transform(frame, plan, "price", 0)
    assert list(out.columns) == ["x", "city", "price", "city_freq_enc"]
    assert out["city_freq_enc"].tolist() == [0.4, 0.4, 0.4, 0.2, 0.4]


def test_feature_name_param_overrides_platform_default() -> None:
    frame = _price_frame()
    custom_freq = FrequencyEncoding(columns=["city"], feature_name="custom_freq")
    assert "custom_freq" in custom_freq.fit(frame).transform(frame).columns
    assert "city_freq_enc" not in custom_freq.fit(frame).transform(frame).columns

    default_freq = FrequencyEncoding(columns=["city"])
    assert "city_freq_enc" in default_freq.fit(frame).transform(frame).columns
