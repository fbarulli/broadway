from __future__ import annotations

import pandas as pd

import broadway.tracking.baseline as baseline_module
from broadway.discover.profile import build_profile
from broadway.lineage.models import TransformAudit
from broadway.tracking.baseline import (
    BaselineOptions,
    BaselineStats,
    DataCardOptions,
    build_baseline_stats,
    build_baseline_stats_from_parquet,
    build_feature_baseline,
    dump_baseline_counts,
    load_baseline_stats,
    render_data_card,
    write_baseline_stats,
    write_data_card,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "cat": ["a", "b", "a", "a", "b", "c", "a", None],
            "when": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-06",
                    "2024-01-07",
                    "2024-01-08",
                ]
            ),
        }
    )


def _audit() -> TransformAudit:
    return TransformAudit(
        rows_in=10,
        rows_out=8,
        rows_dropped_total=2,
        rows_dropped_unexplained=0,
        reasons=["dedup: -2 rows"],
        columns_before=["num", "cat", "when", "drop_me"],
        columns_after=["num", "cat", "when"],
        columns_added=[],
        columns_removed=["drop_me"],
    )


def test_numeric_moments_and_bins() -> None:
    stats = build_baseline_stats(
        _frame(), dataset="demo", options=BaselineOptions(n_bins=4, top_k=3)
    )
    num = stats.features["num"]
    assert num.kind == "numeric"
    assert num.mean == 4.5
    assert num.std is not None and num.std > 0
    assert sum(num.bin_counts) == 8
    assert len(num.bin_edges) == len(num.bin_counts) + 1


def test_categorical_top_k_bins() -> None:
    stats = build_baseline_stats(
        _frame(), dataset="demo", options=BaselineOptions(n_bins=4, top_k=2)
    )
    cat = stats.features["cat"]
    assert cat.kind == "categorical"
    assert cat.missing_count == 1
    assert cat.bin_labels[0] == "a"
    assert len(cat.bin_counts) <= 2
    assert sum(cat.bin_counts) == 6  # top-2 truncation drops the singleton "c"
    full = build_baseline_stats(
        _frame(), dataset="demo", options=BaselineOptions(n_bins=4, top_k=10)
    )
    assert sum(full.features["cat"].bin_counts) == 7


def test_datetime_and_constant_columns() -> None:
    df = pd.DataFrame(
        {
            "const": [5.0, 5.0, 5.0],
            "when": pd.to_datetime(["2024-01-01", "2024-06-01", "2024-12-31"]),
        }
    )
    stats = build_baseline_stats(df, dataset="demo")
    assert stats.features["const"].bin_counts == [3]
    assert stats.features["when"].kind == "datetime"
    assert stats.features["when"].min_value is not None


def test_from_parquet_matches_frame(tmp_path) -> None:
    df = _frame()
    parquet = tmp_path / "canonical.parquet"
    df.to_parquet(parquet, index=False)
    stats = build_baseline_stats_from_parquet(parquet, dataset="demo")
    assert stats.row_count == 8
    assert set(stats.features) == {"num", "cat", "when"}


def test_write_round_trip(tmp_path) -> None:
    stats = build_baseline_stats(_frame(), dataset="demo")
    path = write_baseline_stats(stats, tmp_path / "tracking" / "baseline_stats.json")
    reloaded = load_baseline_stats(path)
    assert reloaded == stats
    assert dump_baseline_counts(path)["num"] == stats.features["num"].bin_counts


def test_data_card_sections() -> None:
    df = _frame()
    profile = build_profile("demo", "demo.parquet", df)
    baseline = build_baseline_stats(df, dataset="demo")
    card = render_data_card(
        dataset="demo",
        profile=profile,
        audits=[_audit()],
        baseline=baseline,
        options=DataCardOptions(biases=["weekday-only sampling"], generated_at="2026-09-21"),
    )
    assert "2024-01-01" in card and "2024-01-08" in card
    assert "weekday-only sampling" in card
    assert "dedup: -2 rows" in card and "drop_me" in card
    assert "cat: 1/8" in card


def test_data_card_no_bias_means_not_declared() -> None:
    card = render_data_card(dataset="demo")
    assert "not declared" in card
    assert "unknown (no DatasetProfile provided)" in card


def test_numeric_all_null_returns_empty_baseline() -> None:
    df = pd.DataFrame({"num": pd.Series([float("nan"), float("nan")], dtype="float64")})
    stats = build_baseline_stats(df, dataset="demo")
    num = stats.features["num"]
    assert num.kind == "numeric"
    assert num.count == 0
    assert num.missing_count == 2
    assert num.mean is None and num.std is None
    assert num.bin_counts == [] and num.bin_edges == []


def test_other_kind_branch_covered(monkeypatch) -> None:
    monkeypatch.setattr(baseline_module, "_classify", lambda series: "other")
    feat = build_feature_baseline(
        "mystery",
        pd.Series([1, None, 3], dtype="object"),
        options=BaselineOptions(),
        row_count=3,
    )
    assert feat.kind == "other"
    assert feat.missing_count == 1
    assert feat.count == 2


def test_date_range_without_datetime_bounds_renders_unknown() -> None:
    df = pd.DataFrame({"value": [1, 2, 3]})
    profile = build_profile("demo", "demo.parquet", df)
    card = render_data_card(dataset="demo", profile=profile)
    assert "unknown (profile carries no datetime bounds)" in card


def test_write_data_card(tmp_path) -> None:
    card = render_data_card(dataset="demo", audits=[_audit()])
    path = write_data_card(card, tmp_path / "tracking" / "data_card.md")
    assert path.read_text().endswith("\n")
    assert isinstance(BaselineStats, type)
