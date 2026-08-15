from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError
from scipy import stats

from broadway import viz
import broadway.discover.module as discover_module
from broadway.config.loader import load_config
from broadway.config.viz import QqZonesConfig, load_viz_config
from broadway.discover.profile import ColumnProfile, DatasetProfile
from broadway.discover.qq import (
    QqFeature,
    QqOverview,
    _plot_chunk,
    _qq_points,
    midpoint_bin_edges,
    plot_numeric_qq,
)
from broadway.lineage import records
from broadway.reports import audit
from broadway.timeline import module as timeline_module
from broadway.timeline import runners
from broadway.timeline.evidence import NormalityEvidence


def _cfg():
    return load_config("stats", dataset="taxi", analysis="taxi_hypothesis")


def _profile():
    return DatasetProfile(
        name="taxi",
        path="data/processed/training_data.parquet",
        row_count=10,
        columns={
            "id": ColumnProfile(
                dtype="int64",
                null_count=0,
                cardinality=10,
                min="1",
                max="10",
                datetime_min=None,
                datetime_max=None,
                identifier_score=0.5,
            )
        },
    )


def test_run_normality_produces_single_joint_qq(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(timeline_module, "TIMELINE_DIR", tmp_path / "timeline")
    cfg = _cfg()
    rng = np.random.default_rng(0)
    groups = {g: rng.normal(10.0, 2.0, 30) for g in ("A", "B", "C")}
    figures_dir = tmp_path / "reports" / "figures"
    out_dir = tmp_path / "timeline" / "taxi_hypothesis"

    step = runners.run_normality(
        cfg.analysis, 2, "q?", groups, out_dir, figures_dir, "canonical", None
    )

    assert (figures_dir / "normality_qq.png").exists()
    assert list(figures_dir.glob("normality_*.png")) == [figures_dir / "normality_qq.png"]
    assert [f.path for f in step.figures] == ["figures/normality_qq.png"]
    assert "standardized" in step.figures[0].caption
    assert "fitted" in step.figures[0].caption
    assert step.result_summary.get("standardization") == "per-group z-score"

    evidence = NormalityEvidence.model_validate_json(
        (out_dir / "evidence" / "normality.json").read_text()
    )
    assert evidence.figure == "figures/normality_qq.png"
    assert evidence.standardization == "per-group z-score"


def test_run_normality_caps_at_twelve_groups(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(timeline_module, "TIMELINE_DIR", tmp_path / "timeline")
    cfg = _cfg()
    rng = np.random.default_rng(0)
    groups = {f"g{i}": rng.normal(float(i), 1.0, 20) for i in range(15)}
    figures_dir = tmp_path / "reports" / "figures"

    step = runners.run_normality(
        cfg.analysis, 2, "q?", groups, tmp_path / "timeline" / "taxi_hypothesis",
        figures_dir, "canonical", None,
    )
    assert "truncation" in step.result_summary
    assert "12" in step.result_summary["truncation"]
    assert "15" in step.result_summary["truncation"]


def test_qq_points_returns_fit_line_params() -> None:
    rng = np.random.default_rng(0)
    vals = rng.normal(0.0, 1.0, 200)

    osm, osr, slope, intercept = _qq_points(vals, 10000)

    assert osm.shape == (200,)
    assert osr.shape == (200,)
    assert isinstance(slope, float)
    assert isinstance(intercept, float)
    assert slope == pytest.approx(1.0, abs=0.05)
    assert intercept == pytest.approx(0.0, abs=0.05)


def test_qq_points_thins_large_input() -> None:
    rng = np.random.default_rng(0)
    vals = rng.normal(0.0, 1.0, 5_000)

    osm, osr, slope, intercept = _qq_points(vals, 2000)

    assert osm.size == 2_000
    assert osr.size == 2_000


def test_midpoint_bin_edges_three_values() -> None:
    edges = midpoint_bin_edges(np.array([-1.75, 0, 1.75]))
    assert np.allclose(edges, [-2.625, -0.875, 0.875, 2.625])


def test_midpoint_bin_edges_integer_values() -> None:
    edges = midpoint_bin_edges(np.array([0, 1, 2]))
    assert np.allclose(edges, [-0.5, 0.5, 1.5, 2.5])


def test_midpoint_bin_edges_counts_land_in_correct_slots() -> None:
    values = np.array([-1.75, 0, 0, 1.75])
    counts, edges = np.histogram(values, bins=midpoint_bin_edges(values))
    assert counts.tolist() == [1, 2, 1]


def _qq_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data: dict[str, object] = {}
    for i in range(14):
        data[f"f{i}"] = rng.normal(float(i), 1.0, 20)
    data["mixed"] = np.concatenate([rng.normal(0.0, 1.0, 19), [np.nan]])
    data["allnan"] = np.full(20, np.nan)
    data["const"] = np.full(20, 5.0)
    return pd.DataFrame(data)


def _qq_columns(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({f"f{i}": rng.normal(float(i), 1.0, 20) for i in range(n)})


def test_plot_numeric_qq_excludes_and_splits(tmp_path) -> None:
    df = _qq_df()
    figures_dir = tmp_path / "reports" / "figures"
    evidence_path = tmp_path / "artifacts" / "discover" / "qq_overview.json"

    overview = plot_numeric_qq(df, figures_dir, evidence_path, source_path="data.csv")

    total_cols = len(df.columns)
    max_per_figure = load_viz_config().max_features_per_figure
    plotted = [f.feature for f in overview.features if f.status == "plotted"]

    assert overview.total_features == total_cols
    assert overview.excluded_features == 2
    assert overview.discrete_features == 0
    assert overview.non_numeric_columns == []
    assert overview.flagged_id_columns == []
    assert "allnan: non-finite" in overview.excluded_notes
    assert "const: zero variance" in overview.excluded_notes

    assert overview.figures == [
        "figures/numeric_qq_1.png",
        "figures/numeric_qq_2.png",
    ]
    assert overview.dist_figures == [
        "figures/numeric_dist_1.png",
        "figures/numeric_dist_2.png",
    ]
    assert overview.diagnostics_figures == ["figures/numeric_diagnostics.png"]
    assert (figures_dir / "numeric_qq_1.png").exists()
    assert (figures_dir / "numeric_qq_2.png").exists()
    assert (figures_dir / "numeric_dist_1.png").exists()
    assert (figures_dir / "numeric_dist_2.png").exists()
    assert (figures_dir / "numeric_diagnostics.png").exists()

    by_figure: dict[str, list[str]] = {}
    dist_by_figure: dict[str, list[str]] = {}
    for f in overview.features:
        if f.figure:
            by_figure.setdefault(f.figure, []).append(f.feature)
        if f.dist_figure:
            dist_by_figure.setdefault(f.dist_figure, []).append(f.feature)
    assert len(by_figure["figures/numeric_qq_1.png"]) == max_per_figure
    assert len(by_figure["figures/numeric_qq_2.png"]) == len(plotted) - max_per_figure
    assert len(dist_by_figure["figures/numeric_dist_1.png"]) == max_per_figure
    assert len(dist_by_figure["figures/numeric_dist_2.png"]) == len(plotted) - max_per_figure

    mixed = next(f for f in overview.features if f.feature == "mixed")
    assert mixed.n_valid == int(df["mixed"].notna().sum())
    assert mixed.n_excluded == int(df["mixed"].isna().sum())

    f0 = next(f for f in overview.features if f.feature == "f0")
    vals = df["f0"].to_numpy(dtype=float)
    assert f0.n_valid == 20
    assert f0.n_excluded == 0
    assert f0.mean == pytest.approx(float(np.mean(vals)))
    assert f0.std == pytest.approx(float(np.std(vals)))
    assert f0.skew == pytest.approx(float(stats.skew(vals)))
    assert f0.kurtosis == pytest.approx(float(stats.kurtosis(vals)))

    for f in overview.features:
        if f.status == "plotted":
            assert f.figure and f.dist_figure
            assert f.skew is not None and f.kurtosis is not None

    excluded = {f.feature for f in overview.features if f.figure == ""}
    assert excluded == {"allnan", "const"}

    reloaded = QqOverview.model_validate_json(evidence_path.read_text())
    assert reloaded.figures == overview.figures
    assert reloaded.dist_figures == overview.dist_figures
    assert reloaded.diagnostics_figures == overview.diagnostics_figures


def test_plot_numeric_qq_boundary_no_overflow(tmp_path) -> None:
    figures_dir = tmp_path / "reports" / "figures"
    evidence_path = tmp_path / "artifacts" / "discover" / "qq_overview.json"

    overview = plot_numeric_qq(
        _qq_columns(12), figures_dir, evidence_path, source_path="data.csv"
    )
    assert overview.figures == ["figures/numeric_qq_1.png"]
    assert overview.dist_figures == ["figures/numeric_dist_1.png"]

    overview = plot_numeric_qq(
        _qq_columns(13), figures_dir, evidence_path, source_path="data.csv"
    )
    assert overview.figures == ["figures/numeric_qq_1.png", "figures/numeric_qq_2.png"]
    assert overview.dist_figures == ["figures/numeric_dist_1.png", "figures/numeric_dist_2.png"]


def test_plot_numeric_qq_discrete_feature(tmp_path) -> None:
    figures_dir = tmp_path / "reports" / "figures"
    evidence_path = tmp_path / "artifacts" / "discover" / "qq_overview.json"
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "passenger_count": rng.integers(1, 6, 200).astype(float),
            "continuous": rng.normal(0.0, 1.0, 200),
        }
    )

    overview = plot_numeric_qq(df, figures_dir, evidence_path, source_path="data.csv")

    pc = next(f for f in overview.features if f.feature == "passenger_count")
    assert pc.status == "discrete"
    assert pc.figure == ""
    assert pc.dist_figure == "figures/numeric_dist_1.png"
    assert pc.reason == "discrete (5 unique values)"
    assert overview.discrete_features == 1
    assert overview.figures == ["figures/numeric_qq_1.png"]
    assert (figures_dir / "numeric_dist_1.png").exists()


def test_plot_numeric_qq_distribution_metrics_and_flags(tmp_path) -> None:
    figures_dir = tmp_path / "reports" / "figures"
    evidence_path = tmp_path / "artifacts" / "discover" / "qq_overview.json"
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "skewed_positive": np.exp(rng.normal(0.0, 1.0, 500)),
            "zero_inflated": np.concatenate(
                [np.zeros(300), rng.normal(1.0, 0.1, 200)]
            ),
        }
    )

    overview = plot_numeric_qq(df, figures_dir, evidence_path, source_path="data.csv")

    skewed = next(f for f in overview.features if f.feature == "skewed_positive")
    inflated = next(f for f in overview.features if f.feature == "zero_inflated")

    for f in (skewed, inflated):
        assert f.median is not None
        assert f.p99 is not None
        assert f.max is not None
        assert f.flags != []

    assert skewed.log_skew is not None
    assert inflated.log_skew is None
    assert any("skew" in fl for fl in skewed.flags)
    assert any("zero_rate" in fl for fl in inflated.flags)


def test_plot_numeric_qq_declared_id_excluded(tmp_path) -> None:
    figures_dir = tmp_path / "reports" / "figures"
    evidence_path = tmp_path / "artifacts" / "discover" / "qq_overview.json"
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "pickup_location_id": rng.integers(1, 265, 200),
            "continuous": rng.normal(0.0, 1.0, 200),
        }
    )

    overview = plot_numeric_qq(
        df, figures_dir, evidence_path, source_path="data.csv",
        exclude=["pickup_location_id"],
    )

    pid = next(f for f in overview.features if f.feature == "pickup_location_id")
    assert pid.status == "excluded"
    assert pid.reason == "declared id"
    assert pid.figure == ""
    assert pid.dist_figure == ""
    assert overview.excluded_features == 1
    assert "pickup_location_id: declared id" in overview.excluded_notes
    assert overview.flagged_id_columns == []


def test_plot_numeric_qq_id_heuristic_flag(tmp_path) -> None:
    figures_dir = tmp_path / "reports" / "figures"
    evidence_path = tmp_path / "artifacts" / "discover" / "qq_overview.json"
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "foo_id": rng.normal(0.0, 1.0, 200),
            "continuous": rng.normal(1.0, 1.0, 200),
        }
    )

    overview = plot_numeric_qq(df, figures_dir, evidence_path, source_path="data.csv")

    foo = next(f for f in overview.features if f.feature == "foo_id")
    assert foo.status == "plotted"
    assert foo.figure == "figures/numeric_qq_1.png"
    assert "foo_id" in overview.flagged_id_columns


def test_plot_numeric_qq_non_numeric_columns(tmp_path) -> None:
    figures_dir = tmp_path / "reports" / "figures"
    evidence_path = tmp_path / "artifacts" / "discover" / "qq_overview.json"
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "continuous": rng.normal(0.0, 1.0, 200),
            "when": pd.to_datetime("2024-01-01") + pd.to_timedelta(np.arange(200), unit="D"),
            "label": ["a", "b"] * 100,
        }
    )

    overview = plot_numeric_qq(df, figures_dir, evidence_path, source_path="data.csv")

    assert set(overview.non_numeric_columns) == {"when", "label"}
    plotted = {f.feature for f in overview.features}
    assert "when" not in plotted
    assert "label" not in plotted


def test_plot_numeric_qq_min_unique_env_override(tmp_path, monkeypatch) -> None:
    figures_dir = tmp_path / "reports" / "figures"
    evidence_path = tmp_path / "artifacts" / "discover" / "qq_overview.json"
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.integers(0, 20, 300).astype(float)})

    monkeypatch.setenv("BROADWAY_QQ_MIN_UNIQUE", "21")
    overview = plot_numeric_qq(df, figures_dir, evidence_path, source_path="data.csv")
    x = next(f for f in overview.features if f.feature == "x")
    assert x.status == "discrete"

    monkeypatch.delenv("BROADWAY_QQ_MIN_UNIQUE")
    overview = plot_numeric_qq(df, figures_dir, evidence_path, source_path="data.csv")
    x = next(f for f in overview.features if f.feature == "x")
    assert x.status == "plotted"


def test_plot_numeric_qq_source_path_is_data_path(tmp_path) -> None:
    figures_dir = tmp_path / "reports" / "figures"
    evidence_path = tmp_path / "artifacts" / "discover" / "qq_overview.json"

    overview = plot_numeric_qq(
        _qq_columns(2), figures_dir, evidence_path, source_path="data/raw/taxi.parquet"
    )

    assert overview.source_path == "data/raw/taxi.parquet"
    assert overview.source_path != str(evidence_path)


def test_discover_run_writes_qq_overview(tmp_path, monkeypatch) -> None:
    rng = np.random.default_rng(0)
    n = 40
    csv = tmp_path / "data.csv"
    pd.DataFrame(
        {"foo_id": np.arange(n), "value": rng.normal(0.0, 1.0, n)}
    ).to_csv(csv, index=False)
    monkeypatch.setattr(discover_module, "CONFIGS_DIR", tmp_path / "configs")
    monkeypatch.setattr(discover_module, "ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setattr(discover_module, "DATASET_DIR", "dataset")
    monkeypatch.setattr(discover_module, "FIGURES_DIR", tmp_path / "reports" / "figures")
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")

    discover_module.run(str(csv), "value", "regression")

    qq_path = tmp_path / "artifacts" / "discover" / "qq_overview.json"
    assert qq_path.exists()
    overview = QqOverview.model_validate_json(qq_path.read_text())
    assert overview.total_features == 2
    assert overview.source_path == str(csv)
    assert overview.flagged_id_columns == ["foo_id"]
    assert overview.discrete_features == 0
    assert (tmp_path / "reports" / "figures" / "numeric_qq_1.png").exists()
    assert (tmp_path / "reports" / "figures" / "numeric_dist_1.png").exists()


def test_discover_profile_writes_qq_overview(tmp_path, monkeypatch) -> None:
    data = pd.DataFrame({"a": [1, 2, 3], "t": [10.0, 20.0, 30.0]})
    parquet = tmp_path / "data.parquet"
    data.to_parquet(parquet)

    monkeypatch.setattr(discover_module, "CONFIGS_DIR", tmp_path / "configs")
    monkeypatch.setattr(discover_module, "ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setattr(discover_module, "DATASET_DIR", "dataset")
    monkeypatch.setattr(discover_module, "FIGURES_DIR", tmp_path / "reports" / "figures")
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")

    config_dir = tmp_path / "configs" / "dataset"
    config_dir.mkdir(parents=True)
    contract = {
        "name": "mydata",
        "path": str(parquet),
        "target": "t",
        "task": "regression",
        "datetime_column": None,
        "columns": {
            "a": {"dtype": "int64", "null_count": 0, "role": "feature"},
            "t": {"dtype": "float64", "null_count": 0, "role": "target"},
        },
        "lookup_tables": {},
        "exclude_from_profiling": ["a"],
    }
    (config_dir / "mydata.yaml").write_text(yaml.safe_dump(contract), encoding="utf-8")

    discover_module.profile("mydata")

    qq_path = tmp_path / "artifacts" / "discover" / "qq_overview.json"
    assert qq_path.exists()
    overview = QqOverview.model_validate_json(qq_path.read_text())
    assert overview.total_features == 2
    assert overview.source_path == str(parquet)
    a = next(f for f in overview.features if f.feature == "a")
    assert a.status == "excluded"
    assert a.reason == "declared id"
    assert overview.discrete_features == 1
    assert (tmp_path / "reports" / "figures" / "numeric_dist_1.png").exists()
    assert not (tmp_path / "reports" / "figures" / "numeric_qq_1.png").exists()


def test_render_profile_includes_qq_evidence() -> None:
    qq = QqOverview(
        source_path="data/raw/taxi.csv",
        total_features=5,
        plotted_features=2,
        excluded_features=1,
        discrete_features=1,
        non_numeric_columns=["when"],
        flagged_id_columns=["foo_id"],
        excluded_notes=["const: zero variance"],
        features=[
            QqFeature(
                feature="a", n_valid=4, n_excluded=0, mean=2.5, std=1.0,
                status="plotted", reason=None, figure="figures/numeric_qq_1.png",
                dist_figure="figures/numeric_dist_1.png",
                zero_rate=0.125, skew=0.5, kurtosis=-0.25,
                flags=["skew 0.50 exceeds 1.0"],
            ),
            QqFeature(
                feature="b", n_valid=4, n_excluded=0, mean=2.5, std=1.0,
                status="plotted", reason=None, figure="figures/numeric_qq_1.png",
                dist_figure="figures/numeric_dist_1.png",
                zero_rate=0.0, skew=-0.5, kurtosis=0.75,
            ),
            QqFeature(
                feature="const", n_valid=4, n_excluded=0, mean=5.0, std=0.0,
                status="excluded", reason="zero variance", figure="", dist_figure="",
            ),
            QqFeature(
                feature="disc", n_valid=4, n_excluded=0, mean=1.5, std=0.5,
                status="discrete", reason="discrete (3 unique values)", figure="",
                dist_figure="figures/numeric_dist_1.png",
                zero_rate=0.333, skew=0.0, kurtosis=-1.2,
            ),
        ],
        figures=["figures/numeric_qq_1.png"],
        dist_figures=["figures/numeric_dist_1.png"],
        diagnostics_figures=["figures/numeric_diagnostics.png"],
        sample_size=10000,
    )
    md = audit.render_profile(_profile(), source="artifacts/discover/profile.json", qq=qq)
    assert "## Profile evidence" in md
    assert "![Per-feature Q-Q plots — figure 1 of 1](../figures/numeric_qq_1.png)" in md
    assert "![Per-feature distributions — figure 1 of 1](../figures/numeric_dist_1.png)" in md
    assert "In this figure: a, b." in md
    assert "In this figure: a, b, disc." in md
    assert "Traces are per-feature z-score." in md
    assert "Sample size: n = 10,000" in md
    assert "Histograms are in raw units." in md
    assert "How to read (Q-Q)" in md
    assert "fitted reference line" in md
    assert "How to read (distribution)" in md
    assert "### Distribution diagnostics" in md
    assert "| Variable | n | mean | std | skew | excess_kurtosis | zero_rate | p99/median | max/median | log_skew |" in md
    assert "| a | 4 | 2.5 | 1 | 0.5 | -0.25 | 0.125 | - | - | - |" in md
    assert "| b | 4 | 2.5 | 1 | -0.5 | 0.75 | 0.000 | - | - | - |" in md
    assert "| disc | 4 | 1.5 | 0.5 | 0 | -1.2 | 0.333 | - | - | - |" in md
    assert "![Per-feature distribution diagnostics — figure 1 of 1](../figures/numeric_diagnostics.png)" in md
    assert "How to read (diagnostics)" in md
    assert "### Decision flags" in md
    assert "- a: skew 0.50 exceeds 1.0" in md
    assert "- const: zero variance" in md
    assert "- disc: discrete (3 unique values)" in md
    assert "- foo_id: name suggests an identifier" in md
    assert "- not profiled (non-numeric): when" in md
    assert "![a, b]" not in md
    assert "{" not in md
    assert "}" not in md


def test_render_profile_diagnostics_absent_without_figures() -> None:
    qq = QqOverview(
        source_path="data/raw/taxi.csv",
        total_features=1,
        plotted_features=1,
        excluded_features=0,
        discrete_features=0,
        non_numeric_columns=[],
        flagged_id_columns=[],
        excluded_notes=[],
        features=[
            QqFeature(
                feature="a", n_valid=4, n_excluded=0, mean=2.5, std=1.0,
                status="plotted", reason=None, figure="figures/numeric_qq_1.png",
                dist_figure="figures/numeric_dist_1.png",
                zero_rate=0.125, skew=0.5, kurtosis=-0.25,
            ),
        ],
        figures=["figures/numeric_qq_1.png"],
        dist_figures=["figures/numeric_dist_1.png"],
        diagnostics_figures=[],
    )
    md = audit.render_profile(_profile(), source="artifacts/discover/profile.json", qq=qq)
    assert "### Distribution diagnostics" in md
    assert "![Per-feature distribution diagnostics" not in md
    assert "How to read (diagnostics)" not in md


def test_render_profile_without_qq_has_no_evidence_section() -> None:
    md = audit.render_profile(_profile(), source="artifacts/discover/profile.json")
    assert "## Profile evidence" not in md


def test_plot_numeric_qq_default_palette_is_shared() -> None:
    sig = inspect.signature(plot_numeric_qq)
    assert sig.parameters["palette"].default is None
    assert load_viz_config().palette == viz.default_palette()


def test_plot_numeric_qq_defaults_come_from_config() -> None:
    sig = inspect.signature(plot_numeric_qq)
    cfg = load_viz_config()
    assert sig.parameters["max_features_per_figure"].default is None
    assert sig.parameters["fig_size_per_subplot"].default is None
    assert sig.parameters["dpi"].default is None
    assert sig.parameters["max_points_per_trace"].default is None
    assert sig.parameters["sample_size"].default is None
    assert sig.parameters["random_state"].default is None
    assert cfg.max_points_per_trace == 10000
    assert cfg.max_features_per_figure == 12
    assert cfg.fig_size_per_subplot == 3.0
    assert cfg.dpi == 100
    assert cfg.min_unique_for_qq == 15
    assert cfg.qq_sample_size == 10000
    assert cfg.qq_random_state == 42


def test_plot_numeric_qq_downsamples_to_qq_sample_size(tmp_path) -> None:
    figures_dir = tmp_path / "reports" / "figures"
    evidence_path = tmp_path / "artifacts" / "discover" / "qq_overview.json"
    rng = np.random.default_rng(0)
    n = 50_000
    df = pd.DataFrame({"x": rng.normal(0.0, 1.0, n)})

    overview = plot_numeric_qq(df, figures_dir, evidence_path, source_path="data.csv")

    qq_sample_size = load_viz_config().qq_sample_size
    assert overview.sample_size == qq_sample_size
    assert overview.sample_size == min(n, qq_sample_size)
    x = next(f for f in overview.features if f.feature == "x")
    assert x.n_valid == qq_sample_size


def test_qq_subplot_titles_are_feature_names_only(tmp_path, monkeypatch) -> None:
    import broadway.discover.qq as qq_module

    captured: list = []
    monkeypatch.setattr(qq_module.plt, "close", lambda fig: captured.append(fig))
    traces = [
        ("my_feature", np.array([-1.0, 0.0, 1.0]), np.array([-1.0, 0.0, 1.0]), 1.0, 0.0, 0.05)
    ]
    _plot_chunk(
        traces, tmp_path / "qq_titles.png", 1, 1, 3.0, 100, "BuPu_r", 10000,
        load_viz_config().qq_zones,
    )
    titles = [ax.get_title() for ax in captured[-1].axes]
    assert "my_feature" in titles
    assert not any("(n=" in t for t in titles)


def test_qq_zones_config_loaded() -> None:
    zones = load_viz_config().qq_zones
    assert zones.enabled is True
    assert zones.central_quantiles == [0.25, 0.75]
    assert zones.tail_threshold == 1.96
    assert zones.zero_mass_threshold == 0.05
    assert zones.central_alpha == 0.15
    assert zones.tail_alpha == 0.08
    assert zones.shelf_color == "#d62728"
    with pytest.raises(ValidationError):
        QqZonesConfig(
            enabled=True,
            central_quantiles=[0.25, 0.75],
            tail_threshold=1.96,
            zero_mass_threshold=0.05,
            central_alpha=0.15,
            tail_alpha=0.08,
            zone_color="#999999",
            shelf_color="#d62728",
            unknown_key="nope",
        )


def test_qq_zones_enabled_draws_bands(tmp_path, monkeypatch) -> None:
    import broadway.discover.qq as qq_module

    captured: list = []
    monkeypatch.setattr(qq_module.plt, "close", lambda fig: captured.append(fig))
    zones = load_viz_config().qq_zones
    traces = [
        ("f", np.array([-1.0, 0.0, 1.0]), np.array([-1.0, 0.0, 1.0]), 1.0, 0.0, 0.0)
    ]
    _plot_chunk(traces, tmp_path / "zones_on.png", 1, 1, 3.0, 100, "BuPu_r", 10000, zones)
    ax = captured[-1].axes[0]
    assert len(ax.patches) >= 2
    assert all(p.get_zorder() == 0 for p in ax.patches)


def test_qq_zones_disabled(tmp_path, monkeypatch) -> None:
    import broadway.discover.qq as qq_module

    captured: list = []
    monkeypatch.setattr(qq_module.plt, "close", lambda fig: captured.append(fig))
    zones = load_viz_config().qq_zones.model_copy(update={"enabled": False})
    traces = [
        ("f", np.array([-1.0, 0.0, 1.0]), np.array([-1.0, 0.0, 1.0]), 1.0, 0.0, 0.05)
    ]
    _plot_chunk(traces, tmp_path / "zones_off.png", 1, 1, 3.0, 100, "BuPu_r", 10000, zones)
    ax = captured[-1].axes[0]
    assert len(ax.patches) == 0
    assert len(ax.lines) == 1


def test_qq_zero_mass_shelf_drawn(tmp_path, monkeypatch) -> None:
    import broadway.discover.qq as qq_module

    zones = load_viz_config().qq_zones
    trace_high = ("f", np.array([-1.0, 0.0, 1.0]), np.array([-1.0, 0.0, 1.0]), 1.0, 0.0, 0.5)
    trace_low = ("f", np.array([-1.0, 0.0, 1.0]), np.array([-1.0, 0.0, 1.0]), 1.0, 0.0, 0.01)

    captured: list = []
    monkeypatch.setattr(qq_module.plt, "close", lambda fig: captured.append(fig))
    _plot_chunk([trace_high], tmp_path / "shelf_on.png", 1, 1, 3.0, 100, "BuPu_r", 10000, zones)
    ax = captured[-1].axes[0]
    assert len(ax.lines) == 2
    assert any(l.get_color() == zones.shelf_color for l in ax.lines)

    _plot_chunk([trace_low], tmp_path / "shelf_off.png", 1, 1, 3.0, 100, "BuPu_r", 10000, zones)
    ax = captured[-1].axes[0]
    assert len(ax.lines) == 1


def test_qq_zero_mass_shelf_backward_compat(tmp_path, monkeypatch) -> None:
    import broadway.discover.qq as qq_module

    captured: list = []
    monkeypatch.setattr(qq_module.plt, "close", lambda fig: captured.append(fig))
    zones = load_viz_config().qq_zones
    traces = [
        ("f", np.array([-1.0, 0.0, 1.0]), np.array([-1.0, 0.0, 1.0]), 1.0, 0.0, None)
    ]
    _plot_chunk(traces, tmp_path / "shelf_none.png", 1, 1, 3.0, 100, "BuPu_r", 10000, zones)
    ax = captured[-1].axes[0]
    assert len(ax.lines) == 1


def test_qq_figure_legend(tmp_path, monkeypatch) -> None:
    import broadway.discover.qq as qq_module

    zones = load_viz_config().qq_zones
    captured: list = []
    monkeypatch.setattr(qq_module.plt, "close", lambda fig: captured.append(fig))

    shelf_trace = ("f", np.array([-1.0, 0.0, 1.0]), np.array([-1.0, 0.0, 1.0]), 1.0, 0.0, 0.5)
    _plot_chunk([shelf_trace], tmp_path / "legend_shelf.png", 1, 1, 3.0, 100, "BuPu_r", 10000, zones)
    fig = captured[-1]
    assert len(fig.legends) == 1
    assert len(fig.legends[0].legend_handles) == 3

    no_shelf_trace = ("f", np.array([-1.0, 0.0, 1.0]), np.array([-1.0, 0.0, 1.0]), 1.0, 0.0, 0.0)
    _plot_chunk([no_shelf_trace], tmp_path / "legend_no_shelf.png", 1, 1, 3.0, 100, "BuPu_r", 10000, zones)
    fig = captured[-1]
    assert len(fig.legends) == 1
    assert len(fig.legends[0].legend_handles) == 2


def test_qq_zones_on_joint(tmp_path, monkeypatch) -> None:
    import broadway.timeline.runners as runners_module

    rng = np.random.default_rng(0)
    groups = {g: rng.normal(float(i), 1.0, 30) for i, g in enumerate(["A", "B", "C"])}
    captured: list = []
    monkeypatch.setattr(runners_module.plt, "close", lambda fig: captured.append(fig))
    runners._plot_qq_joint(groups, tmp_path / "joint.png")
    fig = captured[-1]
    for ax in fig.axes:
        if ax.get_visible():
            assert len(ax.patches) >= 2
    assert len(fig.legends) == 1
    assert len(fig.legends[0].legend_handles) == 2


def test_dist_subplot_titles_are_feature_names_only(tmp_path, monkeypatch) -> None:
    import broadway.discover.qq as qq_module

    captured: list = []
    monkeypatch.setattr(qq_module.plt, "close", lambda fig: captured.append(fig))
    hists = [("my_feature", np.array([1, 2, 3]), np.array([0.0, 1.0, 2.0, 3.0]))]
    qq_module._plot_dist_chunk(hists, tmp_path / "dist_titles.png", 1, 1, 3.0, 100, "BuPu_r", 10000)
    titles = [ax.get_title() for ax in captured[-1].axes]
    assert "my_feature" in titles
    assert not any("(n=" in t for t in titles)
