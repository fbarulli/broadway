from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import broadway.discover.module as discover_module
from broadway.config.loader import load_config
from broadway.discover.profile import ColumnProfile, DatasetProfile
from broadway.discover.qq import QqFeature, QqOverview, plot_numeric_qq
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

    assert overview.total_features == 17
    assert overview.excluded_features == 2
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
    assert (figures_dir / "numeric_qq_1.png").exists()
    assert (figures_dir / "numeric_qq_2.png").exists()
    assert (figures_dir / "numeric_dist_1.png").exists()
    assert (figures_dir / "numeric_dist_2.png").exists()

    by_figure: dict[str, list[str]] = {}
    dist_by_figure: dict[str, list[str]] = {}
    for f in overview.features:
        if f.figure:
            by_figure.setdefault(f.figure, []).append(f.feature)
        if f.dist_figure:
            dist_by_figure.setdefault(f.dist_figure, []).append(f.feature)
    assert len(by_figure["figures/numeric_qq_1.png"]) == 12
    assert len(by_figure["figures/numeric_qq_2.png"]) == 3
    assert len(dist_by_figure["figures/numeric_dist_1.png"]) == 12
    assert len(dist_by_figure["figures/numeric_dist_2.png"]) == 3

    mixed = next(f for f in overview.features if f.feature == "mixed")
    assert mixed.n_valid == 19
    assert mixed.n_excluded == 1

    f0 = next(f for f in overview.features if f.feature == "f0")
    vals = df["f0"].to_numpy(dtype=float)
    assert f0.n_valid == 20
    assert f0.n_excluded == 0
    assert f0.mean == pytest.approx(float(np.mean(vals)))
    assert f0.std == pytest.approx(float(np.std(vals)))

    for f in overview.features:
        if f.status == "plotted":
            assert f.figure and f.dist_figure

    excluded = {f.feature for f in overview.features if f.figure == ""}
    assert excluded == {"allnan", "const"}

    reloaded = QqOverview.model_validate_json(evidence_path.read_text())
    assert reloaded.figures == overview.figures
    assert reloaded.dist_figures == overview.dist_figures


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


def test_plot_numeric_qq_source_path_is_data_path(tmp_path) -> None:
    figures_dir = tmp_path / "reports" / "figures"
    evidence_path = tmp_path / "artifacts" / "discover" / "qq_overview.json"

    overview = plot_numeric_qq(
        _qq_columns(2), figures_dir, evidence_path, source_path="data/raw/taxi.parquet"
    )

    assert overview.source_path == "data/raw/taxi.parquet"
    assert overview.source_path != str(evidence_path)


def test_discover_run_writes_qq_overview(tmp_path, monkeypatch) -> None:
    csv = tmp_path / "data.csv"
    pd.DataFrame(
        {"id": [1, 2, 3, 4], "value": [10.0, 20.0, 30.0, 40.0]}
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
        "row_count": 3,
    }
    (config_dir / "mydata.yaml").write_text(yaml.safe_dump(contract), encoding="utf-8")

    discover_module.profile("mydata")

    qq_path = tmp_path / "artifacts" / "discover" / "qq_overview.json"
    assert qq_path.exists()
    overview = QqOverview.model_validate_json(qq_path.read_text())
    assert overview.total_features == 2
    assert overview.source_path == str(parquet)
    assert (tmp_path / "reports" / "figures" / "numeric_qq_1.png").exists()
    assert (tmp_path / "reports" / "figures" / "numeric_dist_1.png").exists()


def test_render_profile_includes_qq_evidence() -> None:
    qq = QqOverview(
        source_path="data/raw/taxi.csv",
        total_features=3,
        plotted_features=2,
        excluded_features=1,
        excluded_notes=["const: zero variance"],
        features=[
            QqFeature(
                feature="a", n_valid=4, n_excluded=0, mean=2.5, std=1.0,
                status="plotted", reason=None, figure="figures/numeric_qq_1.png",
                dist_figure="figures/numeric_dist_1.png",
            ),
            QqFeature(
                feature="b", n_valid=4, n_excluded=0, mean=2.5, std=1.0,
                status="plotted", reason=None, figure="figures/numeric_qq_1.png",
                dist_figure="figures/numeric_dist_1.png",
            ),
        ],
        figures=["figures/numeric_qq_1.png"],
        dist_figures=["figures/numeric_dist_1.png"],
    )
    md = audit.render_profile(_profile(), source="artifacts/discover/profile.json", qq=qq)
    assert "## Profile evidence" in md
    assert "![a, b](../figures/numeric_qq_1.png)" in md
    assert "![a, b](../figures/numeric_dist_1.png)" in md
    assert "Traces are per-feature z-score." in md
    assert "Histograms are in raw units." in md
    assert "- const: zero variance" in md
    assert "How to read (Q-Q)" in md
    assert "How to read (distribution)" in md


def test_render_profile_without_qq_has_no_evidence_section() -> None:
    md = audit.render_profile(_profile(), source="artifacts/discover/profile.json")
    assert "## Profile evidence" not in md
