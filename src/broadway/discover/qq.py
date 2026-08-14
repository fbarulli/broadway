"""Flexible, data-agnostic per-feature Q-Q plots as small multiples.

- Auto-detects numeric columns or accepts an explicit list
- Records (never plots) non-finite / zero-variance features
- Declared identifiers are excluded from every grid
- Low-cardinality (discrete) features are excluded from the Q-Q grid but kept
  as integer-aligned bar charts in the distribution grid
- One subplot per feature with a y=x reference; figure scales with the grid
- Chunks into multiple PNGs beyond `max_features_per_figure`
- Does NOT mutate the input DataFrame
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless; set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from scipy import stats

from broadway import viz
from broadway.config.viz import load_viz_config


class QqFeature(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feature: str
    n_valid: int
    n_excluded: int
    mean: float | None
    std: float | None
    status: str                # "plotted" | "discrete" | "excluded"
    reason: str | None
    figure: str
    dist_figure: str = ""


class QqOverview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_path: str
    total_features: int
    plotted_features: int
    excluded_features: int
    excluded_notes: list[str]
    features: list[QqFeature]
    figures: list[str]
    dist_figures: list[str] = []
    standardization: str = "per-feature z-score"
    discrete_features: int = 0
    non_numeric_columns: list[str] = []
    flagged_id_columns: list[str] = []


def _numeric_cols(df: pd.DataFrame, cols: list[str] | None) -> list[str]:
    if cols is not None:
        return [c for c in cols if c in df.columns]
    return list(df.select_dtypes(include="number").columns)


def _resolve_min_unique(override: int | None, default: int) -> int:
    if override is not None:
        return override
    return int(os.getenv("BROADWAY_QQ_MIN_UNIQUE", str(default)))


def _qq_points(
    vals: np.ndarray,
    max_points_per_trace: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Standardized Q-Q points for one feature, thinned for plotting, plus fit line."""
    if vals.size < 2:
        return np.array([]), np.array([]), 0.0, 0.0
    z = (vals - vals.mean()) / vals.std(ddof=0)
    (osm, osr), (slope, intercept, _) = stats.probplot(z, dist="norm", fit=True)
    if osm.size > max_points_per_trace:
        idx = np.linspace(0, osm.size - 1, max_points_per_trace).astype(int)
        osm, osr = osm[idx], osr[idx]
    return osm, osr, float(slope), float(intercept)


def _grid_dims(n: int) -> tuple[int, int]:
    n_cols = max(1, int(np.ceil(np.sqrt(n))))
    n_rows = max(1, int(np.ceil(n / n_cols)))
    return n_rows, n_cols


def _plot_chunk(
    traces: list[tuple[str, np.ndarray, np.ndarray, float, float]],
    out_path: Path,
    fig_num: int,
    n_figs: int,
    subplot_size: float,
    dpi: int,
    palette: str,
) -> None:
    n = len(traces)
    n_rows, n_cols = _grid_dims(n)
    colors = viz.palette_colors(n, palette)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * subplot_size, n_rows * subplot_size),
        squeeze=False,
        layout="constrained",
    )
    ax_flat = axes.ravel()
    for color, ax, (name, osm, osr, slope, intercept) in zip(colors, ax_flat, traces):
        ax.scatter(
            osm, osr,
            s=viz.QQ_SCATTER_SIZE,
            alpha=viz.QQ_SCATTER_ALPHA,
            edgecolor=viz.QQ_SCATTER_EDGE_COLOR,
            color=color,
        )
        xs = np.array([osm.min(), osm.max()])
        ax.plot(
            xs, slope * xs + intercept,
            color=viz.QQ_REF_LINE_COLOR,
            linestyle=viz.QQ_REF_LINE_STYLE,
            linewidth=viz.QQ_REF_LINE_WIDTH,
        )
        ax.set_xlabel(viz.QQ_XLABEL, fontsize=viz.LABEL_FONTSIZE)
        ax.set_ylabel(viz.QQ_YLABEL, fontsize=viz.LABEL_FONTSIZE)
        ax.set_title(name, fontsize=viz.TITLE_FONTSIZE)
        ax.grid(True, alpha=viz.GRID_ALPHA)
        ax.tick_params(labelsize=viz.TICK_FONTSIZE)
        viz.despine(ax)
    for ax in ax_flat[n:]:
        ax.set_visible(False)
    title = "Per-feature Q-Q plots (standardized)"
    if n_figs > 1:
        title += f" - figure {fig_num} of {n_figs}"
    fig.suptitle(title, fontsize=viz.SUPTITLE_FONTSIZE)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_dist_chunk(
    hists: list[tuple[str, np.ndarray, np.ndarray]],
    out_path: Path,
    fig_num: int,
    n_figs: int,
    subplot_size: float,
    dpi: int,
    palette: str,
) -> None:
    n = len(hists)
    n_rows, n_cols = _grid_dims(n)
    colors = viz.palette_colors(n, palette)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * subplot_size, n_rows * subplot_size),
        squeeze=False,
        layout="constrained",
    )
    ax_flat = axes.ravel()
    for color, ax, (name, counts, edges) in zip(colors, ax_flat, hists):
        ax.stairs(counts, edges, fill=True, color=color)
        ax.set_xlabel("Value (raw units)", fontsize=viz.LABEL_FONTSIZE)
        ax.set_ylabel("Count", fontsize=viz.LABEL_FONTSIZE)
        ax.set_title(name, fontsize=viz.TITLE_FONTSIZE)
        ax.grid(True, alpha=viz.GRID_ALPHA)
        ax.tick_params(labelsize=viz.TICK_FONTSIZE)
        viz.despine(ax)
    for ax in ax_flat[n:]:
        ax.set_visible(False)
    title = "Per-feature distributions (raw units)"
    if n_figs > 1:
        title += f" - figure {fig_num} of {n_figs}"
    fig.suptitle(title, fontsize=viz.SUPTITLE_FONTSIZE)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_numeric_qq(
    df: pd.DataFrame,
    figures_dir: Path,
    evidence_path: Path,
    *,
    source_path: str,
    cols: list[str] | None = None,
    exclude: list[str] | None = None,
    min_unique_for_qq: int | None = None,
    max_features_per_figure: int | None = None,
    fig_size_per_subplot: float | None = None,
    dpi: int | None = None,
    max_points_per_trace: int | None = None,
    standardization: str = "per-feature z-score",
    palette: str | None = None,
) -> QqOverview:
    cfg = load_viz_config()
    figures_dir = Path(figures_dir)
    evidence_path = Path(evidence_path)
    exclude = exclude or []
    if max_features_per_figure is None:
        max_features_per_figure = cfg.max_features_per_figure
    if fig_size_per_subplot is None:
        fig_size_per_subplot = cfg.fig_size_per_subplot
    if dpi is None:
        dpi = cfg.dpi
    if max_points_per_trace is None:
        max_points_per_trace = cfg.max_points_per_trace
    if palette is None:
        palette = cfg.palette
    min_unique = _resolve_min_unique(min_unique_for_qq, cfg.min_unique_for_qq)

    features: list[QqFeature] = []
    traces: dict[str, tuple[np.ndarray, np.ndarray, float, float]] = {}
    hists: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    excluded_notes: list[str] = []
    flagged_id_columns: list[str] = []

    numeric_cols = _numeric_cols(df, cols)
    non_numeric_columns = [c for c in df.columns if c not in numeric_cols]

    for name in numeric_cols:
        if name in exclude:
            excluded_notes.append(f"{name}: declared id")
            features.append(QqFeature(
                feature=name, n_valid=0, n_excluded=0,
                mean=None, std=None, status="excluded", reason="declared id", figure="",
            ))
            continue

        arr = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        finite = arr[np.isfinite(arr)]
        n_valid = int(finite.size)
        n_excluded = int(arr.size) - n_valid

        if name.endswith("_id"):
            flagged_id_columns.append(name)

        if n_valid < 2:
            reason = "non-finite" if n_valid == 0 else "insufficient samples"
            excluded_notes.append(f"{name}: {reason}")
            features.append(QqFeature(
                feature=name, n_valid=n_valid, n_excluded=n_excluded,
                mean=None, std=None, status="excluded", reason=reason, figure="",
            ))
            continue

        std_val = float(finite.std(ddof=0))
        if std_val == 0.0:
            excluded_notes.append(f"{name}: zero variance")
            features.append(QqFeature(
                feature=name, n_valid=n_valid, n_excluded=n_excluded,
                mean=float(finite.mean()), std=0.0, status="excluded",
                reason="zero variance", figure="",
            ))
            continue

        n_unique = int(np.unique(finite).size)
        mean_val = float(finite.mean())
        if n_unique <= min_unique:
            lo = int(np.floor(finite.min()))
            hi = int(np.ceil(finite.max()))
            counts, edges = np.histogram(finite, bins=np.arange(lo, hi + 2))
            hists[name] = (counts, edges)
            features.append(QqFeature(
                feature=name, n_valid=n_valid, n_excluded=n_excluded,
                mean=mean_val, std=std_val, status="discrete",
                reason=f"discrete ({n_unique} unique values)", figure="",
            ))
            continue

        osm, osr, slope, intercept = _qq_points(finite, max_points_per_trace)
        traces[name] = (osm, osr, slope, intercept)
        counts, edges = np.histogram(finite, bins="auto")
        hists[name] = (counts, edges)
        features.append(QqFeature(
            feature=name, n_valid=n_valid, n_excluded=n_excluded,
            mean=mean_val, std=std_val, status="plotted",
            reason=None, figure="",
        ))

    figures_dir.mkdir(parents=True, exist_ok=True)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)

    # Chunk plotable features; assign figure names by NAME (not position),
    # so excluded features can't misalign the mapping.
    plot_names = [f.feature for f in features if f.status == "plotted"]
    dist_names = [f.feature for f in features if f.status in ("plotted", "discrete")]
    qq_chunks = [
        plot_names[i : i + max_features_per_figure]
        for i in range(0, len(plot_names), max_features_per_figure)
    ]
    dist_chunks = [
        dist_names[i : i + max_features_per_figure]
        for i in range(0, len(dist_names), max_features_per_figure)
    ]
    name_to_idx = {f.feature: i for i, f in enumerate(features)}

    figures: list[str] = []
    for fig_num, chunk in enumerate(qq_chunks, start=1):
        basename = cfg.qq_figure.format(fig_num=fig_num)
        fig_name = f"figures/{basename}"
        figures.append(fig_name)
        _plot_chunk(
            [(n, *traces[n]) for n in chunk],
            figures_dir / basename,
            fig_num, len(qq_chunks), fig_size_per_subplot, dpi, palette,
        )
        for n in chunk:
            i = name_to_idx[n]
            features[i] = features[i].model_copy(update={"figure": fig_name})

    dist_figures: list[str] = []
    for fig_num, chunk in enumerate(dist_chunks, start=1):
        basename = cfg.dist_figure.format(fig_num=fig_num)
        dist_name = f"figures/{basename}"
        dist_figures.append(dist_name)
        _plot_dist_chunk(
            [(n, *hists[n]) for n in chunk],
            figures_dir / basename,
            fig_num, len(dist_chunks), fig_size_per_subplot, dpi, palette,
        )
        for n in chunk:
            i = name_to_idx[n]
            features[i] = features[i].model_copy(update={"dist_figure": dist_name})

    overview = QqOverview(
        source_path=source_path,
        total_features=len(features),
        plotted_features=len(plot_names),
        excluded_features=sum(1 for f in features if f.status == "excluded"),
        discrete_features=sum(1 for f in features if f.status == "discrete"),
        non_numeric_columns=non_numeric_columns,
        flagged_id_columns=flagged_id_columns,
        excluded_notes=excluded_notes,
        features=features,
        figures=figures,
        dist_figures=dist_figures,
        standardization=standardization,
    )
    evidence_path.write_text(overview.model_dump_json(indent=2), encoding="utf-8")
    return overview


if __name__ == "__main__":
    np.random.seed(42)
    n_rows = 200
    df = pd.DataFrame(
        {f"feat_{i:02d}": np.random.normal(i * 0.5, 1.0, n_rows) for i in range(22)}
    )
    df["feat_05"] = 3.0        # zero variance -> excluded
    df["feat_12"] = np.nan     # non-finite   -> excluded

    overview = plot_numeric_qq(
        df, Path("qq_demo_output"), Path("qq_demo_overview.json"),
        source_path="<demo>",
    )
    print(f"Generated {len(overview.figures)} figures for "
          f"{overview.plotted_features}/{overview.total_features} features")
    print(f"Excluded: {overview.excluded_notes}")