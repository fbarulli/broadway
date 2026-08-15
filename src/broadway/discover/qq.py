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
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from scipy import stats

from broadway import viz
from broadway.config.viz import DiagnosticsConfig, QqZonesConfig, load_viz_config


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
    zero_rate: float | None = None
    skew: float | None = None
    kurtosis: float | None = None
    median: float | None = None
    p99: float | None = None
    max: float | None = None
    log_skew: float | None = None
    flags: list[str] = []


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
    diagnostics_figures: list[str] = []
    standardization: str = "per-feature z-score"
    discrete_features: int = 0
    non_numeric_columns: list[str] = []
    flagged_id_columns: list[str] = []
    sample_size: int | None = None


def midpoint_bin_edges(values: np.ndarray) -> np.ndarray:
    """Bin edges centered on the observed unique values of a discrete feature."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    uniq = np.unique(vals)
    if uniq.size == 0:
        raise ValueError("cannot build midpoint bin edges from no finite values")
    if uniq.size == 1:
        return np.array([uniq[0] - 0.5, uniq[0] + 0.5])
    interior = (uniq[:-1] + uniq[1:]) / 2
    left = uniq[0] - (uniq[1] - uniq[0]) / 2
    right = uniq[-1] + (uniq[-1] - uniq[-2]) / 2
    return np.concatenate([[left], interior, [right]])


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
    traces: list[tuple[str, np.ndarray, np.ndarray, float, float, float | None]],
    out_path: Path,
    fig_num: int,
    n_figs: int,
    subplot_size: float,
    dpi: int,
    palette: str,
    n_rows: int,
    zones: QqZonesConfig,
) -> None:
    n = len(traces)
    grid_rows, grid_cols = _grid_dims(n)
    colors = viz.palette_colors(n, palette)
    fig, axes = plt.subplots(
        grid_rows, grid_cols,
        figsize=(grid_cols * subplot_size, grid_rows * subplot_size),
        squeeze=False,
        layout="constrained",
    )
    ax_flat = axes.ravel()
    for color, ax, (name, osm, osr, slope, intercept, zero_rate) in zip(colors, ax_flat, traces):
        ax.scatter(
            osm, osr,
            s=viz.QQ_SCATTER_SIZE,
            alpha=viz.QQ_SCATTER_ALPHA,
            edgecolor=viz.QQ_SCATTER_EDGE_COLOR,
            color=color,
            zorder=3,
        )
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xs = np.array([osm.min(), osm.max()])
        ax.plot(
            xs, slope * xs + intercept,
            color=viz.QQ_REF_LINE_COLOR,
            linestyle=viz.QQ_REF_LINE_STYLE,
            linewidth=viz.QQ_REF_LINE_WIDTH,
            zorder=2,
        )
        if zones.enabled:
            ax.axvspan(
                xmin, -zones.tail_threshold,
                color=zones.zone_color, alpha=zones.tail_alpha, zorder=0,
            )
            ax.axvspan(
                zones.tail_threshold, xmax,
                color=zones.zone_color, alpha=zones.tail_alpha, zorder=0,
            )
            ax.axvspan(
                stats.norm.ppf(zones.central_quantiles[0]),
                stats.norm.ppf(zones.central_quantiles[1]),
                color=zones.zone_color, alpha=zones.central_alpha, zorder=0,
            )
            if (
                zero_rate is not None
                and zero_rate > zones.zero_mass_threshold
                and ymin <= 0 <= ymax
            ):
                ax.axhline(
                    y=0, color=zones.shelf_color,
                    linestyle="--", linewidth=1, alpha=0.7, zorder=2,
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
    title += f" — n = {n_rows:,}"
    mid = int((zones.central_quantiles[1] - zones.central_quantiles[0]) * 100)
    title += f"\nshaded = middle {mid}% (centre) / ±{zones.tail_threshold}σ tails; red dashed h-line = zero-mass shelf"
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
    n_rows: int,
) -> None:
    n = len(hists)
    grid_rows, grid_cols = _grid_dims(n)
    colors = viz.palette_colors(n, palette)
    fig, axes = plt.subplots(
        grid_rows, grid_cols,
        figsize=(grid_cols * subplot_size, grid_rows * subplot_size),
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
    title += f" — n = {n_rows:,}"
    fig.suptitle(title, fontsize=viz.SUPTITLE_FONTSIZE)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_diagnostics_heatmap(
    rows: list[tuple[str, float, float, float]],
    out_path: Path,
    dpi: int,
    diag_cfg: DiagnosticsConfig,
) -> None:
    n_features = len(rows)
    names = [r[0] for r in rows]
    raw = np.array([[r[1], r[2], r[3]] for r in rows], dtype=float)
    z = np.zeros_like(raw)
    for col in range(raw.shape[1]):
        col_vals = raw[:, col]
        col_std = float(col_vals.std(ddof=0))
        if col_std == 0.0:
            z[:, col] = 0.0
        else:
            z[:, col] = (col_vals - col_vals.mean()) / col_std
    fig, ax = plt.subplots(
        figsize=(8.0, max(2.0, 0.35 * n_features)), layout="constrained",
    )
    ax.imshow(
        z, aspect="auto", cmap=diag_cfg.colormap,
        norm=matplotlib.colors.TwoSlopeNorm(vcenter=0),
    )
    ax.set_xticks(range(raw.shape[1]))
    ax.set_xticklabels(["skew", "kurtosis", "zero_rate"], fontsize=viz.TICK_FONTSIZE)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(names, fontsize=viz.TICK_FONTSIZE)
    if diag_cfg.annotate:
        for i in range(n_features):
            for j in range(raw.shape[1]):
                ax.text(
                    j, i, f"{raw[i, j]:.2f}",
                    ha="center", va="center", fontsize=viz.TICK_FONTSIZE,
                )
    fig.colorbar(ax.images[0], ax=ax)
    viz.despine(ax)
    fig.suptitle(
        "Per-feature distribution diagnostics (per-column z-score)",
        fontsize=viz.SUPTITLE_FONTSIZE,
    )
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
    sample_size: int | None = None,
    random_state: int | None = None,
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
    if sample_size is None:
        sample_size = cfg.qq_sample_size
    if random_state is None:
        random_state = cfg.qq_random_state
    thresholds = cfg.diagnostics.thresholds

    features: list[QqFeature] = []
    traces: dict[str, tuple[np.ndarray, np.ndarray, float, float, float | None]] = {}
    hists: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    excluded_notes: list[str] = []
    flagged_id_columns: list[str] = []

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)
    n_rows = int(len(df))

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
        zero_rate = float((finite == 0).mean())
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
                zero_rate=zero_rate,
            ))
            continue

        std_val = float(finite.std(ddof=0))
        if std_val == 0.0:
            excluded_notes.append(f"{name}: zero variance")
            features.append(QqFeature(
                feature=name, n_valid=n_valid, n_excluded=n_excluded,
                mean=float(finite.mean()), std=0.0, status="excluded",
                reason="zero variance", figure="", zero_rate=zero_rate,
            ))
            continue

        n_unique = int(np.unique(finite).size)
        mean_val = float(finite.mean())
        skew_val = float(stats.skew(finite))
        kurt_val = float(stats.kurtosis(finite))
        median_val = float(np.median(finite))
        p99_val = float(np.percentile(finite, 99))
        max_val = float(finite.max())
        log_skew_val = (
            float(stats.skew(np.log(finite)))
            if (skew_val > thresholds.skew and finite.min() > 0)
            else None
        )
        flags: list[str] = []
        if zero_rate > thresholds.zero_rate:
            flags.append(f"zero_rate {zero_rate:.3f} exceeds {thresholds.zero_rate}")
        if skew_val > thresholds.skew:
            flags.append(f"skew {skew_val:.2f} exceeds {thresholds.skew}")
        if kurt_val > thresholds.kurtosis:
            flags.append(f"kurtosis {kurt_val:.2f} exceeds {thresholds.kurtosis}")
        if p99_val > 0 and (max_val / p99_val) > thresholds.max_p99_ratio:
            flags.append(f"max/p99 ratio {max_val / p99_val:.1f} exceeds {thresholds.max_p99_ratio}")
        if n_unique <= min_unique:
            counts, edges = np.histogram(finite, bins=midpoint_bin_edges(finite))
            hists[name] = (counts, edges)
            features.append(QqFeature(
                feature=name, n_valid=n_valid, n_excluded=n_excluded,
                mean=mean_val, std=std_val, status="discrete",
                reason=f"discrete ({n_unique} unique values)", figure="",
                zero_rate=zero_rate, skew=skew_val, kurtosis=kurt_val,
                median=median_val, p99=p99_val, max=max_val,
                log_skew=log_skew_val, flags=flags,
            ))
            continue

        osm, osr, slope, intercept = _qq_points(finite, max_points_per_trace)
        traces[name] = (osm, osr, slope, intercept, zero_rate)
        counts, edges = np.histogram(finite, bins="auto")
        hists[name] = (counts, edges)
        features.append(QqFeature(
            feature=name, n_valid=n_valid, n_excluded=n_excluded,
            mean=mean_val, std=std_val, status="plotted",
            reason=None, figure="", zero_rate=zero_rate,
            skew=skew_val, kurtosis=kurt_val,
            median=median_val, p99=p99_val, max=max_val,
            log_skew=log_skew_val, flags=flags,
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
            fig_num, len(qq_chunks), fig_size_per_subplot, dpi, palette, n_rows,
            cfg.qq_zones,
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
            fig_num, len(dist_chunks), fig_size_per_subplot, dpi, palette, n_rows,
        )
        for n in chunk:
            i = name_to_idx[n]
            features[i] = features[i].model_copy(update={"dist_figure": dist_name})

    diag_rows = [
        (f.feature, f.skew, f.kurtosis, f.zero_rate)
        for f in features
        if f.status in ("plotted", "discrete")
        and f.skew is not None and f.kurtosis is not None and f.zero_rate is not None
    ]
    diagnostics_figures: list[str] = []
    if diag_rows:
        _plot_diagnostics_heatmap(
            diag_rows, figures_dir / cfg.diagnostics.figure, dpi, cfg.diagnostics
        )
        diagnostics_figures.append(f"figures/{cfg.diagnostics.figure}")

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
        diagnostics_figures=diagnostics_figures,
        standardization=standardization,
        sample_size=n_rows,
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