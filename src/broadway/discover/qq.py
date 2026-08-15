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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from scipy import stats

from broadway import viz
from broadway.config.viz import DiagnosticsConfig, QqMarkersConfig, QqZonesConfig, load_viz_config


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
    log_figure: str = ""
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
    log_figures: list[str] = []
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


def draw_qq_zones(ax, zones: QqZonesConfig, zero_rate: float | None, draw_shelf: bool = True) -> bool:
    """Draw central/tail bands and (optionally) the zero-mass shelf. Return True if a shelf was drawn."""
    if not zones.enabled:
        return False
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.axvspan(xmin, -zones.tail_threshold, color=zones.zone_color, alpha=zones.tail_alpha, zorder=0)
    ax.axvspan(zones.tail_threshold, xmax, color=zones.zone_color, alpha=zones.tail_alpha, zorder=0)
    ax.axvspan(stats.norm.ppf(zones.central_quantiles[0]), stats.norm.ppf(zones.central_quantiles[1]), color=zones.zone_color, alpha=zones.central_alpha, zorder=0)
    if draw_shelf and zero_rate is not None and zero_rate > zones.zero_mass_threshold and ymin <= 0 <= ymax:
        ax.axhline(y=0, color=zones.shelf_color, linestyle="--", linewidth=1, alpha=0.7, zorder=2)
        return True
    return False


def _draw_qq_markers(ax, osm: np.ndarray, osr: np.ndarray, markers: QqMarkersConfig) -> None:
    """Draw decision-mapped markers on a standardized Q-Q surface."""
    if not markers.enabled:
        return
    if markers.percentile_rings:
        for p in markers.percentiles:
            ax.plot(
                [stats.norm.ppf(p)],
                [np.quantile(osr, p)],
                marker="o",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor=markers.ring_color,
                markersize=markers.ring_size,
                zorder=4,
            )
    if markers.tail_highlight:
        mask = np.abs(osm) > markers.tail_threshold
        ax.scatter(
            osm[mask],
            osr[mask],
            facecolor="none",
            edgecolor=markers.tail_color,
            s=markers.tail_size,
            zorder=4,
        )
    if markers.robust_line:
        q1 = (stats.norm.ppf(0.25), np.quantile(osr, 0.25))
        q3 = (stats.norm.ppf(0.75), np.quantile(osr, 0.75))
        ax.plot(
            [q1[0], q3[0]],
            [q1[1], q3[1]],
            color=markers.robust_line_color,
            linestyle="-",
            linewidth=markers.robust_line_width,
            zorder=2,
        )


def build_qq_legend_handles(zones: QqZonesConfig, any_shelf: bool) -> list:
    mid = int((zones.central_quantiles[1] - zones.central_quantiles[0]) * 100)
    handles = [
        Patch(facecolor=zones.zone_color, alpha=zones.central_alpha, label=f"middle {mid}%"),
        Patch(facecolor=zones.zone_color, alpha=zones.tail_alpha, label=f"±{zones.tail_threshold}σ tails"),
    ]
    if any_shelf:
        handles.append(Line2D([0], [0], color=zones.shelf_color, linestyle="--", linewidth=1, label="zero-mass shelf"))
    return handles


def attach_qq_legend(fig, zones: QqZonesConfig, any_shelf: bool) -> None:
    if not zones.enabled:
        return
    handles = build_qq_legend_handles(zones, any_shelf)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=viz.TICK_FONTSIZE, frameon=False)


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
    markers: QqMarkersConfig,
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
    any_shelf = False
    for color, ax, (name, osm, osr, slope, intercept, zero_rate) in zip(colors, ax_flat, traces):
        ax.scatter(
            osm, osr,
            s=viz.QQ_SCATTER_SIZE,
            alpha=viz.QQ_SCATTER_ALPHA,
            edgecolor=viz.QQ_SCATTER_EDGE_COLOR,
            color=color,
            zorder=3,
        )
        xs = np.array([osm.min(), osm.max()])
        ax.plot(
            xs, slope * xs + intercept,
            color=viz.QQ_REF_LINE_COLOR,
            linestyle=viz.QQ_REF_LINE_STYLE,
            linewidth=viz.QQ_REF_LINE_WIDTH,
            zorder=2,
        )
        any_shelf = draw_qq_zones(ax, zones, zero_rate) or any_shelf
        _draw_qq_markers(ax, osm, osr, markers)
        ax.set_xlabel(viz.QQ_XLABEL, fontsize=viz.LABEL_FONTSIZE)
        ax.set_ylabel(viz.QQ_YLABEL, fontsize=viz.LABEL_FONTSIZE)
        ax.set_title(name, fontsize=viz.TITLE_FONTSIZE)
        ax.grid(True, alpha=viz.GRID_ALPHA)
        ax.tick_params(labelsize=viz.TICK_FONTSIZE)
        viz.despine(ax)
    for ax in ax_flat[n:]:
        ax.set_visible(False)
    attach_qq_legend(fig, zones, any_shelf)
    title = "Per-feature Q-Q plots (standardized)"
    if n_figs > 1:
        title += f" - figure {fig_num} of {n_figs}"
    title += f" — n = {n_rows:,}"
    fig.suptitle(title, fontsize=viz.SUPTITLE_FONTSIZE)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_raw_log_pairs(
    pairs: list[tuple[str, tuple, tuple, float, float]],
    out_path: Path,
    fig_num: int,
    n_figs: int,
    subplot_size: float,
    dpi: int,
    palette: str,
    n_rows: int,
    zones: QqZonesConfig,
    markers: QqMarkersConfig,
    title_prefix: str,
) -> None:
    n = len(pairs)
    colors = viz.palette_colors(n, palette)
    fig, axes = plt.subplots(
        n, 2,
        figsize=(2 * subplot_size, n * subplot_size),
        sharex="row",
        sharey="row",
        squeeze=False,
        layout="constrained",
    )
    axes[0][0].set_title("raw", fontsize=viz.TITLE_FONTSIZE)
    axes[0][1].set_title("log", fontsize=viz.TITLE_FONTSIZE)
    any_shelf = False
    for i, (name, raw_trace, log_trace, raw_skew, log_skew) in enumerate(pairs):
        axes[i][0].text(
            -0.3, 0.5, name, va="center", ha="center", rotation=90,
            transform=axes[i][0].transAxes, fontsize=viz.TITLE_FONTSIZE,
        )
        for col, (trace, log_panel) in enumerate(((raw_trace, False), (log_trace, True))):
            osm, osr, slope, intercept = trace
            ax = axes[i][col]
            ax.scatter(
                osm, osr,
                s=viz.QQ_SCATTER_SIZE,
                alpha=viz.QQ_SCATTER_ALPHA,
                edgecolor=viz.QQ_SCATTER_EDGE_COLOR,
                color=colors[i],
                zorder=3,
            )
            xs = np.array([osm.min(), osm.max()])
            ax.plot(
                xs, slope * xs + intercept,
                color=viz.QQ_REF_LINE_COLOR,
                linestyle=viz.QQ_REF_LINE_STYLE,
                linewidth=viz.QQ_REF_LINE_WIDTH,
                zorder=2,
            )
            any_shelf = draw_qq_zones(ax, zones, None, draw_shelf=False) or any_shelf
            _draw_qq_markers(ax, osm, osr, markers)
            if log_panel:
                ax.text(
                    0.03, 0.97, f"skew {raw_skew:.2f} → {log_skew:.2f}",
                    transform=ax.transAxes, va="top", fontsize=viz.TICK_FONTSIZE,
                )
            ax.set_xlabel(viz.QQ_XLABEL, fontsize=viz.LABEL_FONTSIZE)
            ax.set_ylabel(viz.QQ_YLABEL, fontsize=viz.LABEL_FONTSIZE)
            ax.grid(True, alpha=viz.GRID_ALPHA)
            ax.tick_params(labelsize=viz.TICK_FONTSIZE)
            viz.despine(ax)
    attach_qq_legend(fig, zones, any_shelf)
    title = title_prefix
    if n_figs > 1:
        title += f" - figure {fig_num} of {n_figs}"
    title += f" — n = {n_rows:,}"
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
    log_pairs: dict[str, tuple[tuple, tuple, float, float]] = {}
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
        log_vals = None
        log_skew_val = None
        if skew_val > thresholds.skew and finite.min() > 0:
            log_vals = np.log(finite)
            log_skew_val = float(stats.skew(log_vals))
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
        if log_vals is not None:
            log_osm, log_osr, log_slope, log_intercept = _qq_points(log_vals, max_points_per_trace)
            log_pairs[name] = (
                (osm, osr, slope, intercept),
                (log_osm, log_osr, log_slope, log_intercept),
                skew_val,
                log_skew_val,
            )
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
            cfg.qq_markers,
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

    log_figures: list[str] = []
    if log_pairs:
        log_names = list(log_pairs.keys())
        log_chunks = [
            log_names[i : i + max_features_per_figure]
            for i in range(0, len(log_names), max_features_per_figure)
        ]
        for fig_num, chunk in enumerate(log_chunks, start=1):
            basename = cfg.qq_log_figure.format(fig_num=fig_num)
            log_name = f"figures/{basename}"
            log_figures.append(log_name)
            _plot_raw_log_pairs(
                [(n, *log_pairs[n]) for n in chunk],
                figures_dir / basename,
                fig_num, len(log_chunks), fig_size_per_subplot, dpi, palette, n_rows,
                cfg.qq_zones,
                cfg.qq_markers,
                title_prefix="Per-feature Q-Q (raw vs log-transformed)",
            )
            for n in chunk:
                i = name_to_idx[n]
                features[i] = features[i].model_copy(update={"log_figure": log_name})

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
        log_figures=log_figures,
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