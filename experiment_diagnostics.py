"""Experiment: distribution-diagnostics redesign.

Renders the diagnostics surface three ways into diagnostics_experiment/<variant>/:
  - zscore/  : current per-column z-score heatmap (control, produced by plot_numeric_qq)
  - ratio/   : value/threshold heatmap, clipped [0, 2], YlOrRd sequential colormap
  - bars/    : four per-metric sorted horizontal bar panels (red = exceeds threshold)
Uses the right-side legend (baseline) for the Q-Q figures generated as a side effect.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd

import broadway.discover.qq as qq
from broadway import viz
from broadway.config.viz import load_viz_config

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "diagnostics_experiment"

TRAINING = ROOT / "data" / "processed" / "training_data.parquet"
EXCLUDE = ["pickup_location_id", "dropoff_location_id"]

METRICS = ["skew", "kurtosis", "zero_rate", "max/p99"]


def right_side_legend(fig, zones, any_shelf, markers=None):
    handles = qq.build_qq_legend_handles(zones, any_shelf, markers)
    if not handles:
        return
    fig.set_layout_engine(None)
    fig_w = fig.get_size_inches()[0]
    frac = min(1.7 / fig_w, 0.4)
    fig.subplots_adjust(right=1.0 - frac)
    band = fig.add_axes([1.0 - frac, 0.0, frac, 1.0])
    band.axis("off")
    band.legend(handles=handles, loc="center", ncol=1,
                frameon=False, fontsize=viz.TICK_FONTSIZE)


def diag_rows(overview):
    rows = []
    for f in overview.features:
        if f.status not in ("plotted", "discrete"):
            continue
        if f.skew is None or f.kurtosis is None or f.zero_rate is None:
            continue
        max_p99 = (
            f.max / f.p99
            if f.p99 is not None and f.max is not None and f.p99 > 0
            else None
        )
        rows.append((f.feature, f.skew, f.kurtosis, f.zero_rate, max_p99))
    return rows


def thresholds():
    t = load_viz_config().diagnostics.thresholds
    return [t.skew, t.kurtosis, t.zero_rate, t.max_p99_ratio]


def raw_matrix(rows):
    m = []
    for name, sk, ku, zr, mp in rows:
        m.append([sk, ku, zr, mp if mp is not None else np.nan])
    return np.array(m, dtype=float)


def plot_ratio_heatmap(rows, out_path, thr, dpi=100):
    names = [r[0] for r in rows]
    raw = raw_matrix(rows)
    ratio = np.clip(raw / np.array(thr), 0.0, 2.0)
    fig, ax = plt.subplots(
        figsize=(6.0, max(2.0, 0.35 * len(rows))), layout="constrained",
    )
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=2.0)
    ax.imshow(ratio, aspect="auto", cmap="YlOrRd", norm=norm)
    ax.set_xticks(range(raw.shape[1]))
    ax.set_xticklabels(METRICS, fontsize=viz.TICK_FONTSIZE)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names, fontsize=viz.TICK_FONTSIZE)
    for i in range(len(rows)):
        for j in range(raw.shape[1]):
            val = raw[i, j]
            if np.isnan(val):
                continue
            text_color = "black" if ratio[i, j] < 1.0 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=viz.TICK_FONTSIZE, color=text_color)
    fig.colorbar(ax.images[0], ax=ax, label="value / threshold (1.0 = flagging boundary)")
    viz.despine(ax)
    fig.suptitle(
        "Per-feature diagnostics — value / threshold (capped at 2.0)",
        fontsize=viz.SUPTITLE_FONTSIZE,
    )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_bars(rows, out_path, thr, dpi=100):
    n = len(rows)
    fig, axes = plt.subplots(
        4, 1, figsize=(8.0, max(6.0, 0.6 * n + 2.0)), layout="constrained",
    )
    for ax, (label, idx, t) in zip(axes, [
        ("skew", 1, thr[0]),
        ("kurtosis", 2, thr[1]),
        ("zero_rate", 3, thr[2]),
        ("max/p99", 4, thr[3]),
    ]):
        valid = [(r[0], r[idx]) for r in rows if r[idx] is not None]
        valid.sort(key=lambda x: x[1], reverse=True)
        names = [v[0] for v in valid]
        vals = [v[1] for v in valid]
        colors = ["#d62728" if v > t else "#bbbbbb" for v in vals]
        y = np.arange(len(names))
        ax.barh(y, vals, color=colors, height=0.7)
        ax.axvline(t, color="#333333", linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=viz.TICK_FONTSIZE)
        ax.invert_yaxis()
        ax.set_xlabel(f"{label} (threshold {t})", fontsize=viz.LABEL_FONTSIZE)
        ax.tick_params(labelsize=viz.TICK_FONTSIZE)
        viz.despine(ax)
    fig.suptitle(
        "Per-feature diagnostics — sorted by metric, red = exceeds threshold",
        fontsize=viz.SUPTITLE_FONTSIZE,
    )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    viz_cfg = load_viz_config()
    thr = thresholds()
    training = pd.read_parquet(TRAINING)

    qq.attach_qq_legend = right_side_legend
    overview = qq.plot_numeric_qq(
        training, OUT / "zscore", OUT / "zscore" / "qq_overview.json",
        source_path=str(TRAINING), exclude=EXCLUDE,
    )
    rows = diag_rows(overview)

    (OUT / "ratio").mkdir(parents=True, exist_ok=True)
    (OUT / "bars").mkdir(parents=True, exist_ok=True)

    plot_ratio_heatmap(rows, OUT / "ratio" / "diagnostics_ratio.png", thr, dpi=viz_cfg.dpi)
    plot_bars(rows, OUT / "bars" / "diagnostics_bars.png", thr, dpi=viz_cfg.dpi)
    print("done")


if __name__ == "__main__":
    main()
