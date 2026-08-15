"""Experiment: Q-Q legend placement (below vs above) across all plot surfaces.

Throwaway — monkeypatches attach_qq_legend and regenerates every plot the
pipeline produces, into legend_experiment/<mode>/. No source changes.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import broadway.discover.qq as qq
from broadway import viz
from broadway.config.viz import load_viz_config
from broadway.stats.describe import describe, plot_describe_figures

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "legend_experiment"

TRAINING = ROOT / "data" / "processed" / "training_data.parquet"
SAMPLE = ROOT / "data" / "processed" / "joined_sample_live.parquet"
EXCLUDE = ["pickup_location_id", "dropoff_location_id"]
GROUP_COLUMN = "Borough"
SOURCE_GROUP = "pickup_borough"
GROUP_VALUES = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
TARGET = "trip_duration_minutes"


def patched_legend_factory(mode):
    def patched(fig, zones, any_shelf, markers=None):
        handles = qq.build_qq_legend_handles(zones, any_shelf, markers)
        if not handles:
            return
        fig.set_layout_engine(None)
        fig_w = fig.get_size_inches()[0]
        legend_w = 1.7
        frac = min(legend_w / fig_w, 0.4)
        fig.subplots_adjust(right=1.0 - frac)
        band = fig.add_axes([1.0 - frac, 0.0, frac, 1.0])
        band.axis("off")
        band.legend(handles=handles, loc="center", ncol=1,
                    frameon=False, fontsize=viz.TICK_FONTSIZE)
    return patched


def render_features(df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    qq.plot_numeric_qq(
        df, out_dir, out_dir / "qq_overview.json",
        source_path=str(TRAINING), exclude=EXCLUDE,
    )


def render_groups(sample_df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_cfg = load_viz_config()
    groups = {
        g: sample_df[sample_df[SOURCE_GROUP] == g][TARGET].dropna().to_numpy()
        for g in GROUP_VALUES
    }
    pooled = np.concatenate(list(groups.values()))
    show_log = bool(
        pooled.size > 1
        and pooled.min() > 0
        and float(stats.skew(pooled)) > viz_cfg.diagnostics.thresholds.skew
    )
    qq._plot_qq_joint(
        groups, out_dir / viz_cfg.normality_figure, None,
        show_log=show_log, markers=viz_cfg.qq_markers,
    )


def render_describe(sample_df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = describe(
        sample_df, GROUP_COLUMN, SOURCE_GROUP, GROUP_VALUES, TARGET,
        str(SAMPLE), "taxi_diagnostic", "diagnostic",
    )
    plot_describe_figures(
        sample_df, SOURCE_GROUP, GROUP_COLUMN, GROUP_VALUES, TARGET, summary,
        out_dir / load_viz_config().describe_figure,
    )


def main():
    training = pd.read_parquet(TRAINING)
    sample = pd.read_parquet(SAMPLE)
    for mode in ("right",):
        qq.attach_qq_legend = patched_legend_factory(mode)
        out = OUT / mode
        print(f"rendering {mode} -> {out}")
        render_features(training, out)
        render_groups(sample, out)
        render_describe(sample, out)
    print("done")


if __name__ == "__main__":
    main()
