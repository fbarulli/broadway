"""22: marginal diagnostics + modified z-score outliers on the working scatter.

One 2x2 figure: fare vs trip_distance with marginal histograms / box plots /
KDE / rug plots along the axes, and modified-z-score outliers (|M| > 3.5,
median + MAD, robust to extremes) highlighted in red on each scatter.
Outlier counts and flagged trips persist to the working dataset's JSON.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from _common import RESULTS, WORKING_DATASET, load_working
from _ols_bp import modified_zscore
from _tests import write_tests_json

OUT = RESULTS / f"{Path(__file__).stem}.png"

THRESHOLD = 10.0
MARGINALS = {
    "histograms": "hist",
    "box plots": "box",
    "KDE": "kde",
    "rug plots": "rug",
}


def draw_scatter(ax, df, outliers) -> None:
    ax.scatter(df["trip_distance"], df["fare_amount"], s=5, alpha=0.1, color="gray")
    ax.scatter(outliers["trip_distance"], outliers["fare_amount"], s=20, color="red", edgecolor="black",
               label=f"outliers |M|>{THRESHOLD}")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("trip_distance (miles)")
    ax.set_ylabel("fare_amount ($)")


def draw_marginal(ax_x, ax_y, x, y, kind: str) -> None:
    if kind == "hist":
        ax_x.hist(x, bins=60, color="gray", alpha=0.6)
        ax_y.hist(y, bins=60, orientation="horizontal", color="gray", alpha=0.6)
    elif kind == "box":
        ax_x.boxplot(x, vert=False, showfliers=True)
        ax_y.boxplot(y, vert=True, showfliers=True)
    elif kind == "kde":
        xs = np.linspace(x.min(), x.max(), 300)
        ax_x.plot(xs, gaussian_kde(x)(xs), color="gray")
        ys = np.linspace(y.min(), y.max(), 300)
        ax_y.plot(gaussian_kde(y)(ys), ys, color="gray")
    elif kind == "rug":
        ax_x.eventplot(x, orientation="horizontal", colors="gray")
        ax_y.eventplot(y, orientation="vertical", colors="gray")
    ax_x.set_xticks([])
    ax_x.set_yticks([])
    ax_y.set_xticks([])
    ax_y.set_yticks([])


def main() -> None:
    df = load_working()
    x = df["trip_distance"].to_numpy()
    y = df["fare_amount"].to_numpy()

    z_dist = modified_zscore(df["trip_distance"])
    z_fare = modified_zscore(df["fare_amount"])
    out_dist = z_dist.abs() > THRESHOLD
    out_fare = z_fare.abs() > THRESHOLD
    outlier_mask = out_dist | out_fare
    outliers = df[outlier_mask]
    print(f"modified-z outliers (|M|>{THRESHOLD}): distance={int(out_dist.sum())}, "
          f"fare={int(out_fare.sum())}, union={int(outlier_mask.sum())} of {len(df)}")

    fig = plt.figure(figsize=(14, 12))
    outer = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)
    for i, (label, kind) in enumerate(MARGINALS.items()):
        inner = outer[i // 2, i % 2].subgridspec(
            2, 2, height_ratios=[1, 4], width_ratios=[4, 1], hspace=0.06, wspace=0.06
        )
        ax_main = fig.add_subplot(inner[1, 0])
        ax_marg_x = fig.add_subplot(inner[0, 0], sharex=ax_main)
        ax_marg_y = fig.add_subplot(inner[1, 1], sharey=ax_main)
        draw_scatter(ax_main, df, outliers)
        draw_marginal(ax_marg_x, ax_marg_y, x, y, kind)
        ax_main.set_title(f"marginal {label}", fontsize=11)
    fig.suptitle(f"Marginal diagnostics — fare vs trip_distance (working dataset, N={len(df)})", fontsize=13)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}")

    results = {
        "modified_zscore_outliers": {
            "method": "M = 0.6745 * (x - median) / MAD",
            "threshold": THRESHOLD,
            "n_trip_distance": int(out_dist.sum()),
            "n_fare_amount": int(out_fare.sum()),
            "n_union": int(outlier_mask.sum()),
            "top_flagged_trips": (
                df.loc[outlier_mask, ["trip_distance", "fare_amount"]]
                .assign(
                    z_dist=z_dist[outlier_mask].abs(),
                    z_fare=z_fare[outlier_mask].abs(),
                )
                .assign(max_abs_z=lambda d: d[["z_dist", "z_fare"]].max(axis=1))
                .sort_values("max_abs_z", ascending=False)
                .head(50)
                .to_dict("records")
            ),
        }
    }
    out = write_tests_json(
        WORKING_DATASET,
        results,
        "22_ratecode1_marginal_outliers.py",
        n_rows=len(pd.read_parquet(WORKING_DATASET)),
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
