import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from _common import RESULTS, TARGET, load_sample

CSV_OUT = RESULTS / "21_surge_waterfall.csv"
PNG_OUT = RESULTS / "21_surge_waterfall.png"
MD_OUT = RESULTS / "21_surge_waterfall.md"

GRIDLOCK_HOURS = [7, 8, 9, 16, 17, 18, 19]
SURGE_MULTIPLIER = 1.2
PRICE_ELASTICITY = -0.3


def fmt_m(v: float) -> str:
    return f"${v / 1e6:.2f}M"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_sample()

    df = df.copy()
    df["pickup_hour"] = pd.to_datetime(df["pickup_datetime"]).dt.hour
    rev_col = "total_amount" if "total_amount" in df.columns else TARGET

    grid_mask = df["pickup_hour"].isin(GRIDLOCK_HOURS)
    baseline_grid_rev = float(df.loc[grid_mask, rev_col].sum())
    baseline_non_grid_rev = float(df.loc[~grid_mask, rev_col].sum())
    baseline_total = baseline_grid_rev + baseline_non_grid_rev
    grid_rides = int(grid_mask.sum())

    vol_factor = max(0.0, 1.0 + (SURGE_MULTIPLIER - 1.0) * PRICE_ELASTICITY)
    lost_rides = int(round(grid_rides * (1.0 - vol_factor)))

    # Dollar decomposition
    premium_gain = baseline_grid_rev * vol_factor * (SURGE_MULTIPLIER - 1.0)
    volume_loss = baseline_grid_rev * (1.0 - vol_factor)
    simulated_total = baseline_total + premium_gain - volume_loss

    steps = pd.DataFrame([
        {"step": "Baseline Revenue", "value": baseline_total},
        {"step": "Surge Premium (retained rides)", "value": premium_gain},
        {"step": "Lost Ride Revenue (volume drop)", "value": -volume_loss},
        {"step": "Simulated Revenue", "value": simulated_total},
    ])
    steps.to_csv(CSV_OUT, index=False)
    print(steps.to_string(index=False))
    print(f"lost rides: {lost_rides:,}")
    print(f"wrote {CSV_OUT}")

    # --- Waterfall plot ---
    labels = ["Baseline\nRevenue", "Surge Premium\n(retained rides)", "Lost Rides\n(volume drop)", "Simulated\nRevenue"]
    heights = [baseline_total, premium_gain, volume_loss, simulated_total]
    bottoms = [0.0, baseline_total, baseline_total + premium_gain, 0.0]
    colors = ["#4c72b0", "#2ca02c", "#d62728", "#4c72b0"]

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    x = np.arange(len(labels))
    ax.bar(x, heights, bottom=bottoms, color=colors, width=0.6, edgecolor="black", linewidth=0.8)

    # Connector lines
    levels = [baseline_total, baseline_total + premium_gain, simulated_total]
    for i, lvl in enumerate(levels):
        ax.plot([x[i] + 0.3, x[i + 1] - 0.3], [lvl, lvl], color="gray", ls="--", lw=1)

    # Annotations
    tops = [baseline_total, baseline_total + premium_gain, baseline_total + premium_gain, simulated_total]
    texts = [
        fmt_m(baseline_total),
        f"+{fmt_m(premium_gain)}",
        f"-{fmt_m(volume_loss)}\n(-{lost_rides:,} rides)",
        fmt_m(simulated_total),
    ]
    for xi, top, text in zip(x, tops, texts):
        ax.text(xi, top + baseline_total * 0.015, text, ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fleet Revenue ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v / 1e6:.0f}M"))
    ax.set_title(f"Surge Mechanics: {SURGE_MULTIPLIER}x on Gridlock Hours (elasticity = {PRICE_ELASTICITY})")
    ax.grid(True, alpha=0.3, axis="y")

    fig.savefig(PNG_OUT, dpi=150)
    plt.close(fig)
    print(f"wrote {PNG_OUT}")

    md_lines = [
        "# 21: Surge Waterfall (The Mechanics)",
        "",
        f"Scenario: {SURGE_MULTIPLIER}x surge on gridlock hours {GRIDLOCK_HOURS}, elasticity {PRICE_ELASTICITY}.",
        "",
        f"![Waterfall]({PNG_OUT.name})",
        "",
        "## How to read it",
        "",
        f"- **Green bar:** extra dollars extracted from riders who still take the trip (+{fmt_m(premium_gain)}).",
        f"- **Red bar:** dollars lost from the {lost_rides:,} riders priced out (-{fmt_m(volume_loss)}).",
        f"- **Net effect:** {fmt_m(premium_gain - volume_loss)}. Surge wins when the green bar dwarfs the red bar.",
        "",
        "Break-even: when elasticity approaches -1.0, the green and red bars become equal and surge turns revenue-neutral.",
    ]
    MD_OUT.write_text("\n".join(md_lines))
    print(f"wrote {MD_OUT}")


if __name__ == "__main__":
    main()
