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

CSV_OUT = RESULTS / "19_surge_sensitivity.csv"
PNG_OUT = RESULTS / "19_surge_sensitivity.png"
MD_OUT = RESULTS / "19_surge_sensitivity.md"

GRIDLOCK_HOURS = [7, 8, 9, 16, 17, 18, 19]
MULTIPLIERS = np.arange(1.0, 2.01, 0.05)
ELASTICITIES = [-0.2, -0.3, -0.4, -0.6, -0.8, -1.0, -1.2]
HIGHLIGHT_E = -0.3  # our working assumption


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_sample()

    df = df.copy()
    df["pickup_hour"] = pd.to_datetime(df["pickup_datetime"]).dt.hour
    rev_col = "total_amount" if "total_amount" in df.columns else TARGET

    hourly = df.groupby("pickup_hour")[rev_col].sum()
    base_total = float(hourly.sum())
    grid_total = float(hourly[hourly.index.isin(GRIDLOCK_HOURS)].sum())
    non_grid_total = base_total - grid_total
    grid_share = grid_total / base_total

    print(f"Gridlock hours revenue share: {grid_share * 100:.1f}% of fleet revenue")

    rows = []
    for e in ELASTICITIES:
        for m in MULTIPLIERS:
            vol = max(0.0, 1.0 + (m - 1.0) * e)  # volume factor, floored at 0
            sim_rev = non_grid_total + grid_total * m * vol
            sim_rides_share = (1.0 - grid_share) + grid_share * vol
            rows.append({
                "elasticity": e,
                "multiplier": round(float(m), 2),
                "revenue_pct_change": (sim_rev / base_total - 1.0) * 100.0,
                "rides_pct_change": (sim_rides_share - 1.0) * 100.0,
            })
    sweep = pd.DataFrame(rows)
    sweep.to_csv(CSV_OUT, index=False)
    print(f"wrote {CSV_OUT}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)

    for e in ELASTICITIES:
        sub = sweep[sweep["elasticity"] == e]
        highlight = (e == HIGHLIGHT_E)
        kwargs = dict(
            linewidth=3.0 if highlight else 1.2,
            color="#d62728" if highlight else None,
            alpha=1.0 if highlight else 0.6,
            label=f"elasticity = {e}" + (" (assumed)" if highlight else ""),
        )
        ax1.plot(sub["multiplier"], sub["revenue_pct_change"], marker="o", ms=3, **kwargs)
        ax2.plot(sub["multiplier"], sub["rides_pct_change"], marker="o", ms=3, **kwargs)

    for ax, ylabel, title in (
        (ax1, "Change in fleet revenue (%)", "Revenue impact of gridlock-hour surge"),
        (ax2, "Change in fleet rides (%)", "Ride volume impact of gridlock-hour surge"),
    ):
        ax.axhline(0, color="black", lw=1.0)
        ax.set_xlabel("Surge multiplier (applied to gridlock hours only)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.savefig(PNG_OUT, dpi=150)
    plt.close(fig)
    print(f"wrote {PNG_OUT}")

    md_lines = [
        "# 19: Surge Sensitivity Analysis",
        "",
        f"Surge applied only to gridlock hours {GRIDLOCK_HOURS}, which carry {grid_share * 100:.1f}% of fleet revenue.",
        "",
        "Each line is a demand-elasticity assumption. The thick red line (e = -0.3) is our working assumption.",
        "",
        f"![Sensitivity]({PNG_OUT.name})",
        "",
        "## How to read it",
        "",
        "- Lines above the zero line = surge increases fleet revenue.",
        "- The steeper the elasticity, the sooner surge destroys revenue.",
        "- At e = -1.0, every price gain is exactly cancelled by volume loss (revenue-neutral).",
        "- Below e = -1.0, surging is self-defeating even at 1.2x.",
    ]
    MD_OUT.write_text("\n".join(md_lines))
    print(f"wrote {MD_OUT}")


if __name__ == "__main__":
    main()
