import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib

matplotlib.use("Agg")


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from _common import RESULTS, TARGET, load_sample

TOP_N = 15
CSV_OUT = RESULTS / "14_top_revenue_cells.csv"
PNG_OUT = RESULTS / "14_revenue_and_signatures.png"
MD_OUT = RESULTS / "14_demand_revenue_signatures.md"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_sample()

    df = df.copy()
    df["pickup_hour"] = pd.to_datetime(df["pickup_datetime"]).dt.hour

    agg = df.groupby(["pickup_location_id", "pickup_hour"]).agg(
        rides=(TARGET, "size"),
        revenue=(TARGET, "sum"),
    ).reset_index()

    # Top zones by total revenue, not ride count
    zone_rev = agg.groupby("pickup_location_id")["revenue"].sum().sort_values(ascending=False)
    top_zones = zone_rev.head(TOP_N).index

    sub = agg[agg["pickup_location_id"].isin(top_zones)]

    rev_pivot = sub.pivot(index="pickup_location_id", columns="pickup_hour", values="revenue")
    rev_pivot = rev_pivot.reindex(zone_rev.head(TOP_N).index)

    sig_pivot = sub.pivot(index="pickup_location_id", columns="pickup_hour", values="rides")
    sig_pivot = sig_pivot.div(sig_pivot.sum(axis=1), axis=0) * 100.0
    sig_pivot = sig_pivot.reindex(rev_pivot.index)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)

    sns.heatmap(rev_pivot / 1000.0, cmap="YlOrRd", ax=ax1,
                cbar_kws={"label": "Revenue ($k)"})
    ax1.set_title("Where the money is: Revenue by Zone × Hour")
    ax1.set_ylabel("Pickup Zone ID")
    ax1.set_xlabel("Hour")

    sns.heatmap(sig_pivot, cmap="YlOrRd", ax=ax2,
                cbar_kws={"label": "% of zone's daily rides"})
    ax2.set_title("When each zone peaks: Normalized temporal signature")
    ax2.set_ylabel("")
    ax2.set_xlabel("Hour")

    fig.savefig(PNG_OUT, dpi=150)
    plt.close(fig)
    print(f"wrote {PNG_OUT}")

    top_cells = (
        agg.sort_values("revenue", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    top_cells["revenue"] = top_cells["revenue"].round(0)
    top_cells.to_csv(CSV_OUT, index=False)
    print("--- Top 10 revenue cells (zone × hour) ---")
    print(top_cells.to_string(index=False))
    print(f"wrote {CSV_OUT}")

    md_lines = [
        "# 14: Revenue Heatmap + Temporal Signatures",
        "",
        "Raw ride counts hide the business signal: magnitude dominates and count is not money.",
        "",
        f"![Revenue and signatures]({PNG_OUT.name})",
        "",
        "- **Left:** revenue per zone × hour. The cells that actually fund the fleet.",
        "- **Right:** each zone row normalized to its own daily share. Size removed, behaviour exposed.",
        "",
        "## Top 10 revenue cells",
        "",
        top_cells.to_markdown(index=False),
    ]
    MD_OUT.write_text("\n".join(md_lines))
    print(f"wrote {MD_OUT}")


if __name__ == "__main__":
    main()
