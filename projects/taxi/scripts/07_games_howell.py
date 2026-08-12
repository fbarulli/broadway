"""
Step 7: Games-Howell post-hoc test - which specific boroughs differ?

ANOVA/Welch/Kruskal only tell us "at least one group differs." Games-Howell
does all pairwise comparisons (Manhattan vs Brooklyn, Manhattan vs Queens,
etc.) while correcting for multiple comparisons AND not assuming equal
variance or equal sample sizes - which fits our data (Levene's test failed,
group sizes range from 84 to 155,502).

Run with: python projects/taxi/scripts/07_games_howell.py
"""
import pandas as pd

from projects.taxi import data
from broadway.stats import post_hoc


def main() -> None:
    groups = data.load_borough_durations()

    rows = []
    for borough, values in groups.items():
        for v in values:
            rows.append({data.PICKUP_BOROUGH_COL: borough, data.DURATION_COL: v})

    df = pd.DataFrame(rows)
    print(f"Total rows across all groups: {len(df)}")
    print(df.groupby(data.PICKUP_BOROUGH_COL)[data.DURATION_COL].agg(["count", "mean", "std"]))
    print()

    # pingouin's Games-Howell: pairwise comparisons, Welch-Satterthwaite df,
    # studentized range distribution, no equal-variance/equal-n assumption
    result = post_hoc.games_howell(
        df, dv=data.DURATION_COL, between=data.PICKUP_BOROUGH_COL
    )
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", None)
    print("=== Games-Howell pairwise comparisons ===")
    print(result.round(4))

    print("\n=== Plain-English summary ===")
    for _, row in result.iterrows():
        sig = "DIFFERENT" if row["pval"] < 0.05 else "not significantly different"
        print(f"{row['A']} vs {row['B']}: mean diff={row['diff']:.2f} min, p={row['pval']:.4f} -> {sig}")


if __name__ == "__main__":
    main()
