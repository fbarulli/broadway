"""
Step 7: Games-Howell post-hoc test - which specific boroughs differ?

ANOVA/Welch/Kruskal only tell us "at least one group differs." Games-Howell
does all pairwise comparisons (Manhattan vs Brooklyn, Manhattan vs Queens,
etc.) while correcting for multiple comparisons AND not assuming equal
variance or equal sample sizes - which fits our data (Levene's test failed,
group sizes range from 84 to 155,502).

Run with: python learning/stats/07_games_howell.py
"""
from pyspark.sql import functions as F
import pandas as pd
import pingouin as pg

from _config import (
    BOROUGHS, MIN_ROWS_FOR_SAMPLING, RANDOM_STATE, SAMPLE_FRACTION,
    PICKUP_BOROUGH_COL, DURATION_COL,
    get_spark_session, _load_boroughs,
)

spark = get_spark_session()

trips_with_borough = _load_boroughs(spark)

rows = []
for borough in BOROUGHS:
    borough_df = trips_with_borough.filter(F.col(PICKUP_BOROUGH_COL) == borough)
    total = borough_df.count()
    if total > MIN_ROWS_FOR_SAMPLING:
        sampled = borough_df.select(DURATION_COL).sample(fraction=SAMPLE_FRACTION, seed=RANDOM_STATE).collect()
    else:
        sampled = borough_df.select(DURATION_COL).collect()
    for r in sampled:
        rows.append({PICKUP_BOROUGH_COL: borough, DURATION_COL: r[DURATION_COL]})

df = pd.DataFrame(rows)
print(f"Total rows across all groups: {len(df)}")
print(df.groupby(PICKUP_BOROUGH_COL)[DURATION_COL].agg(["count", "mean", "std"]))
print()

# pingouin's Games-Howell: pairwise comparisons, Welch-Satterthwaite df,
# studentized range distribution, no equal-variance/equal-n assumption
result = pg.pairwise_gameshowell(data=df, dv=DURATION_COL, between=PICKUP_BOROUGH_COL)
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", None)
print("=== Games-Howell pairwise comparisons ===")
print(result.round(4))

print("\n=== Plain-English summary ===")
for _, row in result.iterrows():
    sig = "DIFFERENT" if row["pval"] < 0.05 else "not significantly different"
    print(f"{row['A']} vs {row['B']}: mean diff={row['diff']:.2f} min, p={row['pval']:.4f} -> {sig}")

spark.stop()
