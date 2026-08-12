"""
Step 5: Check ANOVA's assumptions before trusting the last step's result.

1. Homogeneity of variance -> Levene's test
2. Normality within each group -> skew/kurtosis + Shapiro (on a subsample)

Run with: python learning/stats/05_anova_assumptions.py
"""
from scipy import stats
import numpy as np

from _config import (
    get_spark_session, load_borough_durations,
)

spark = get_spark_session()

groups = load_borough_durations(spark)

# --- Assumption 1: Homogeneity of variance (Levene's test) ---
# H0: all groups have equal variance. Unlike ANOVA, Levene's test is
# robust even when the data isn't normal - good, since we're about to
# check normality separately and don't want to assume it yet.
print("=== Variance per group ===")
for name, vals in groups.items():
    print(f"{name}: variance={vals.var():.2f}, std={vals.std():.2f}")

levene_stat, levene_p = stats.levene(*groups.values())
print(f"\nLevene's test: stat={levene_stat:.2f}, p={levene_p:.6e}")
if levene_p < 0.05:
    print("-> p < 0.05: variances are significantly different across groups.")
    print("   Homogeneity of variance is VIOLATED. Standard ANOVA's F-test")
    print("   is not reliable here - use Welch's ANOVA instead (next step).")
else:
    print("-> p >= 0.05: no evidence variances differ. Assumption holds.")

# --- Assumption 2: Normality within each group ---
# Shapiro-Wilk needs a small sample (it's overly sensitive / breaks down
# on large n - even trivial deviations from normal get p<0.05 past a few
# thousand rows). We check skew/kurtosis on the full group, and run
# Shapiro on a small random subsample just to see the shape.
print("\n=== Normality per group ===")
for name, vals in groups.items():
    skewness = stats.skew(vals)
    kurt = stats.kurtosis(vals)
    subsample = np.random.choice(vals, size=min(500, len(vals)), replace=False)
    shapiro_stat, shapiro_p = stats.shapiro(subsample)
    print(f"{name}: skew={skewness:.2f}, kurtosis={kurt:.2f}, "
          f"Shapiro p (n=500 subsample)={shapiro_p:.4f}")

spark.stop()
