"""
Step 5: Check ANOVA's assumptions before trusting the last step's result.

1. Homogeneity of variance -> Levene's test
2. Normality within each group -> skew/kurtosis + Shapiro (on a subsample)

Run with: python project/scripts/05_anova_assumptions.py
"""
from project import data
from broadway.stats import assumptions


def main() -> None:
    groups = {k: v for k, v in data.load_borough_durations().items() if len(v) > 0}

    # --- Assumption 1: Homogeneity of variance (Levene's test) ---
    # H0: all groups have equal variance. Unlike ANOVA, Levene's test is
    # robust even when the data isn't normal - good, since we're about to
    # check normality separately and don't want to assume it yet.
    print("=== Variance per group ===")
    for name, vals in groups.items():
        print(f"{name}: variance={vals.var():.2f}, std={vals.std():.2f}")

    levene = assumptions.run_levene(groups)
    levene_stat = levene["statistic"]
    levene_p = levene["p_value"]
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
    normality = assumptions.check_normality(groups)
    for name, res in normality.items():
        print(f"{name}: skew={res['skew']:.2f}, kurtosis={res['kurtosis']:.2f}, "
              f"Shapiro p (n=5000 subsample)={res['shapiro_p']:.4f}")


if __name__ == "__main__":
    main()
