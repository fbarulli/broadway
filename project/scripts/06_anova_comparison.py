"""
Step 6: Compare four ways of testing "do boroughs differ in trip duration",
given that we found unequal variance AND non-normal data.

1. Standard (Fisher's) ANOVA        - what we ran originally, assumptions violated
2. Log-transformed + standard ANOVA - fixes skew, may fix variance too
3. Welch's ANOVA                    - doesn't assume equal variance
4. Kruskal-Wallis                   - doesn't assume normality OR equal variance

Run with: python project/scripts/06_anova_comparison.py
"""
from project import data
from broadway.stats import anova


def main() -> None:
    groups = {k: v for k, v in data.load_borough_durations().items() if len(v) > 0}

    print("Group sizes:", {k: len(v) for k, v in groups.items()})
    print()

    plans = {
        "Standard ANOVA": anova.run_anova(groups),
        "Welch's ANOVA": anova.run_welch(groups),
        "Kruskal-Wallis": anova.run_kruskal(groups),
    }

    print(f"{'Test':20s} {'statistic':>12s} {'p':>12s}  verdict")
    print("-" * 62)
    for name, plan in plans.items():
        stat = plan.statistics["statistic"]
        p_value = plan.statistics["p_value"]
        verdict = "significant" if p_value < 0.05 else "NOT significant"
        print(f"{name:20s} {stat:>12.2f} {p_value:>12.4e}  -> {verdict}")

    print()
    for name, plan in plans.items():
        print(f"{name}:")
        for line in plan.reason:
            print(f"  {line}")
        for warning in plan.warnings:
            print(f"  warning: {warning}")
        print()

    print("=== Summary ===")
    for name, plan in plans.items():
        p_value = plan.statistics["p_value"]
        verdict = "significant" if p_value < 0.05 else "NOT significant"
        print(f"  {name:20s} p={p_value:.4e}  -> {verdict}")


if __name__ == "__main__":
    main()
