"""
Step 4: ANOVA - does trip duration differ across boroughs?

A t-test compares 2 groups. ANOVA compares 3+ groups at once, asking
one question: "is at least one group's mean different from the others?"
It does NOT tell you which one(s) - that needs a post-hoc test (next step).

Run with: python projects/taxi/scripts/04_anova_boroughs.py
"""
from projects.taxi import data
from broadway.stats import anova


def main() -> None:
    groups = {k: v for k, v in data.load_borough_durations().items() if len(v) > 0}

    for borough, values in groups.items():
        print(f"{borough}: sampled_n={len(values)}, mean={values.mean():.2f}")

    print()

    plan = anova.run_anova(groups)

    stat = plan.statistics["statistic"]
    p_value = plan.statistics["p_value"]
    eta_squared = plan.effect_sizes["eta_squared"]
    omega_squared = plan.effect_sizes["omega_squared"]

    print("=== One-way ANOVA ===")
    print(f"F-statistic:   {stat:.2f}")
    print(f"p-value:       {p_value:.6e}")
    print(f"eta-squared:   {eta_squared:.4f}")
    print(f"omega-squared: {omega_squared:.4f}")
    print(f"passed:        {plan.passed}")

    print("\nReason:")
    for line in plan.reason:
        print(f"  {line}")

    if plan.warnings:
        print("\nWarnings:")
        for warning in plan.warnings:
            print(f"  - {warning}")

    print()
    if p_value < 0.05:
        print("-> p < 0.05: reject the null hypothesis.")
        print("   At least one borough's average trip duration is significantly")
        print("   different from the others. ANOVA doesn't say which one(s) -")
        print("   that needs a post-hoc test (e.g. Games-Howell) as a follow-up.")
    else:
        print("-> p >= 0.05: no significant difference detected across boroughs.")


if __name__ == "__main__":
    main()
