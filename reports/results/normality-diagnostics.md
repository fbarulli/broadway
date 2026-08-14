# Normality diagnostics

## Question

Is the distributional shape problematic?

## What was run

check_normality

## What it found

- Manhattan.skew: 2.61
- Manhattan.kurtosis: 13
- Manhattan.shapiro_p: < 0.001
- Brooklyn.skew: 1.2
- Brooklyn.kurtosis: 1.24
- Brooklyn.shapiro_p: < 0.001
- Queens.skew: 0.841
- Queens.kurtosis: 1.52
- Queens.shapiro_p: < 0.001
- Bronx.skew: 1.14
- Bronx.kurtosis: 1.22
- Bronx.shapiro_p: < 0.001
- Staten Island.skew: 1.02
- Staten Island.kurtosis: 0.606
- Staten Island.shapiro_p: < 0.001
- standardization: per-group z-score

## Why it matters

distributional shape is skewed/heavy-tailed in some groups; consider this alongside sample size before choosing a method.

## Figures

![How to read: one plot per group; each group's standardized values are plotted against theoretical normal quantiles; points following the fitted reference line are approximately normal; curvature or heavy tails indicate non-normality.](../figures/normality_qq.png)
